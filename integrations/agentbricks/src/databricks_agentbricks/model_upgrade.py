"""The engine behind `agentbricks models upgrade`: evaluate candidate models on an agent's own traces.

Replays the agent's recent production requests (root ``invoke`` spans from its bound trace
experiment) through the project's own agent code, once per candidate model, and scores each answer
against what production returned with an LLM judge. The search itself is Smart Model Upgrades'
``optimize_prompts_and_models`` (GEPA), which routes the agent's calls to a temporary ``<service>_exp``
clone of the bound model service, so production traffic is never touched while candidates run.

Heavy dependencies (``smart_model_upgrades``, ``gepa``, full ``mlflow``) are imported lazily and only
here, behind the ``upgrade`` extra, so the rest of the CLI stays light.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import pathlib
import re
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_types import AgentFramework
from databricks_agentkit.runtime.tool_manifest import MODEL_SERVICE_ENV

UPGRADE_EXTRA_HINT = "Install it with: pip install 'databricks-agentbricks[upgrade]'"
# Upgrade-run history (for `models status` / `models rollback`), next to the project metadata.
HISTORY_PATH = pathlib.Path(".agentbricks/model_upgrades.json")
DEFAULT_JUDGE_MODEL = "databricks-claude-sonnet-4-6"
# Below this many usable traces the train/val split is too small to say anything.
MIN_RECORDS = 5

_SYSTEM_AI = "system.ai."
_JUDGE_PROMPT = """You are grading an AI agent's answer against a reference answer that the agent's \
current production model gave for the same request.

Request:
{request}

Reference answer (production):
{reference}

Candidate answer:
{candidate}

How well does the candidate answer the request compared to the reference? 1 means at least as \
good (correct, complete, helpful); 0 means wrong or unhelpful. Partial credit is allowed.
Reply with only a number between 0 and 1."""


@dataclass
class UpgradeReport:
    """What `models upgrade` found: the recommendation plus per-model scores for the table."""

    service: str
    current_model: str
    recommended_model: str
    baseline_score: float
    best_score: float
    model_scores: dict[str, float] = field(default_factory=dict)
    train_records: int = 0
    val_records: int = 0

    @property
    def changed(self) -> bool:
        return self.recommended_model != self.current_model

    def to_json(self) -> dict[str, Any]:
        return {
            "model_service": self.service,
            "current_model": self.current_model,
            "recommended_model": self.recommended_model,
            "changed": self.changed,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "model_scores": self.model_scores,
            "train_records": self.train_records,
            "val_records": self.val_records,
        }


# --- model names ----------------------------------------------------------------


def system_ai_name(model: str) -> str:
    """``claude-sonnet-4-5`` / ``system.ai.claude-sonnet-4-5`` → ``system.ai.claude-sonnet-4-5``."""
    return model if model.startswith(_SYSTEM_AI) else f"{_SYSTEM_AI}{model}"


def bare_name(model: str) -> str:
    """The optimizer's model key: the ``system.ai.*`` name without its schema prefix."""
    return model.removeprefix(_SYSTEM_AI)


# --- answers --------------------------------------------------------------------

_ASSISTANT_ROLES = {"assistant", "ai"}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ]
        return "".join(part for part in parts if part)
    return "" if content is None else str(content)


def final_text(value: Any) -> str:
    """The agent's final answer text from a run output, a trace response, or a message list.

    Handles both templates' shapes: the OpenAI Agents ``{"output": final_output}`` and the LangGraph
    last ``updates`` payload (``{node: {"messages": [...]}}``, as message objects or their dumps).
    The last assistant message wins; tool, user, and system messages are skipped.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            text = final_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        role = value.get("role") or value.get("type")
        if "content" in value and role is not None:
            return _content_text(value["content"]) if role in _ASSISTANT_ROLES else ""
        if "output" in value:
            return final_text(value["output"])
        if "messages" in value:
            return final_text(value["messages"])
        return final_text(list(value.values()))
    role = getattr(value, "type", None) or getattr(value, "role", None)
    if hasattr(value, "content"):
        return _content_text(value.content) if role in _ASSISTANT_ROLES else ""
    return str(value)


# --- eval set from traces -------------------------------------------------------


def _loads(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def records_from_traces(traces: Sequence[Any]) -> list[dict[str, Any]]:
    """Turn traces into optimizer records: the agent's input, and production's answer as reference.

    Only successful traces with a non-empty request and answer are kept. The input is carried
    verbatim under ``agent_input`` so the replay hands the agent exactly what production received.
    """
    records = []
    for trace in traces:
        info = getattr(trace, "info", None)
        state = str(getattr(info, "state", None) or getattr(info, "status", None) or "")
        if state and "OK" not in state.upper():
            continue
        data = getattr(trace, "data", None)
        request = _loads(getattr(data, "request", None))
        reference = final_text(_loads(getattr(data, "response", None)))
        if not request or not reference:
            continue
        records.append(
            {
                "inputs": {"agent_input": request},
                "expectations": {"expected_response": reference},
            }
        )
    return records


def split_records(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split into (train, val): about 30% validation, at least two records on each side."""
    if len(records) < MIN_RECORDS:
        raise AgentCliError(
            f"Found {len(records)} usable traces; models upgrade needs at least {MIN_RECORDS}.",
            hint="Send the deployed agent more traffic (or raise --traces), then retry.",
        )
    val_size = min(max(2, round(len(records) * 0.3)), len(records) - 2)
    return records[val_size:], records[:val_size]


def load_traces(profile: Optional[str], experiment_name: str, limit: int) -> list[Any]:
    """The most recent ``limit`` traces from the bound workspace trace experiment."""
    from databricks_agentbricks.cli import tracing  # noqa: PLC0415 - avoid import cycle

    mlflow = tracing._mlflow()
    tracing._set_tracking_uri(mlflow, profile)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise AgentCliError(
            f"Trace experiment '{experiment_name}' doesn't exist in this workspace yet.",
            hint="Deploy the agent (`agentbricks deploy`) and send it traffic first.",
        )
    return mlflow.search_traces(
        locations=[experiment.experiment_id], max_results=limit, return_type="list"
    )


# --- replaying the agent --------------------------------------------------------


def make_predict_fn(
    root: pathlib.Path, framework: AgentFramework, service: str
) -> Callable[[dict], str]:
    """A sync ``predict(inputs) -> answer`` that runs the project's own agent in-process.

    The agent must call ``service`` (the templates read it from ``AGENT_MODEL_SERVICE`` at import), or
    the optimizer's redirect to the ``<service>_exp`` clone never matches and every candidate would
    be scored on the template's default model. Each call gets a fresh session id. No session or
    memory store env is set for the replay, so the templates fall back to an in-memory checkpointer
    and no memory tools: replays never read or write production state.
    """
    os.environ[MODEL_SERVICE_ENV] = service
    os.environ.setdefault("AGENTBRICKS_PROJECT_ROOT", str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        agent_module = importlib.import_module("agent.agent")
    except ImportError as exc:
        raise AgentCliError(
            f"Could not import the agent from {root / 'agent' / 'agent.py'}: {exc}.",
            hint="Run from the project's environment so its dependencies resolve, e.g. "
            "`uv run --with 'databricks-agentbricks[upgrade]' agentbricks models upgrade`.",
        ) from exc
    run_agent = agent_module.run_agent

    if framework == AgentFramework.LANGGRAPH:

        async def _run(agent_input: Any) -> Any:
            last = None
            async for event in run_agent(agent_input, session_id=uuid.uuid4().hex):
                if isinstance(event, tuple) and event and event[0] == "updates":
                    last = event[1]
            return last

    else:

        async def _run(agent_input: Any) -> Any:
            async with run_agent(agent_input, session_id=uuid.uuid4().hex) as result:
                async for _ in result.stream_events():
                    pass
            return result.final_output

    def predict(inputs: dict) -> str:
        return final_text(asyncio.run(_run(inputs["agent_input"])))

    return predict


# --- scoring --------------------------------------------------------------------


def make_equivalence_scorer(judge_model: str = DEFAULT_JUDGE_MODEL) -> Callable[..., float]:
    """An LLM judge scoring a candidate answer against production's answer, 0 to 1. No labels needed."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    client = WorkspaceClient().serving_endpoints.get_open_ai_client()

    def equivalence(inputs: dict, expectations: dict, answer: Any) -> float:
        prompt = _JUDGE_PROMPT.format(
            request=json.dumps(inputs.get("agent_input"), default=str)[:4000],
            reference=str(expectations.get("expected_response", ""))[:4000],
            candidate=str(answer)[:4000],
        )
        response = client.chat.completions.create(
            model=judge_model, messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text)
        if match is None:
            raise ValueError(f"judge returned no score: {text[:80]!r}")
        return float(match.group())

    return equivalence


# --- the search -----------------------------------------------------------------


def _smu():
    try:
        import smart_model_upgrades as smu  # noqa: PLC0415 - heavy optional dependency
        from smart_model_upgrades import ai_gateway  # noqa: PLC0415
    except ImportError as exc:
        raise AgentCliError(
            "`agentbricks models upgrade` requires the 'upgrade' extra.", hint=UPGRADE_EXTRA_HINT
        ) from exc
    ai_gateway.set_backend("model_services")
    return smu


def _model_scores(result: Any, service: str) -> dict[str, float]:
    """Best validation score GEPA saw per model, for the comparison table (empty if unavailable)."""
    gepa_result = getattr(result, "gepa_result", None)
    candidates = getattr(gepa_result, "candidates", None) or []
    scores = getattr(gepa_result, "val_aggregate_scores", None) or []
    best: dict[str, float] = {}
    for candidate, score in zip(candidates, scores):
        model = candidate.get(f"model:{service}") if isinstance(candidate, dict) else None
        if model is None:
            continue
        key = system_ai_name(model)
        best[key] = max(best.get(key, float("-inf")), float(score))
    return best


def run_upgrade(
    *,
    root: pathlib.Path,
    framework: AgentFramework,
    service: str,
    current_model: str,
    candidates: Sequence[str],
    traces: Sequence[Any],
    budget: int,
    judge_model: str,
    weights: tuple[float, float, float],
    experiment_name: Optional[str],
    profile: Optional[str],
) -> UpgradeReport:
    """Search ``candidates`` (plus the current model) for the best model behind ``service``."""
    if profile:
        # The optimizer and the agent build their own WorkspaceClients; point them at this profile.
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    smu = _smu()
    train, val = split_records(records_from_traces(traces))
    predict = make_predict_fn(root, framework, service)
    scorer = make_equivalence_scorer(judge_model)
    if experiment_name:
        import mlflow  # noqa: PLC0415

        # Log runs to a sibling experiment: replays must not land in the agent's own trace
        # experiment, or the next upgrade would train on its own eval traffic.
        mlflow.set_experiment(f"{experiment_name}-model-upgrades")
    weight_quality, weight_latency, weight_cost = weights
    result = smu.optimize_prompts_and_models(
        predict,
        train,
        val,
        gateway_endpoints={service: [bare_name(m) for m in candidates]},
        scorers=[scorer],
        max_metric_calls=budget,
        weight_quality=weight_quality,
        weight_latency=weight_latency,
        weight_cost=weight_cost,
        display_progress_bar=False,
    )
    winner = result.best_candidate.get(f"model:{service}") or bare_name(current_model)
    return UpgradeReport(
        service=service,
        current_model=system_ai_name(current_model),
        recommended_model=system_ai_name(winner),
        baseline_score=float(result.baseline_score),
        best_score=float(result.best_score),
        model_scores=_model_scores(result, service),
        train_records=len(train),
        val_records=len(val),
    )


# --- history (rollback) ---------------------------------------------------------


def read_history(root: pathlib.Path) -> list[dict[str, Any]]:
    path = root / HISTORY_PATH
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentCliError(f"Could not read model upgrade history at {path}: {exc}.") from exc
    return history if isinstance(history, list) else []


def record_change(
    root: pathlib.Path,
    *,
    service: str,
    previous: Optional[str],
    model: str,
    reason: str,
    report: Optional[UpgradeReport] = None,
) -> None:
    """Append one repoint of ``service`` (``previous`` → ``model``) to the project's history."""
    entry: dict[str, Any] = {
        "timestamp": int(time.time()),
        "model_service": service,
        "previous_model": previous,
        "model": model,
        "reason": reason,
    }
    if report is not None:
        entry["baseline_score"] = report.baseline_score
        entry["best_score"] = report.best_score
    history = [*read_history(root), entry]
    path = root / HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
