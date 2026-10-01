"""Unit tests for the `agentbricks models upgrade` engine's pure parts: names, answers, eval set."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from databricks_agentbricks import model_upgrade
from databricks_agentbricks.errors import AgentCliError


def _trace(request, response, state="OK"):
    return SimpleNamespace(
        info=SimpleNamespace(state=state),
        data=SimpleNamespace(
            request=json.dumps(request) if request is not None else None,
            response=json.dumps(response) if response is not None else None,
        ),
    )


def test_model_names_round_trip():
    assert model_upgrade.system_ai_name("claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    assert model_upgrade.system_ai_name("system.ai.claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    assert model_upgrade.bare_name("system.ai.claude-haiku-4-5") == "claude-haiku-4-5"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain", "plain"),
        # OpenAI Agents template: the root span's outputs.
        ({"output": "final answer"}, "final answer"),
        # LangGraph template: the last `updates` payload, dumped into the trace.
        (
            {
                "model": {
                    "messages": [
                        {"type": "ai", "content": [{"type": "text", "text": "Hi "}, {"text": "there"}]}
                    ]
                }
            },
            "Hi there",
        ),
        # The last assistant message wins; trailing tool / user messages are skipped.
        (
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "tool output"},
            ],
            "a1",
        ),
        (None, ""),
    ],
)
def test_final_text(value, expected):
    assert model_upgrade.final_text(value) == expected


def test_final_text_reads_message_objects():
    message = SimpleNamespace(type="ai", content="from an AIMessage")
    tool = SimpleNamespace(type="tool", content="tool output")
    assert model_upgrade.final_text({"tools": {"messages": [message, tool]}}) == "from an AIMessage"


def test_records_from_traces_keeps_ok_traces_with_answers():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    traces = [
        _trace(request, {"output": "hi"}),
        _trace(request, {"output": "boom"}, state="ERROR"),
        _trace(None, {"output": "no request"}),
        _trace(request, {"output": ""}),
    ]
    assert model_upgrade.records_from_traces(traces) == [
        {"inputs": {"agent_input": request}, "expectations": {"expected_response": "hi"}}
    ]


def test_split_records_holds_out_about_thirty_percent():
    records = [{"i": i} for i in range(10)]
    train, val = model_upgrade.split_records(records)
    assert len(val) == 3
    assert len(train) == 7
    assert sorted(r["i"] for r in train + val) == list(range(10))


def test_split_records_needs_enough_traces():
    with pytest.raises(AgentCliError, match="at least"):
        model_upgrade.split_records([{"i": i} for i in range(model_upgrade.MIN_RECORDS - 1)])


def test_history_round_trip(tmp_path):
    assert model_upgrade.read_history(tmp_path) == {"runs": [], "changes": []}
    model_upgrade.record_run(tmp_path, {"upgrade_id": "u1", "run_id": 7, "state": "PENDING"})
    model_upgrade.update_run(tmp_path, "u1", state="SUCCESS")
    model_upgrade.record_change(
        tmp_path, service="a.b.c", previous="system.ai.x", model="system.ai.y", reason="set"
    )
    history = model_upgrade.read_history(tmp_path)
    assert history["runs"][0]["state"] == "SUCCESS"
    (entry,) = history["changes"]
    assert entry["previous_model"] == "system.ai.x"
    assert entry["model"] == "system.ai.y"
    assert (tmp_path / ".agentbricks" / "model_upgrades.json").is_file()


def test_job_config_round_trips_through_the_job_parameter():
    config = model_upgrade.JobConfig(
        upgrade_id="u1",
        service="a.b.c",
        framework="langgraph",
        candidates=["system.ai.x"],
        trace_experiment="/Shared/t",
        trace_limit=50,
        budget=None,
        judge_model="databricks-claude-sonnet-4-6",
        weights=(0.7, 0.2, 0.1),
    )
    assert model_upgrade.JobConfig.from_param(config.to_param()) == config


def test_report_round_trips_through_json():
    report = model_upgrade.UpgradeReport(
        service="a.b.c",
        current_model="system.ai.x",
        recommended_model="system.ai.y",
        baseline_score=0.5,
        best_score=0.6,
        model_scores={"system.ai.y": 0.6},
        train_records=7,
        val_records=3,
    )
    assert model_upgrade.UpgradeReport.from_json(report.to_json()) == report


def test_upgrade_requirement_override(monkeypatch):
    monkeypatch.setenv(model_upgrade.UPGRADE_REQUIREMENT_ENV, "databricks-agentbricks[upgrade] @ git+x")
    assert model_upgrade.upgrade_requirement() == "databricks-agentbricks[upgrade] @ git+x"


def test_submit_upgrade_run_uploads_runner_and_installs_project(monkeypatch):
    monkeypatch.setenv(model_upgrade.UPGRADE_REQUIREMENT_ENV, "databricks-agentbricks[upgrade]==9.9")
    class _Client:
        def __init__(self):
            self.uploaded, self.submitted = {}, {}

        def upload_workspace_file(self, path, content):
            self.uploaded[path] = content

        def submit_serverless_python_run(self, **kwargs):
            self.submitted = kwargs
            return 9

        def get_run(self, run_id):
            return SimpleNamespace(run_page_url="https://ws/run/9")

    client = _Client()
    config = model_upgrade.JobConfig("u1", "a.b.c", "openai", ["system.ai.x"], "/Shared/t", 20, 40, "j", (1.0, 0.0, 0.0))
    run_id, url = model_upgrade.submit_upgrade_run(client, workspace_path="/Workspace/Users/me/p", config=config)
    assert (run_id, url) == (9, "https://ws/run/9")
    runner = "/Workspace/Users/me/p/.agentbricks/model_upgrade_job.py"
    assert "job_main" in client.uploaded[runner]
    # Set before any import: psycopg's bundled OpenSSL aborts on FIPS compute.
    assert client.uploaded[runner].index("PSYCOPG_IMPL") < client.uploaded[runner].index("import job_main")
    assert client.submitted["python_file"] == runner
    assert client.submitted["parameters"] == ["/Workspace/Users/me/p", config.to_param()]
    assert client.submitted["dependencies"] == [
        "/Workspace/Users/me/p",
        "databricks-agentbricks[upgrade]==9.9",
    ]
    assert client.submitted["environment_version"] == model_upgrade.SERVERLESS_ENVIRONMENT_VERSION
