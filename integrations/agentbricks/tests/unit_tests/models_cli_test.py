"""CLI tests for `agentbricks models` (bind / unbind / status / set / rollback / upgrade).

Uses a real temp AgentProject and a fake client holding one model service's destination. The
optimizer itself is stubbed: `model_upgrade_test.py` covers the trace and answer handling.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_agentbricks import model_upgrade
from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli.models import models
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_config import write_project_metadata

SERVICE = "main.my_agent.llm"


def _service(model: str) -> dict:
    return {
        "name": f"model-services/{SERVICE}",
        "config": {
            "routing": {
                "destinations": [
                    {
                        "name": "primary",
                        "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                        "pay_per_token_config": {"model": f"models/{model}"},
                        "traffic_percentage": 100,
                    }
                ]
            }
        },
    }


class _FakeClient:
    def __init__(self, model: str | None = "system.ai.claude-sonnet-4-5"):
        self.model = model
        self.calls: list[tuple] = []

    def get_model_service(self, name):
        self.calls.append(("get", name))
        if self.model is None:
            raise AgentCliError("not found", error_code="NOT_FOUND")
        return _service(self.model)

    def set_model_service_model(self, name, model):
        self.calls.append(("set", name, model))
        self.model = model
        return _service(model)

    def list_chat_model_services(self):
        return ["system.ai.claude-haiku-4-5", "system.ai.claude-sonnet-4-5"]


class _Ctx:
    def __init__(self, client=None, output="text"):
        self._client = client
        self.output = output
        self.profile = None

    def client(self):
        if self._client is None:
            raise AssertionError("This command must not contact the workspace")
        return self._client


def _project(tmp_path: pathlib.Path, *, bind=True, tracing=True) -> pathlib.Path:
    project = tmp_path / "agent-langgraph"
    (project / "agent").mkdir(parents=True)
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    created = AgentProject.create(
        project,
        framework="langgraph",
        server="agentbricks",
        experiment_name="/Shared/agentbricks_traces/my-agent" if tracing else None,
    )
    if bind:
        created.bind_model_service(SERVICE, "system.ai.claude-sonnet-4-5")
    created.write()
    return project


def _invoke(args, obj):
    return CliRunner().invoke(models, args, obj=obj)


# --- bind / unbind ----------------------------------------------------------------


@pytest.mark.parametrize("output", ["text", "json"])
def test_bind_writes_agent_toml_without_contacting_workspace(tmp_path, output):
    project = _project(tmp_path, bind=False)
    result = _invoke(
        ["bind", SERVICE, "--default", "claude-sonnet-4-5", "--source", str(project)],
        _Ctx(output=output),
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.output) == {
            "model_service": SERVICE,
            "default": "system.ai.claude-sonnet-4-5",
            "manifest": str(project / "agent.toml"),
        }
    reloaded = AgentProject.load(project)
    assert reloaded.model_service == SERVICE
    assert reloaded.model_service_default == "system.ai.claude-sonnet-4-5"


def test_bind_keeps_recorded_default_when_omitted(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["bind", "main.my_agent.other", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    reloaded = AgentProject.load(project)
    assert reloaded.model_service == "main.my_agent.other"
    assert reloaded.model_service_default == "system.ai.claude-sonnet-4-5"


def test_bind_rejects_non_three_part_name(tmp_path):
    project = _project(tmp_path, bind=False)
    result = _invoke(["bind", "just-a-name", "--source", str(project)], _Ctx())
    assert result.exit_code != 0
    assert AgentProject.load(project).model_service is None


def test_unbind_removes_the_table(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["unbind", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).model_service is None
    assert "model_service" not in (project / "agent.toml").read_text()


# --- status / set / rollback --------------------------------------------------------


def test_status_reports_current_destination(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_service"] == SERVICE
    assert payload["model"] == "system.ai.claude-sonnet-4-5"
    assert payload["last_change"] is None


def test_status_without_binding_points_at_bind(tmp_path):
    project = _project(tmp_path, bind=False)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient()))
    assert result.exit_code != 0
    assert "no model service bound" in result.output


def test_status_before_deploy_points_at_deploy(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(model=None)))
    assert result.exit_code != 0
    assert "doesn't exist yet" in result.output


def test_set_switches_and_records_history_then_rollback_restores(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["set", "claude-haiku-4-5", "--yes", "--source", str(project)], _Ctx(client, "json")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    history = model_upgrade.read_history(project)
    assert history[-1]["previous_model"] == "system.ai.claude-sonnet-4-5"
    assert history[-1]["model"] == "system.ai.claude-haiku-4-5"
    assert history[-1]["reason"] == "set"

    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert model_upgrade.read_history(project)[-1]["reason"] == "rollback"


def test_set_json_without_yes_never_switches(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(["set", "claude-haiku-4-5", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] is False
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert not any(call[0] == "set" for call in client.calls)


def test_set_to_current_model_is_a_no_op(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["set", "system.ai.claude-sonnet-4-5", "--yes", "--source", str(project)], _Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert not any(call[0] == "set" for call in client.calls)
    assert model_upgrade.read_history(project) == []


def test_rollback_with_no_history_errors(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(_FakeClient()))
    assert result.exit_code != 0
    assert "No recorded switch" in result.output


# --- upgrade ----------------------------------------------------------------------------


def _report(recommended: str) -> model_upgrade.UpgradeReport:
    return model_upgrade.UpgradeReport(
        service=SERVICE,
        current_model="system.ai.claude-sonnet-4-5",
        recommended_model=recommended,
        baseline_score=0.80,
        best_score=0.82,
        model_scores={"system.ai.claude-sonnet-4-5": 0.80, recommended: 0.82},
        train_records=7,
        val_records=3,
    )


@pytest.fixture
def stub_upgrade(monkeypatch):
    calls = {}

    def _load_traces(profile, experiment_name, limit):
        calls["load_traces"] = (profile, experiment_name, limit)
        return ["t"] * 10

    def _run_upgrade(**kwargs):
        calls["run_upgrade"] = kwargs
        return _report("system.ai.claude-haiku-4-5")

    monkeypatch.setattr(model_upgrade, "load_traces", _load_traces)
    monkeypatch.setattr(model_upgrade, "run_upgrade", _run_upgrade)
    return calls


def test_upgrade_switches_to_recommendation_with_yes(tmp_path, stub_upgrade):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5,system.ai.gpt-5-4-mini", "--yes", "--source", str(project)],
        _Ctx(client, "json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["recommended_model"] == "system.ai.claude-haiku-4-5"
    assert payload["switched"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    kwargs = stub_upgrade["run_upgrade"]
    assert kwargs["service"] == SERVICE
    assert kwargs["current_model"] == "system.ai.claude-sonnet-4-5"
    assert kwargs["candidates"] == ["system.ai.claude-haiku-4-5", "system.ai.gpt-5-4-mini"]
    assert kwargs["weights"] == (0.7, 0.2, 0.1)
    assert kwargs["budget"] == 40
    assert stub_upgrade["load_traces"][1] == "/Shared/agentbricks_traces/my-agent"
    entry = model_upgrade.read_history(project)[-1]
    assert entry["reason"] == "upgrade"
    assert entry["best_score"] == pytest.approx(0.82)


def test_upgrade_dry_run_never_switches(tmp_path, stub_upgrade):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5", "--dry-run", "--yes", "--source", str(project)],
        _Ctx(client),
    )
    assert result.exit_code == 0, result.output
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert "Recommended" in result.output


def test_upgrade_requires_tracing(tmp_path, stub_upgrade):
    project = _project(tmp_path, tracing=False)
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5", "--source", str(project)], _Ctx(_FakeClient())
    )
    assert result.exit_code != 0
    assert "tracing" in result.output
    assert "run_upgrade" not in stub_upgrade


def test_upgrade_rejects_bad_weights(tmp_path, stub_upgrade):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5", "--weights", "0.5,0.5,0.5", "--source", str(project)],
        _Ctx(_FakeClient()),
    )
    assert result.exit_code != 0
    assert "run_upgrade" not in stub_upgrade


def test_list_shows_chat_models(tmp_path):
    result = _invoke(["list"], _Ctx(_FakeClient(), "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["system.ai.claude-haiku-4-5", "system.ai.claude-sonnet-4-5"]
