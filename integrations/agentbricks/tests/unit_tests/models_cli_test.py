"""CLI tests for `agentbricks models` (bind / unbind / status / set / rollback / upgrade).

Uses a real temp AgentProject and a fake client holding one model service's destination. The
optimizer itself is stubbed: `model_upgrade_test.py` covers the trace and answer handling.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

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
    assert payload["latest_run"] is None


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
    history = model_upgrade.read_history(project)["changes"]
    assert history[-1]["previous_model"] == "system.ai.claude-sonnet-4-5"
    assert history[-1]["model"] == "system.ai.claude-haiku-4-5"
    assert history[-1]["reason"] == "set"

    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert model_upgrade.read_history(project)["changes"][-1]["reason"] == "rollback"


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
    assert model_upgrade.read_history(project)["changes"] == []


def test_rollback_with_no_history_errors(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(_FakeClient()))
    assert result.exit_code != 0
    assert "No recorded switch" in result.output


# --- upgrade (job) / status / apply ------------------------------------------------------


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


class _JobClient(_FakeClient):
    current_user = "me@example.com"

    def __init__(self, life_cycle="RUNNING", result=None, denied=()):
        super().__init__()
        self.life_cycle, self.result, self.denied = life_cycle, result, set(denied)

    def can_execute_model(self, model):
        return model not in self.denied

    def get_run(self, run_id):
        state = SimpleNamespace(life_cycle_state=self.life_cycle, result_state=self.result)
        return SimpleNamespace(state=state, run_page_url=f"https://ws/run/{run_id}")


@pytest.fixture
def stub_job(monkeypatch):
    calls: dict = {"synced": []}

    def _databricks(args, profile, **kwargs):
        calls["synced"].append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _submit(client, *, workspace_path, config, timeout_hours):
        calls["submit"] = (workspace_path, config, timeout_hours)
        return 42, "https://ws/run/42"

    def _fetch(profile, trace_experiment, upgrade_id):
        calls["fetch"] = (trace_experiment, upgrade_id)
        return _report("system.ai.claude-haiku-4-5")

    monkeypatch.setattr("databricks_agentbricks.databricks_cli._databricks", _databricks)
    monkeypatch.setattr(model_upgrade, "submit_upgrade_run", _submit)
    monkeypatch.setattr(model_upgrade, "fetch_report", _fetch)
    return calls


def test_upgrade_uploads_project_and_submits_job_without_switching(tmp_path, stub_job):
    project = _project(tmp_path)
    client = _JobClient()
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5,system.ai.gpt-5-4-mini", "--source", str(project)],
        _Ctx(client, "json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == 42
    assert payload["run"] is None  # returned once submitted

    ((sync_args,),) = [stub_job["synced"]]
    ws_path = "/Workspace/Users/me@example.com/agentbricks_model_upgrades/agent-langgraph"
    assert sync_args[:3] == ["sync", str(project), ws_path]
    submitted_path, config, timeout_hours = stub_job["submit"]
    assert submitted_path == ws_path
    assert config.service == SERVICE
    assert config.framework == "langgraph"
    assert config.candidates == ["system.ai.claude-haiku-4-5", "system.ai.gpt-5-4-mini"]
    assert config.trace_experiment == "/Shared/agentbricks_traces/my-agent"
    assert config.weights == (0.7, 0.2, 0.1)
    assert timeout_hours == 6.0
    # Submitting never switches the model.
    assert client.model == "system.ai.claude-sonnet-4-5"
    (run,) = model_upgrade.read_history(project)["runs"]
    assert run["run_id"] == 42 and run["upgrade_id"] == config.upgrade_id


def test_status_then_apply_switches_to_finished_runs_recommendation(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke(["upgrade", "-c", "claude-haiku-4-5", "--source", str(project)], _Ctx(_JobClient(), "json"))

    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    status = _invoke(["status", "--source", str(project)], _Ctx(client, "json"))
    assert status.exit_code == 0, status.output
    latest = json.loads(status.output)["latest_run"]
    assert latest["state"] == "SUCCESS"
    assert latest["report"]["recommended_model"] == "system.ai.claude-haiku-4-5"

    applied = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    change = model_upgrade.read_history(project)["changes"][-1]
    assert change["reason"] == "upgrade"
    assert change["best_score"] == pytest.approx(0.82)


def test_apply_before_the_run_finishes_refuses(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke(["upgrade", "-c", "claude-haiku-4-5", "--source", str(project)], _Ctx(_JobClient(), "json"))
    client = _JobClient(life_cycle="RUNNING")
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client))
    assert result.exit_code != 0
    assert "no recommendation" in result.output
    assert client.model == "system.ai.claude-sonnet-4-5"


def test_apply_with_no_runs_errors(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(_JobClient()))
    assert result.exit_code != 0
    assert "No upgrade run" in result.output


def test_upgrade_refuses_candidates_the_owner_cannot_execute(tmp_path, stub_job):
    project = _project(tmp_path)
    client = _JobClient(denied={"system.ai.gpt-5-4-mini"})
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5,gpt-5-4-mini", "--source", str(project)], _Ctx(client)
    )
    assert result.exit_code != 0
    assert "gpt-5-4-mini" in result.output
    assert "submit" not in stub_job and stub_job["synced"] == []


def test_upgrade_requires_tracing(tmp_path, stub_job):
    project = _project(tmp_path, tracing=False)
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5", "--source", str(project)], _Ctx(_JobClient())
    )
    assert result.exit_code != 0
    assert "tracing" in result.output
    assert "submit" not in stub_job


def test_upgrade_rejects_bad_weights(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", "-c", "claude-haiku-4-5", "--weights", "0.5,0.5,0.5", "--source", str(project)],
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "submit" not in stub_job


def test_list_shows_chat_models(tmp_path):
    result = _invoke(["list"], _Ctx(_FakeClient(), "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == ["system.ai.claude-haiku-4-5", "system.ai.claude-sonnet-4-5"]
