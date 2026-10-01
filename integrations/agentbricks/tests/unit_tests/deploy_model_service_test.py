"""Tests for deploy's model-service reconcile: create when missing, never repoint an existing one."""

from __future__ import annotations

import pathlib

import pytest

from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.errors import AgentCliError

SERVICE = "main.my_agent.llm"


class _FakeClient:
    def __init__(self, exists: bool):
        self.exists = exists
        self.created: list[tuple[str, str]] = []

    def get_model_service(self, name):
        if not self.exists:
            raise AgentCliError("Resource not found", error_code="RESOURCE_DOES_NOT_EXIST")
        return {"name": f"model-services/{name}"}

    def create_model_service(self, name, model, *, comment=None):
        self.created.append((name, model))
        return {"name": f"model-services/{name}"}


def _project(tmp_path: pathlib.Path, default: str | None = "system.ai.claude-sonnet-4-5"):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.bind_model_service(SERVICE, default)
    project.write()
    return AgentProject.load(tmp_path)


def test_unbound_project_reconciles_nothing():
    assert deploy_mod._reconcile_model_service(None, _FakeClient(exists=False)) is None


def test_missing_service_is_created_with_default(tmp_path):
    client = _FakeClient(exists=False)
    assert deploy_mod._reconcile_model_service(_project(tmp_path), client) == SERVICE
    assert client.created == [(SERVICE, "system.ai.claude-sonnet-4-5")]


def test_existing_service_is_left_alone(tmp_path):
    client = _FakeClient(exists=True)
    assert deploy_mod._reconcile_model_service(_project(tmp_path), client) == SERVICE
    assert client.created == []


def test_missing_service_without_default_explains_how_to_fix(tmp_path):
    with pytest.raises(AgentCliError, match="no default model") as exc_info:
        deploy_mod._reconcile_model_service(_project(tmp_path, default=None), _FakeClient(False))
    assert "--default" in (exc_info.value.hint or "")

