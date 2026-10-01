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
    assert model_upgrade.read_history(tmp_path) == []
    model_upgrade.record_change(
        tmp_path, service="a.b.c", previous="system.ai.x", model="system.ai.y", reason="set"
    )
    (entry,) = model_upgrade.read_history(tmp_path)
    assert entry["previous_model"] == "system.ai.x"
    assert entry["model"] == "system.ai.y"
    assert (tmp_path / ".agentbricks" / "model_upgrades.json").is_file()
