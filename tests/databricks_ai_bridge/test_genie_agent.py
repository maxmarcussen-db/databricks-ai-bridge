import json
from unittest.mock import patch

import pytest

from databricks_ai_bridge.genie import (
    FUNCTION_CALL,
    FUNCTION_CALL_OUTPUT,
    MESSAGE,
    REASONING,
    Genie,
    GenieAgentResponse,
    _iter_sse_events,
    _parse_agent_response,
)


@pytest.fixture
def mock_workspace_client():
    with patch("databricks_ai_bridge.genie.WorkspaceClient") as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
        yield mock_client


@pytest.fixture
def agent(mock_workspace_client):
    # agent_mode=True skips the get_space() description lookup and routes ask_question
    # through the agent-mode responses API.
    return Genie(space_id="test_agent_id", agent_mode=True)


# A terminal response object matching the real API shape captured against a live space:
# multi-step (3 SQL calls), reasoning, and a final assistant message with a narrative
# chunk (with a citation annotation) plus a table chunk (with metadata).
COMPLETED_RESPONSE = {
    "id": "resp_1",
    "status": "completed",
    "conversation_id": "conv_1",
    "model": "genie-agent",
    "object": "response",
    "output": [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "I'll find the top cities."}],
        },
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "execute_sql",
            "arguments": json.dumps({"title": "Top cities", "sql": "SELECT city FROM t LIMIT 3"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "output": "**Top cities**\n\n| city |\n| --- |\n| Phuket |",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "The top city is Phuket.",
                    "annotations": [
                        {"url": "https://example.databricks.com/genie/rooms/x?gra_focus=call_a"}
                    ],
                },
                {
                    "type": "output_text",
                    "text": "**Top cities**\n\n| city | property_count |\n| --- | --- |",
                    "metadata": {
                        "sql": "SELECT city, COUNT(*) property_count FROM t GROUP BY city",
                        "columns": [
                            {"name": "city", "type": "STRING"},
                            {"name": "property_count", "type": "BIGINT"},
                        ],
                        "preview_rows": [["Phuket", "1788"], ["Mallorca", "1626"]],
                        "total_row_count": 3,
                        "status": "available",
                    },
                },
            ],
        },
    ],
}


def _sse(events):
    """Render a list of event dicts as an SSE byte stream like the API returns."""
    lines = []
    for ev in events:
        lines.append(f"event:{ev.get('type', 'message')}")
        lines.append("data:" + json.dumps(ev))
        lines.append("")
    return ("\n".join(lines)).encode("utf-8")


class _ChunkedStream:
    """A readable byte stream with read(n), like the SDK's streaming response.

    max_chunk caps how many bytes a read() returns, to force line splits across reads.
    """

    def __init__(self, data: bytes, max_chunk: int = 0):
        self._data = data
        self._pos = 0
        self._max_chunk = max_chunk
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data)
        if self._max_chunk:
            n = min(n, self._max_chunk)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def test_iter_sse_events_parses_data_lines():
    stream = _sse(
        [
            {"type": "response.created", "response": {"id": "r", "status": "in_progress"}},
            {"type": "response.completed", "response": {"id": "r", "status": "completed"}},
        ]
    )
    events = list(_iter_sse_events(stream))
    assert len(events) == 2
    assert events[0]["type"] == "response.created"
    assert events[1]["response"]["status"] == "completed"


def test_iter_sse_events_from_readable_stream():
    stream = _ChunkedStream(_sse([{"type": "response.completed", "response": {"id": "r"}}]))
    events = list(_iter_sse_events(stream))
    assert events[0]["response"]["id"] == "r"


def test_iter_sse_events_from_text_stream():
    # A readable stream whose read() returns str (EOF == "") must terminate, not hang.
    text = _sse([{"type": "response.completed", "response": {"id": "r"}}]).decode("utf-8")
    stream = _ChunkedStream(text)  # yields str chunks, "" at EOF
    events = list(_iter_sse_events(stream))
    assert events[0]["response"]["id"] == "r"


def test_iter_sse_events_across_chunk_boundaries():
    # 1-byte reads force lines to split across chunks; buffering must still parse them.
    stream = _ChunkedStream(
        _sse([{"type": "response.completed", "response": {"id": "r", "n": 42}}]), max_chunk=1
    )
    events = [e for e in _iter_sse_events(stream) if "response" in e]
    assert events[0]["response"]["n"] == 42


def test_iter_sse_events_skips_done_and_garbage():
    raw = b'data: [DONE]\ndata: not-json\ndata: {"ok": true}\n'
    events = list(_iter_sse_events(raw))
    assert events == [{"ok": True}]


def test_parse_agent_response_structure():
    resp = _parse_agent_response(COMPLETED_RESPONSE)
    assert isinstance(resp, GenieAgentResponse)
    assert resp.status == "completed"
    assert resp.conversation_id == "conv_1"
    assert resp.answer == "The top city is Phuket."


def test_parse_agent_response_preserves_full_trace():
    resp = _parse_agent_response(COMPLETED_RESPONSE)
    types = [s.type for s in resp.steps]
    assert types == [REASONING, FUNCTION_CALL, FUNCTION_CALL_OUTPUT, MESSAGE]

    reasoning = resp.steps[0]
    assert reasoning.text == "I'll find the top cities."

    call = resp.steps[1]
    assert call.title == "Top cities"
    assert call.sql == "SELECT city FROM t LIMIT 3"
    assert call.call_id == "call_a"

    output = resp.steps[2]
    assert output.call_id == "call_a"
    assert "Phuket" in output.output


def test_parse_agent_response_sql_queries_helper():
    resp = _parse_agent_response(COMPLETED_RESPONSE)
    assert resp.sql_queries == ["SELECT city FROM t LIMIT 3"]


def test_parse_agent_response_extracts_citations():
    resp = _parse_agent_response(COMPLETED_RESPONSE)
    assert resp.citations == ["https://example.databricks.com/genie/rooms/x?gra_focus=call_a"]


def test_parse_agent_response_table_markdown_not_in_answer():
    # The rendered-table chunk (has metadata) is skipped, not folded into the narrative answer.
    resp = _parse_agent_response(COMPLETED_RESPONSE)
    assert "| city |" not in resp.answer


def test_parse_agent_response_failed():
    failed = {
        "id": "resp_x",
        "status": "failed",
        "conversation_id": "conv_x",
        "error": {"type": "sql_execution_error", "code": "SQL_ERROR"},
    }
    resp = _parse_agent_response(failed)
    assert resp.status == "failed"
    assert resp.answer == ""
    assert resp.error["type"] == "sql_execution_error"


def test_parse_agent_response_system_error_message():
    # When the agent returns a system-role message (error string) and no assistant text.
    resp_obj = {
        "id": "r",
        "status": "completed",
        "conversation_id": "c",
        "output": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "output_text", "text": "I could not answer that."}],
            }
        ],
    }
    resp = _parse_agent_response(resp_obj)
    assert resp.answer == "I could not answer that."


def test_ask_question_new_conversation(agent, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"contents": _sse_stream(COMPLETED_RESPONSE)}
    resp = agent.ask_question("What are the top cities?")

    assert resp.answer == "The top city is Phuket."
    assert resp.conversation_id == "conv_1"

    # Verify the request shape: no conversation_id, correct endpoint, streaming raw.
    args, kwargs = mock_workspace_client.genie._api.do.call_args
    assert args[0] == "POST"
    assert args[1] == "/api/2.0/genie/agents/test_agent_id/responses"
    assert kwargs["raw"] is True
    body = kwargs["body"]
    assert "conversation_id" not in body
    assert body["input"][0]["content"][0]["text"] == "What are the top cities?"


def test_ask_question_continues_conversation(agent, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"contents": _sse_stream(COMPLETED_RESPONSE)}
    agent.ask_question("Follow up", conversation_id="conv_1")

    _, kwargs = mock_workspace_client.genie._api.do.call_args
    assert kwargs["body"]["conversation_id"] == "conv_1"


def test_stream_emits_one_span_per_timeline_item(agent, mock_workspace_client, tmp_path):
    # Emits one span per streamed output item under a genie_timeline parent.
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("genie-agent-span-test")

    mock_workspace_client.genie._api.do.return_value = {"contents": _sse_stream(COMPLETED_RESPONSE)}
    agent.ask_question("What are the top cities?")

    exp = mlflow.get_experiment_by_name("genie-agent-span-test")
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id], max_results=1, return_type="list"
    )
    assert traces, "expected a trace to be logged"
    span_names = [s.name for s in (traces[0].data.spans or [])]

    # One parent timeline span plus a span per output item (reasoning, function_call,
    # function_call_output, message = 4 items in COMPLETED_RESPONSE).
    assert "genie_timeline" in span_names
    assert any(n == "reasoning" for n in span_names)
    assert any(n.startswith("function_call:") for n in span_names)
    assert "function_call_output" in span_names
    assert any(n.startswith("message") for n in span_names)


def test_create_response_falls_back_to_latest_without_terminal_event(agent, mock_workspace_client):
    # Stream ends without an explicit response.completed; latest snapshot is used.
    events = [
        {
            "type": "response.created",
            "response": {"id": "r", "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.done",
            "response": {"id": "r", "status": "completed", "conversation_id": "c", "output": []},
        },
    ]
    mock_workspace_client.genie._api.do.return_value = {"contents": _ChunkedStream(_sse(events))}
    resp = agent.create_agent_response("q")
    assert resp["status"] == "completed"
    assert resp["conversation_id"] == "c"


def test_create_response_raises_on_empty_stream(agent, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"contents": _ChunkedStream(b"")}
    with pytest.raises(RuntimeError, match="without a response object"):
        agent.create_agent_response("q")


@pytest.mark.parametrize("failing_method", ["start_span", "end_span"])
def test_span_failure_does_not_break_turn(agent, mock_workspace_client, failing_method):
    # A per-item span start OR end failure mid-stream must not fail an otherwise-good turn.
    import mlflow

    mock_workspace_client.genie._api.do.return_value = {"contents": _sse_stream(COMPLETED_RESPONSE)}
    with patch.object(
        mlflow.tracking.MlflowClient,
        failing_method,
        side_effect=mlflow.exceptions.MlflowTracingException("boom"),
    ):
        resp = agent.ask_question("What are the top cities?")
    assert resp.status == "completed"
    assert resp.answer == "The top city is Phuket."


@pytest.mark.parametrize("bad_idx", [None, [0], {"a": 1}])
def test_item_events_with_bad_output_index_are_skipped(agent, mock_workspace_client, bad_idx):
    # Missing or unhashable output_index must not raise; the response still parses.
    item = {"type": "reasoning", "content": []}
    added = {"type": "response.output_item.added", "item": item}
    done = {"type": "response.output_item.done", "item": item}
    if bad_idx is not None:
        added["output_index"] = bad_idx
        done["output_index"] = bad_idx
    events = [
        {"type": "response.created", "response": {**COMPLETED_RESPONSE, "status": "in_progress"}},
        added,
        done,
        {"type": "response.completed", "response": COMPLETED_RESPONSE},
    ]
    mock_workspace_client.genie._api.do.return_value = {"contents": _ChunkedStream(_sse(events))}
    resp = agent.ask_question("What are the top cities?")
    assert resp.status == "completed"


def _sse_stream(response_obj):
    """Build a mock streaming object whose read() returns a full SSE exchange, incl. per-item
    added/done events so the span-emission path is exercised."""
    events = [
        {"type": "response.created", "response": {**response_obj, "status": "in_progress"}},
    ]
    for i, item in enumerate(response_obj.get("output", [])):
        events.append({"type": "response.output_item.added", "output_index": i, "item": item})
        events.append({"type": "response.output_item.done", "output_index": i, "item": item})
    events.append({"type": "response.completed", "response": response_obj})
    return _ChunkedStream(_sse(events))
