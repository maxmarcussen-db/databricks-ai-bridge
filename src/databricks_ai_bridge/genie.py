import bisect
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from databricks.sdk import WorkspaceClient

from ._compat import mlflow_trace

MAX_TOKENS_OF_DATA = 20000
MAX_ITERATIONS = 500  # for 250 s total
ITERATION_FREQUENCY = 0.5  # seconds

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "QUERY_RESULT_EXPIRED",
}

# Agent response item types.
REASONING = "reasoning"
FUNCTION_CALL = "function_call"
FUNCTION_CALL_OUTPUT = "function_call_output"
MESSAGE = "message"

_AGENT_TERMINAL_EVENTS = {"response.completed", "response.failed"}

_AGENT_SPAN_TYPE_BY_ITEM = {
    REASONING: "LLM",
    FUNCTION_CALL: "TOOL",
    FUNCTION_CALL_OUTPUT: "TOOL",
    MESSAGE: "LLM",
}


# Define a function to count tokens
def _count_tokens(text):
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


@dataclass
class GenieResponse:
    result: Union[str, pd.DataFrame]
    query: Optional[str] = ""
    description: Optional[str] = ""
    conversation_id: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    text_attachment_content: Optional[str] = ""


@dataclass
class GenieAgentStep:
    # One item in an agent response's trace; fields are populated per `type`.
    type: str
    text: str = ""
    title: str = ""
    sql: str = ""
    output: str = ""
    call_id: str = ""
    role: str = ""


@dataclass
class GenieAgentResponse:
    answer: str
    steps: List[GenieAgentStep] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    status: str = "completed"
    error: Optional[Dict[str, Any]] = None

    @property
    def sql_queries(self) -> List[str]:
        return [s.sql for s in self.steps if s.type == FUNCTION_CALL and s.sql]


@mlflow_trace(span_type="PARSER")
def _parse_query_result(
    resp, truncate_results, return_pandas: bool = False
) -> Union[str, pd.DataFrame]:
    output = resp["result"]
    if not output:
        return "EMPTY"

    columns = resp["manifest"]["schema"]["columns"]
    header: list[str] = [str(col["name"]) for col in columns]
    rows: list[list] = []

    # Track float column indices (+1 offset for index column added by to_markdown)
    float_column_indices: list[int] = []
    for idx, col in enumerate(columns):
        if col["type_name"] in ["FLOAT", "DOUBLE", "DECIMAL"]:
            float_column_indices.append(idx + 1)

    for item in output["data_array"]:
        row: list = []
        for column, value in zip(columns, item):
            type_name = column["type_name"]
            if value is None:
                row.append(None)
                continue

            if type_name in ["INT", "LONG", "SHORT", "BYTE"]:
                row.append(int(value))
            elif type_name in ["FLOAT", "DOUBLE", "DECIMAL"]:
                if return_pandas:
                    row.append(float(value))
                else:
                    # Keep as string to avoid scientific notation in markdown
                    row.append(value)
            elif type_name == "BOOLEAN":
                row.append(value.lower() == "true")
            elif type_name == "DATE":
                # first 10 characters represent the date
                row.append(datetime.strptime(value[:10], "%Y-%m-%d").date())
            elif type_name == "TIMESTAMP":
                # https://docs.databricks.com/aws/en/sql/language-manual/data-types/timestamp-type
                # first 19 characters represent the date and time to the second
                # doesn't account for possibility of +/- in first character
                stripped_value = value[:19]
                timestamp_formats = [
                    "%Y-%m-%dT%H:%M:%S",  # 2023-10-01T14:30:45
                    "%Y-%m-%d %H:%M:%S",  # 2023-10-01 14:30:45
                    "%Y-%m-%dT%H:%M",  # 2023-10-01T14:30
                    "%Y-%m-%d %H:%M",  # 2023-10-01 14:30
                    "%Y-%m-%dT%H",  # 2023-10-01T14
                    "%Y-%m-%d %H",  # 2023-10-01 14
                    "%Y-%m-%d",  # 2023-10-01
                ]

                parsed_timestamp = None
                for fmt in timestamp_formats:
                    try:
                        parsed_timestamp = datetime.strptime(stripped_value, fmt)
                        break
                    except ValueError:
                        continue

                if parsed_timestamp is None:
                    # Fallback: just parse the date part
                    parsed_timestamp = datetime.strptime(value[:10], "%Y-%m-%d")

                row.append(parsed_timestamp)
            elif type_name == "BINARY":
                row.append(bytes(value, "utf-8"))
            else:
                row.append(value)

        rows.append(row)

    dataframe = pd.DataFrame(rows, columns=header)  # type: ignore[arg-type]: see astral-sh/ty#1714
    if return_pandas:
        return dataframe

    # Disable numparse for float columns to prevent scientific notation while preserving INT alignment
    disable_numparse = float_column_indices if float_column_indices else False

    if truncate_results:
        query_result = _truncate_result(dataframe, disable_numparse)
    else:
        query_result = dataframe.to_markdown(disable_numparse=disable_numparse)

    return (query_result or "").strip()


def _truncate_result(dataframe, disable_numparse=False):
    query_result = dataframe.to_markdown(disable_numparse=disable_numparse)
    tokens_used = _count_tokens(query_result)

    # If the full result fits, return it
    if tokens_used <= MAX_TOKENS_OF_DATA:
        return (query_result or "").strip()

    def is_too_big(n):
        return (
            _count_tokens(dataframe.iloc[:n].to_markdown(disable_numparse=disable_numparse))
            > MAX_TOKENS_OF_DATA
        )

    # Use bisect_left to find the cutoff point of rows within the max token data limit in a O(log n) complexity
    # Passing True, as this is the target value we are looking for when _is_too_big returns
    cutoff = bisect.bisect_left(range(len(dataframe) + 1), True, key=is_too_big)

    # Slice to the found limit
    truncated_df = dataframe.iloc[:cutoff]

    # Edge case: Cannot return any rows because of tokens so return an empty string
    if len(truncated_df) == 0:
        return ""

    truncated_result = truncated_df.to_markdown(disable_numparse=disable_numparse)

    # Double-check edge case if we overshot by one
    if _count_tokens(truncated_result) > MAX_TOKENS_OF_DATA:
        truncated_result = truncated_df.iloc[:-1].to_markdown(disable_numparse=disable_numparse)
    return truncated_result


def _end_current_span(client, parent_trace_id, current_span, final_state, error=None):
    """helper function to safely end a span with exception handling."""
    import mlflow

    if current_span is None:
        return None

    try:
        attributes = {"final_state": final_state}
        if error is not None:
            attributes["error"] = error

        client.end_span(
            trace_id=parent_trace_id,
            span_id=current_span.span_id,
            attributes=attributes,
        )
    except mlflow.exceptions.MlflowTracingException as e:
        logging.warning(f"Failed to end span for {final_state}: {e}")

    return None


def _parse_attachments(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Parse attachments from a Genie API response.

    Returns the final query attachment, all text attachments belonging to the final
    answer (an answer may carry several, e.g. a summary plus a follow-up), and the
    suggested-questions attachment.
    """
    result: Dict[str, Any] = {
        "query_attachment": None,
        "text_attachments": [],
        "suggested_questions_attachment": None,
    }

    attachments = resp.get("attachments") or []
    if not isinstance(attachments, list):
        return result

    # Genie may self-correct, emitting multiple query+text pairs. Text strictly
    # between the first and last query is a superseded attempt and is dropped; all
    # other text is part of the answer and kept in order.
    valid = [(i, a) for i, a in enumerate(attachments) if isinstance(a, dict)]
    query_indices = [i for i, a in valid if "query" in a]

    for i, a in valid:
        if "query" in a:
            result["query_attachment"] = a  # last query wins
        elif "text" in a:
            if query_indices and query_indices[0] < i < query_indices[-1]:
                continue
            result["text_attachments"].append(a)
        elif "suggested_questions" in a:
            result["suggested_questions_attachment"] = a

    return result


def _extract_suggested_questions_from_attachment(attachment) -> Optional[List[str]]:
    """Extract suggested follow-up questions from a Genie API response attachment."""
    if not isinstance(attachment, dict):
        return None

    sq_obj = attachment.get("suggested_questions")
    if not isinstance(sq_obj, dict):
        return None

    questions = sq_obj.get("questions")
    if not isinstance(questions, list):
        return None

    return [q for q in questions if isinstance(q, str)] or None


def _extract_text_attachment_content_from_attachments(attachments) -> Optional[str]:
    """Join the text summaries from a list of Genie API response text attachments."""
    if not isinstance(attachments, list):
        return ""

    contents: List[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        text_obj = attachment.get("text")
        if not isinstance(text_obj, dict):
            continue
        content = text_obj.get("content", "")
        if content:
            contents.append(content)

    return "\n\n".join(contents)


# --- Agent-mode helpers --------------------------------------------------------------------


def _join_reasoning(item: Dict[str, Any]) -> str:
    parts = []
    for c in item.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "reasoning_text":
            parts.append(c.get("text", ""))
    return "".join(parts)


def _load_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_citations(chunk: Dict[str, Any]) -> List[str]:
    return [
        ann["url"]
        for ann in chunk.get("annotations") or []
        if isinstance(ann, dict) and isinstance(ann.get("url"), str) and ann["url"]
    ]


@mlflow_trace(span_type="PARSER")
def _parse_agent_response(resp: Dict[str, Any]) -> GenieAgentResponse:
    """Parse a terminal agent-mode response object into a GenieAgentResponse."""
    status = resp.get("status", "completed")
    conversation_id = resp.get("conversation_id")

    if status == "failed":
        return GenieAgentResponse(
            answer="",
            conversation_id=conversation_id,
            status="failed",
            error=resp.get("error"),
        )

    steps: List[GenieAgentStep] = []
    citations: List[str] = []
    answer_chunks: List[str] = []

    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == REASONING:
            steps.append(GenieAgentStep(type=REASONING, text=_join_reasoning(item)))

        elif itype == FUNCTION_CALL:
            args = _load_arguments(item.get("arguments"))
            steps.append(
                GenieAgentStep(
                    type=FUNCTION_CALL,
                    title=args.get("title", ""),
                    sql=args.get("sql", ""),
                    call_id=item.get("call_id", ""),
                )
            )

        elif itype == FUNCTION_CALL_OUTPUT:
            steps.append(
                GenieAgentStep(
                    type=FUNCTION_CALL_OUTPUT,
                    output=item.get("output", ""),
                    call_id=item.get("call_id", ""),
                )
            )

        elif itype == MESSAGE:
            role = item.get("role", "")
            # An assistant message's text chunks (excluding rendered-table chunks) are the
            # answer; a system message carries an error string. Citations come from annotations.
            msg_text = ""
            for chunk in item.get("content") or []:
                if not isinstance(chunk, dict) or chunk.get("type") != "output_text":
                    continue
                if chunk.get("metadata"):  # rendered table chunk; already in the narrative
                    continue
                msg_text += chunk.get("text", "")
                citations.extend(_extract_citations(chunk))
            steps.append(GenieAgentStep(type=MESSAGE, text=msg_text, role=role))
            if msg_text and (role == "assistant" or (role == "system" and not answer_chunks)):
                answer_chunks.append(msg_text)

    return GenieAgentResponse(
        answer="".join(answer_chunks),
        steps=steps,
        citations=citations,
        conversation_id=conversation_id,
        status=status,
    )


def _read_chunks(stream):
    """Yield chunks from a readable stream (bytes or str), or from an iterable of chunks."""
    if hasattr(stream, "read"):
        while True:
            chunk = stream.read(1024)
            if not chunk:  # terminates on both b"" and "" at EOF
                return
            yield chunk
    else:
        yield from stream


def _iter_sse_lines(stream):
    """Yield SSE lines as they arrive from a byte/str blob, a readable stream, or a chunk iterable."""
    if isinstance(stream, (bytes, str)):
        data = stream.decode("utf-8") if isinstance(stream, bytes) else stream
        yield from data.split("\n")
        return

    buffer = ""
    for chunk in _read_chunks(stream):
        buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer:
        yield buffer


def _iter_sse_events(stream):
    """Yield parsed ``data:`` JSON objects from a Server-Sent Events stream, incrementally."""
    for line in _iter_sse_lines(stream):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            logging.debug("Skipping unparseable SSE data line")


def _agent_item_span(item: Dict[str, Any]):
    """(name, span_type, outputs) for a streamed agent timeline item's span."""
    itype = item.get("type")
    span_type = _AGENT_SPAN_TYPE_BY_ITEM.get(itype, "CHAIN")
    if itype == FUNCTION_CALL:
        args = _load_arguments(item.get("arguments"))
        name = f"function_call: {args.get('title') or 'execute_sql'}"
        return name, span_type, {"sql": args.get("sql", "")}
    if itype == FUNCTION_CALL_OUTPUT:
        return "function_call_output", span_type, {"result": item.get("output", "")}
    if itype == REASONING:
        return "reasoning", span_type, {"reasoning": _join_reasoning(item)}
    if itype == MESSAGE:
        return f"message ({item.get('role', 'assistant')})", span_type, None
    return itype or "item", span_type, None


class Genie:
    def __init__(
        self,
        space_id,
        client: Optional["WorkspaceClient"] = None,
        truncate_results=False,
        return_pandas: bool = False,
        agent_mode: bool = False,
    ):
        """A Genie client for chat mode (default) or agent mode (agent_mode=True)."""
        self.space_id = space_id
        workspace_client = client or WorkspaceClient()
        self.genie = workspace_client.genie
        self.agent_mode = agent_mode
        # Agent mode streams Server-Sent Events; chat mode exchanges JSON.
        self.headers = {
            "Accept": "text/event-stream" if agent_mode else "application/json",
            "Content-Type": "application/json",
        }
        self.truncate_results = truncate_results
        self.return_pandas = return_pandas
        # get_space is not available for agent-mode ids; skip the description lookup there.
        self.description = None if agent_mode else self.genie.get_space(space_id).description

    @mlflow_trace
    def start_conversation(self, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow_trace
    def create_message(self, conversation_id, content):
        resp = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow_trace
    def poll_for_result(self, conversation_id, message_id):
        @mlflow_trace
        def poll_query_results(
            attachment_id,
            query_str,
            description,
            conversation_id=conversation_id,
            suggested_questions=None,
            text_attachment_content=None,
        ):
            iteration_count = 0
            while iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self.genie._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result",
                    headers=self.headers,
                )["statement_response"]
                state = resp["status"]["state"]
                returned_conversation_id = resp.get("conversation_id", None)
                if state == "SUCCEEDED":
                    result = _parse_query_result(resp, self.truncate_results, self.return_pandas)
                    return GenieResponse(
                        result,
                        query_str,
                        description,
                        returned_conversation_id,
                        suggested_questions,
                        text_attachment_content,
                    )
                elif state in ["RUNNING", "PENDING"]:
                    logging.debug("Waiting for query result...")
                    time.sleep(ITERATION_FREQUENCY)
                else:
                    return GenieResponse(
                        f"No query result: {resp['state']}",
                        query_str,
                        description,
                        returned_conversation_id,
                        suggested_questions,
                        text_attachment_content,
                    )
            return GenieResponse(
                f"Genie query for result timed out after {MAX_ITERATIONS} iterations of 5 seconds",
                query_str,
                description,
                conversation_id,
                suggested_questions,
                text_attachment_content,
            )

        @mlflow_trace
        def poll_result():
            import mlflow

            iteration_count = 0

            # use MLflow client to get parent of any new spans we create from the current active span
            # (parenting keeps spans in the same trace)
            client = mlflow.tracking.MlflowClient()
            with mlflow.start_span(name="genie_timeline", span_type="CHAIN") as parent:
                parent_trace_id = parent.trace_id if parent else None
                parent_span_id = parent.span_id if parent else None

                # Track last status from API and the current child span
                last_status = None
                current_span = None

                while iteration_count < MAX_ITERATIONS:
                    iteration_count += 1
                    resp = self.genie._api.do(
                        "GET",
                        f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                        headers=self.headers,
                    )
                    returned_conversation_id = resp.get("conversation_id", None)

                    # get current status from API response
                    current_status = resp["status"]

                    # On status change: end previous span, start a new one (excluding terminal states)
                    if current_status != last_status:
                        # END previous span
                        current_span = _end_current_span(
                            client, parent_trace_id, current_span, last_status
                        )

                        # START new span for non-terminal states
                        if current_status not in TERMINAL_STATES:
                            # START new span
                            try:
                                current_span = client.start_span(
                                    name=current_status.lower(),
                                    trace_id=parent_trace_id,
                                    parent_id=parent_span_id,
                                    span_type="CHAIN",
                                    attributes={
                                        "state": current_status,
                                        "conversation_id": conversation_id,
                                        "message_id": message_id,
                                    },
                                )
                            except mlflow.exceptions.MlflowTracingException as e:
                                logging.warning(f"Failed to create span for {current_status}: {e}")
                                current_span = None

                        logging.debug(f"Status: {last_status} → {current_status}")
                        last_status = current_status

                    if current_status == "COMPLETED":
                        parsed = _parse_attachments(resp)
                        suggested_questions = _extract_suggested_questions_from_attachment(
                            parsed["suggested_questions_attachment"]
                        )
                        text_attachment_content = _extract_text_attachment_content_from_attachments(
                            parsed["text_attachments"]
                        )

                        if parsed["query_attachment"]:
                            query_obj = parsed["query_attachment"].get("query") or {}
                            attachment_id = parsed["query_attachment"].get("attachment_id")

                            if attachment_id:
                                return poll_query_results(
                                    attachment_id=attachment_id,
                                    query_str=query_obj.get("query", ""),
                                    description=query_obj.get("description", ""),
                                    suggested_questions=suggested_questions,
                                    conversation_id=returned_conversation_id,
                                    text_attachment_content=text_attachment_content,
                                )

                        # if there is no query attachment, use text attachment as result
                        return GenieResponse(
                            result=text_attachment_content or "",
                            suggested_questions=suggested_questions,
                            conversation_id=returned_conversation_id,
                            text_attachment_content=text_attachment_content,
                        )

                    elif current_status in {"CANCELLED", "QUERY_RESULT_EXPIRED"}:
                        return GenieResponse(result=f"Genie query {current_status.lower()}.")

                    elif current_status == "FAILED":
                        return GenieResponse(
                            result=f"Genie query failed with error: {resp.get('error', 'Unknown error')}"
                        )
                    # includes EXECUTING_QUERY, Genie can retry after this status
                    else:
                        logging.debug(f"Status: {current_status}")
                        time.sleep(ITERATION_FREQUENCY)  # faster poll rate

                # timeout path / end of while loop — close any open spans
                current_span = _end_current_span(
                    client,
                    parent_trace_id,
                    current_span,
                    last_status,
                )
                return GenieResponse(
                    f"Genie query timed out after {MAX_ITERATIONS} iterations of {ITERATION_FREQUENCY} seconds (total {MAX_ITERATIONS * ITERATION_FREQUENCY} seconds)",
                    conversation_id=conversation_id,
                )

        return poll_result()

    @mlflow_trace
    def ask_question(self, question, conversation_id: Optional[str] = None):
        """Ask Genie a question. Returns a GenieAgentResponse in agent mode, else a GenieResponse."""
        if self.agent_mode:
            resp = self.create_agent_response(question, conversation_id=conversation_id)
            return _parse_agent_response(resp)

        # check if a conversation_id is supplied
        # if yes, continue an existing genie conversation
        # otherwise start a new conversation
        if not conversation_id:
            resp = self.start_conversation(question)
        else:
            resp = self.create_message(conversation_id, question)
        genie_response = self.poll_for_result(resp["conversation_id"], resp["message_id"])
        if not genie_response.conversation_id:
            genie_response.conversation_id = resp["conversation_id"]
        return genie_response

    # --- Agent mode ------------------------------------------------------------------------

    @mlflow_trace
    def create_agent_response(
        self, content: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call the agent responses API, consume the SSE stream, and return the final response."""
        body: Dict[str, Any] = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                }
            ]
        }
        if conversation_id:
            body["conversation_id"] = conversation_id

        streamed = self.genie._api.do(
            "POST",
            f"/api/2.0/genie/agents/{self.space_id}/responses",
            body=body,
            headers=self.headers,
            raw=True,
        )
        stream = streamed["contents"] if isinstance(streamed, dict) else streamed
        return self._consume_agent_stream(stream)

    def _consume_agent_stream(self, stream) -> Dict[str, Any]:
        # Like chat mode's poll_result, child spans hang off a genie_timeline parent: one is
        # started per streamed item (output_item.added) and ended when it's done, so each
        # reasoning step, SQL call, result, and message reads as its own trace element. The
        # per-item span start/end calls are best-effort so a transient tracing error mid-stream
        # doesn't abort the turn.
        import mlflow

        client = mlflow.tracking.MlflowClient()
        latest, terminal = None, None
        open_spans: Dict[int, Any] = {}  # output_index -> open child span

        def _end(span, outputs=None):
            try:
                client.end_span(trace_id=trace_id, span_id=span.span_id, outputs=outputs)
            except mlflow.exceptions.MlflowTracingException as e:
                logging.warning(f"Failed to end agent span: {e}")

        with mlflow.start_span(name="genie_timeline", span_type="CHAIN") as parent:
            trace_id, parent_id = parent.trace_id, parent.span_id
            try:
                for event in _iter_sse_events(stream):
                    if not isinstance(event, dict):
                        continue
                    etype = event.get("type")
                    item = event.get("item")
                    idx = event.get("output_index")

                    if (
                        isinstance(idx, int)
                        and etype == "response.output_item.added"
                        and isinstance(item, dict)
                    ):
                        name, span_type, _ = _agent_item_span(item)
                        try:
                            open_spans[idx] = client.start_span(
                                name=name,
                                trace_id=trace_id,
                                parent_id=parent_id,
                                span_type=span_type,
                            )
                        except mlflow.exceptions.MlflowTracingException as e:
                            logging.warning(f"Failed to start agent span: {e}")
                    elif isinstance(idx, int) and etype == "response.output_item.done":
                        span = open_spans.pop(idx, None)
                        if span is not None:
                            outputs = _agent_item_span(item)[2] if isinstance(item, dict) else None
                            _end(span, outputs)

                    if isinstance(event.get("response"), dict):
                        latest = event["response"]
                        if etype in _AGENT_TERMINAL_EVENTS:
                            terminal = event["response"]
            finally:
                for span in open_spans.values():  # close any left open by a truncated stream
                    _end(span)
                if hasattr(stream, "close"):
                    stream.close()

        resolved = terminal or latest
        if resolved is None:
            raise RuntimeError("Genie agent stream ended without a response object")
        return resolved
