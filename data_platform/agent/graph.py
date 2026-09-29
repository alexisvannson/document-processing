"""
Receipts analyst agent, as a LangGraph graph:

    START -> check_freshness -> agent <-> tools -> guard -> END

- check_freshness reads ops.pipeline_runs / ops.documents before the model sees the question,
  so every answer can say how fresh the data is and what is missing.
- agent calls Gemini (google-genai SDK) with two tools; tools runs them read-only. The SDK's
  automatic function calling is off, so the graph, not the SDK, owns the loop.
- guard is deterministic: it masks anything in the answer that looks like a phone number,
  card number or email, whatever the model wrote.
"""
import json
import os
import re
from typing import Any, TypedDict

from google import genai
from google.genai import errors, types
from langgraph.graph import END, START, StateGraph

from data_platform.agent.tools import TOOLS, connect, data_status, execute, schema_description, to_json

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")  # pinned; Pro models have no free-tier quota
# Used when MODEL stays overloaded (503) after the SDK's retries, which is common on the free tier.
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
RETRIES = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=20.0, http_status_codes=[429, 500, 503])
MAX_TOOL_ROUNDS = 8

SYSTEM = """You answer questions about a dataset of shop receipts (Indonesian, amounts in IDR) that a \
pipeline extracted with OCR. Use the tools to look things up; never guess a number.

How the data is built: receipt photos go through OCR, PII redaction and rule-based extraction, then dbt \
publishes receipts whose items add up to their total in marts.fct_receipts (items in marts.fct_line_items). \
Receipts that fail a check are in marts.receipt_quarantine with failed_checks; receipts that failed \
earlier in the pipeline only appear in ops.documents with an *_failed status and an error.

Readable tables:
{schema}

Each question comes with a <data_status> block. When the latest run failed, or receipts are quarantined \
or failed, say what the answer does not cover (for example "139 of 200 receipts; 38 quarantined, 23 \
unreadable") and give the data_as_of date when it matters. If the data can't answer the question, say \
so. Keep answers short and cite doc_ids when you refer to specific receipts."""

PII_PATTERNS = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("[PHONE]", re.compile(r"(?<![\d,.])(?:\+?62|0)8\d{7,11}\b")),
]


class State(TypedDict, total=False):
    question: str
    data_status: dict[str, Any]
    messages: list[types.Content]
    rounds: int
    answer: str
    stop_reason: str


def build_graph(client=None, conn=None):
    client = client or genai.Client(http_options=types.HttpOptions(retry_options=RETRIES))  # reads GEMINI_API_KEY
    conn = conn or connect()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM.format(schema=schema_description(conn)),
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["parameters"])
            for t in TOOLS
        ])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    def check_freshness(state: State) -> State:
        status = data_status(conn)
        text = f"<data_status>\n{to_json(status)}\n</data_status>\n\n{state['question']}"
        return {"data_status": status, "messages": [types.Content(role="user", parts=[types.Part(text=text)])],
                "rounds": 0}

    def agent(state: State) -> State:
        try:
            response = client.models.generate_content(model=MODEL, contents=state["messages"], config=config)
        except errors.ServerError:
            response = client.models.generate_content(model=FALLBACK_MODEL, contents=state["messages"], config=config)
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None or not candidate.content.parts:
            # blocked (prompt_feedback / finish_reason SAFETY ...) or empty: no answer to build on
            return {"stop_reason": "blocked", "answer": "I can't help with that request."}
        # The model's content goes back unchanged (it may carry thought signatures).
        update: State = {"messages": state["messages"] + [candidate.content]}
        if response.function_calls:
            update["stop_reason"] = "tool_use"
        else:
            update["stop_reason"] = "end"
            update["answer"] = "".join(p.text for p in candidate.content.parts if p.text and not p.thought).strip()
        return update

    def tools(state: State) -> State:
        parts = []
        for part in state["messages"][-1].parts:
            if part.function_call:
                content, is_error = execute(conn, part.function_call.name, part.function_call.args or {})
                parts.append(types.Part.from_function_response(
                    name=part.function_call.name, response={"error" if is_error else "result": content}))
        # all results in one message, in call order, so parallel calls keep working
        return {"messages": state["messages"] + [types.Content(role="user", parts=parts)],
                "rounds": state["rounds"] + 1}

    def after_agent(state: State) -> str:
        if state["stop_reason"] == "tool_use":
            return "tools" if state["rounds"] < MAX_TOOL_ROUNDS else "give_up"
        return "guard"

    def give_up(state: State) -> State:
        return {"answer": f"I couldn't answer within {MAX_TOOL_ROUNDS} rounds of lookups; try a narrower question."}

    def guard(state: State) -> State:
        answer = state.get("answer", "")
        for placeholder, pattern in PII_PATTERNS:
            answer = pattern.sub(placeholder, answer)
        return {"answer": answer}

    graph = StateGraph(State)
    graph.add_node("check_freshness", check_freshness)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("give_up", give_up)
    graph.add_node("guard", guard)
    graph.add_edge(START, "check_freshness")
    graph.add_edge("check_freshness", "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "give_up", "guard"])
    graph.add_edge("tools", "agent")
    graph.add_edge("give_up", "guard")
    graph.add_edge("guard", END)
    return graph.compile()


def tool_calls(state: State):
    """(name, args) of every tool call in a finished run, for logging and tests."""
    return [(p.function_call.name, dict(p.function_call.args or {})) for m in state["messages"] if m.role == "model"
            for p in m.parts or [] if p.function_call]


def ask(question, **kwargs):
    return build_graph(**kwargs).invoke({"question": question})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ask the receipts agent a question")
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the data status and tool calls")
    args = parser.parse_args()

    state = ask(args.question)
    if args.verbose:
        print("data status:", json.dumps(state["data_status"], default=str, indent=2))
        for name, tool_input in tool_calls(state):
            print(f"tool {name}: {json.dumps(tool_input)}")
        print()
    print(state["answer"])


if __name__ == "__main__":
    main()
