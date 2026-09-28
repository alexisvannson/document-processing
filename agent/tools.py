"""
The agent's tools. They connect as the read-only `analyst` role (SELECT on marts and ops only,
see infra/init/02_roles.sh), inside read-only transactions with a statement timeout, so even
a bad query can't write, read raw_restricted, or hang the agent.
"""
import json
import os
from decimal import Decimal

import psycopg

DEFAULT_URL = "postgresql://analyst:analyst@localhost:5433/warehouse"
MAX_ROWS = 100


def connect():
    conn = psycopg.connect(os.environ.get("AGENT_DATABASE_URL", DEFAULT_URL), autocommit=True)
    conn.execute("SET default_transaction_read_only = on")
    conn.execute("SET statement_timeout = '5s'")
    return conn


def to_json(value):
    return json.dumps(value, default=lambda v: float(v) if isinstance(v, Decimal) else str(v))


def fetch(conn, query, params=None, limit=MAX_ROWS):
    with conn.transaction():
        cur = conn.execute(query, params)
        columns = [c.name for c in cur.description]
        rows = cur.fetchmany(limit + 1)
    return columns, rows[:limit], len(rows) > limit


def schema_description(conn):
    """The tables the agent may query, from information_schema, in a stable order (for the system prompt)."""
    _, rows, _ = fetch(conn, """
        SELECT table_schema || '.' || table_name, string_agg(column_name || ' ' || data_type, ', ' ORDER BY ordinal_position)
        FROM information_schema.columns
        WHERE table_schema IN ('marts', 'ops')
        GROUP BY table_schema, table_name
        ORDER BY 1
    """, limit=1000)
    return "\n".join(f"- {table}({columns})" for table, columns in rows)


def data_status(conn):
    """Freshness of the data: latest run, latest successful run, receipts not in fct_receipts."""
    _, [latest], _ = fetch(conn, """
        SELECT run_id, status, failed_step, error, started_at, finished_at
        FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1
    """)
    _, success, _ = fetch(conn, """
        SELECT run_id, finished_at FROM ops.pipeline_runs
        WHERE status = 'success' ORDER BY finished_at DESC LIMIT 1
    """)
    _, docs, _ = fetch(conn, "SELECT status, count(*) FROM ops.documents GROUP BY status ORDER BY status")
    _, [[published]], _ = fetch(conn, "SELECT count(*) FROM marts.fct_receipts")
    _, [[quarantined]], _ = fetch(conn, "SELECT count(*) FROM marts.receipt_quarantine")
    run_id, status, failed_step, error, started_at, finished_at = latest
    return {
        "latest_run": {"run_id": run_id, "status": status, "failed_step": failed_step, "error": error,
                       "started_at": started_at, "finished_at": finished_at},
        "data_as_of": success[0][1] if success else None,
        "last_run_failed": status == "failed",
        "documents_by_status": dict(docs),
        "published_receipts": published,
        "quarantined_receipts": quarantined,
    }


def run_sql(conn, query):
    columns, rows, truncated = fetch(conn, query)
    result = {"columns": columns, "rows": rows}
    if truncated:
        result["note"] = f"only the first {MAX_ROWS} rows are shown; aggregate in SQL instead"
    return to_json(result)


def get_receipt(conn, doc_id):
    """Everything the agent may see about one receipt, whichever state it is in."""
    _, doc, _ = fetch(conn, "SELECT doc_id, file_name, status, error, ingested_at FROM ops.documents WHERE doc_id = %s",
                      (doc_id,))
    if not doc:
        return to_json({"error": f"no receipt with doc_id {doc_id}"})
    columns, published, _ = fetch(conn, "SELECT * FROM marts.fct_receipts WHERE doc_id = %s", (doc_id,))
    result = {"document": dict(zip(["doc_id", "file_name", "status", "error", "ingested_at"], doc[0]))}
    if published:
        result["published"] = dict(zip(columns, published[0]))
        item_cols, items, _ = fetch(conn, "SELECT line_idx, name, qty, unit_price, amount FROM marts.fct_line_items"
                                          " WHERE doc_id = %s ORDER BY line_idx", (doc_id,))
        result["items"] = [dict(zip(item_cols, row)) for row in items]
    else:
        columns, quarantined, _ = fetch(conn, "SELECT * FROM marts.receipt_quarantine WHERE doc_id = %s", (doc_id,))
        if quarantined:
            result["quarantined"] = dict(zip(columns, quarantined[0]))
    return to_json(result)


# Tool definitions (name, description, JSON Schema of the arguments), sent to the model.
TOOLS = [
    {
        "name": "run_sql",
        "description": (
            "Run one read-only PostgreSQL SELECT against the warehouse and get back columns and rows (at most "
            f"{MAX_ROWS}). Only the marts and ops schemas are readable. Aggregate in SQL rather than fetching "
            "rows to count them. Amounts are in IDR."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"],
        },
    },
    {
        "name": "get_receipt",
        "description": (
            "Look up one receipt by doc_id: its pipeline status and error, and either its published totals and "
            "items or, if it was quarantined, the checks it failed and its amounts."
        ),
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "integer"}},
            "required": ["doc_id"],
        },
    },
]


def execute(conn, name, tool_input):
    """Runs a tool call; returns (content, is_error). Errors go back to the model, not up the stack."""
    try:
        if name == "run_sql":
            return run_sql(conn, tool_input["query"]), False
        if name == "get_receipt":
            return get_receipt(conn, int(tool_input["doc_id"])), False
        return f"Unknown tool {name}", True
    except (psycopg.Error, KeyError, ValueError, TypeError) as e:
        return f"{type(e).__name__}: {e}", True
