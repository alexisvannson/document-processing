import os

import psycopg

DEFAULT_URL = "postgresql://pipeline:pipeline@localhost:5433/warehouse"


def connect():
    """Autocommit connection: each document gets its own `with conn.transaction()` block."""
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL), autocommit=True)


def process_documents(conn, run_id, from_status, to_status, fail_status, fn, outputs):
    """
    Runs fn(conn, doc) on every document in `from_status`, one transaction per document.
    Success moves it to `to_status`; an exception rolls back its partial writes and parks
    it in `fail_status` with the error, so one bad receipt never blocks the others.
    On failure the document's rows in `outputs` (the step's tables) are also deleted, so
    output from an earlier successful run can't linger next to a failed status.
    """
    docs = conn.execute(
        "SELECT doc_id, file_name, image_path FROM ops.documents WHERE status = %s ORDER BY doc_id",
        (from_status,),
    ).fetchall()

    ok = failed = 0
    for doc_id, file_name, image_path in docs:
        doc = {"doc_id": doc_id, "file_name": file_name, "image_path": image_path}
        try:
            with conn.transaction():
                fn(conn, doc)
                set_status(conn, doc_id, run_id, to_status)
            ok += 1
        except Exception as e:
            with conn.transaction():
                for table in outputs:
                    conn.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))
                set_status(conn, doc_id, run_id, fail_status, error=f"{type(e).__name__}: {e}")
            failed += 1
    return {"ok": ok, "failed": failed}


def set_status(conn, doc_id, run_id, status, error=None):
    conn.execute(
        "UPDATE ops.documents SET status = %s, error = %s, last_run_id = %s, updated_at = now()"
        " WHERE doc_id = %s",
        (status, error, run_id, doc_id),
    )
