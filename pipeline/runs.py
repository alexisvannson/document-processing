"""
Bookkeeping in ops.pipeline_runs. Used by the CLI (pipeline/run.py) and by the Airflow DAG,
whose callbacks run in Airflow's own environment: keep this module's imports to psycopg.
"""
from psycopg.types.json import Jsonb


def start_run(conn, run_id, params=None):
    """Registers the run; a no-op when it already exists (an Airflow task retried)."""
    conn.execute(
        "INSERT INTO ops.pipeline_runs (run_id, params) VALUES (%s, %s) ON CONFLICT (run_id) DO NOTHING",
        (run_id, Jsonb(params or {})),
    )


def record_step(conn, run_id, step, stats):
    conn.execute(
        "UPDATE ops.pipeline_runs SET stats = coalesce(stats, '{}') || jsonb_build_object(%s::text, %s::jsonb)"
        " WHERE run_id = %s",
        (step, Jsonb(stats), run_id),
    )


def finish_run(conn, run_id):
    conn.execute(
        "UPDATE ops.pipeline_runs SET status = 'success', finished_at = now() WHERE run_id = %s", (run_id,)
    )


def fail_run(conn, run_id, step, error):
    conn.execute(
        "UPDATE ops.pipeline_runs SET status = 'failed', failed_step = %s, error = %s, finished_at = now()"
        " WHERE run_id = %s",
        (step, error, run_id),
    )
