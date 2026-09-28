"""
Receipts pipeline, daily: ingest -> ocr -> redact -> extract -> dbt build.

Each step is the pipeline CLI (`python -m pipeline.run --steps <step>`) run in its own venv
under this DAG run's id, so ops.pipeline_runs has one row per DAG run. Airflow owns that
row: `start_run` opens it, `finish_run` closes it, and a task that fails for good (after its
retry) marks it failed with the task's name, which is what the agent will read to say how
fresh the data is. A failing dbt test stops the marts from being rebuilt, so dashboards keep
the last good data.
"""
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import dag, task

from pipeline.db import connect
from pipeline.runs import fail_run, finish_run, start_run

PROJECT = "/opt/airflow/project"
PIPELINE_PY = "/opt/airflow/venvs/pipeline/bin/python"
DBT = "/opt/airflow/venvs/dbt/bin/dbt"


def record_failure(context):
    ti = context["task_instance"]
    error = str(context.get("exception") or "task failed")
    with connect() as conn:
        fail_run(conn, context["run_id"], ti.task_id, error[:2000])


@dag(
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1), "on_failure_callback": record_failure},
    tags=["receipts"],
    doc_md=__doc__,
)
def receipts_daily():
    @task
    def start(run_id=None):
        with connect() as conn:
            start_run(conn, run_id, {"orchestrator": "airflow"})

    def step(name):
        return BashOperator(
            task_id=name,
            cwd=PROJECT,
            bash_command=f"{PIPELINE_PY} -m pipeline.run --steps {name} --run-id '{{{{ run_id }}}}'",
        )

    dbt_build = BashOperator(
        task_id="dbt_build",
        cwd=PROJECT,
        # target/ and logs/ go to /tmp: the project is mounted read-only
        bash_command=f"{DBT} build --project-dir dbt --profiles-dir dbt"
                     " --target-path /tmp/dbt/target --log-path /tmp/dbt/logs",
    )

    @task
    def finish(run_id=None):
        with connect() as conn:
            finish_run(conn, run_id)

    start() >> step("ingest") >> step("ocr") >> step("redact") >> step("extract") >> dbt_build >> finish()


receipts_daily()
