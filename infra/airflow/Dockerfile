FROM apache/airflow:3.3.2-python3.12

# Airflow orchestrates; the work runs in two venvs so their dependencies can't clash with
# Airflow's own: one for the pipeline steps, one for dbt.
COPY pipeline/requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /opt/airflow/venvs/pipeline \
 && /opt/airflow/venvs/pipeline/bin/pip install --no-cache-dir -r /tmp/pipeline-requirements.txt \
 && python -m venv /opt/airflow/venvs/dbt \
 && /opt/airflow/venvs/dbt/bin/pip install --no-cache-dir "dbt-core==1.12.5" "dbt-postgres==1.11.0" \
 && pip install --no-cache-dir "psycopg[binary]>=3.1"  # for the DAG's run bookkeeping (pipeline/runs.py)
