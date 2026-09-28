#!/bin/bash
# Read-only role for dashboards (Metabase) and the agent. It sees the published marts and the
# pipeline's bookkeeping, never `raw` or `raw_restricted`. dbt grants SELECT on each mart table
# it builds (dbt_project.yml), so the grant survives rebuilds.
# A shell script rather than .sql so the password comes from ANALYST_PASSWORD (.env).
set -euo pipefail

psql -v ON_ERROR_STOP=1 -v analyst_password="${ANALYST_PASSWORD:-analyst}" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE analyst LOGIN PASSWORD :'analyst_password';

CREATE SCHEMA IF NOT EXISTS marts;
GRANT USAGE ON SCHEMA marts TO analyst;

GRANT USAGE ON SCHEMA ops TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA ops TO analyst;
SQL
