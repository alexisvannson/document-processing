"""
Provisions the local Metabase (localhost:3000): the admin account on first run, a connection
to the warehouse as the read-only `analyst` role, and the "Receipts" dashboard.
Safe to re-run: the dashboard and its questions are archived and rebuilt from CARDS below.

    python data_platform/infra/metabase/setup.py
"""
import json
import os
import urllib.error
import urllib.request

URL = os.environ.get("MB_URL", "http://localhost:3000")
ADMIN_EMAIL = os.environ.get("MB_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.environ.get("MB_ADMIN_PASSWORD", "receipts-local-2026")
DATABASE_NAME = "Receipts warehouse"
DASHBOARD_NAME = "Receipts"

# Seen from the Metabase container, hence `postgres:5432`. `analyst` can read marts and ops only.
WAREHOUSE = {
    "host": "postgres", "port": 5432, "dbname": os.environ.get("POSTGRES_DB", "warehouse"),
    "user": "analyst", "password": os.environ.get("ANALYST_PASSWORD", "analyst"),
    "schema-filters-type": "inclusion", "schema-filters-patterns": "marts,ops",
}

# (name, display, SQL, visualization settings, (col, row, width, height) on a 24-column grid)
CARDS = [
    ("Published receipts", "scalar", "SELECT count(*) AS receipts FROM marts.fct_receipts", {}, (0, 0, 6, 3)),
    ("Quarantined receipts", "scalar", "SELECT count(*) AS receipts FROM marts.receipt_quarantine", {}, (6, 0, 6, 3)),
    ("Failed in the pipeline", "scalar",
     "SELECT count(*) AS receipts FROM ops.documents WHERE status LIKE '%\\_failed'", {}, (12, 0, 6, 3)),
    ("Last run", "scalar",
     "SELECT status FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1", {}, (18, 0, 6, 3)),
    ("Receipt totals (IDR)", "bar",
     """SELECT bucket, count(*) AS receipts
FROM (
  SELECT CASE
    WHEN total < 25000 THEN '1. < 25k'
    WHEN total < 50000 THEN '2. 25k-50k'
    WHEN total < 100000 THEN '3. 50k-100k'
    WHEN total < 200000 THEN '4. 100k-200k'
    ELSE '5. 200k+' END AS bucket
  FROM marts.fct_receipts
) b
GROUP BY bucket ORDER BY bucket""",
     {"graph.dimensions": ["bucket"], "graph.metrics": ["receipts"]}, (0, 3, 12, 6)),
    ("Payment methods", "pie",
     "SELECT payment_method, count(*) AS receipts FROM marts.fct_receipts GROUP BY 1 ORDER BY 2 DESC",
     {"pie.dimension": "payment_method", "pie.metric": "receipts"}, (12, 3, 12, 6)),
    ("Top items", "row",
     """SELECT name_normalized AS item, count(DISTINCT doc_id) AS receipts
FROM marts.fct_line_items
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10""",
     {"graph.dimensions": ["item"], "graph.metrics": ["receipts"]}, (0, 9, 12, 7)),
    ("Why receipts are quarantined", "row",
     """SELECT unnest(failed_checks) AS check_failed, count(*) AS receipts
FROM marts.receipt_quarantine GROUP BY 1 ORDER BY 2 DESC""",
     {"graph.dimensions": ["check_failed"], "graph.metrics": ["receipts"]}, (12, 9, 12, 7)),
    ("Pipeline runs", "table",
     """SELECT run_id, status, failed_step, started_at, finished_at,
       stats -> 'extract' ->> 'ok' AS extracted, stats -> 'extract' ->> 'failed' AS extract_failed
FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 10""", {}, (0, 16, 24, 6)),
]


def api(method, path, body=None, session=None):
    req = urllib.request.Request(
        URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **({"X-Metabase-Session": session} if session else {})},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:500]}")
    return json.loads(raw) if raw else None


def login():
    props = api("GET", "/api/session/properties")
    if not props.get("has-user-setup"):
        api("POST", "/api/setup", {
            "token": props["setup-token"],
            "user": {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD, "first_name": "Admin", "last_name": "Local",
                     "site_name": "Receipts"},
            "prefs": {"site_name": "Receipts", "site_locale": "en", "allow_tracking": False},
        })
        print(f"Created the Metabase admin {ADMIN_EMAIL}")
    return api("POST", "/api/session", {"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD})["id"]


def warehouse_id(session):
    for db in api("GET", "/api/database", session=session)["data"]:
        if db["name"] == DATABASE_NAME:
            return db["id"]
    db = api("POST", "/api/database", {"engine": "postgres", "name": DATABASE_NAME, "details": WAREHOUSE},
             session=session)
    print(f"Connected Metabase to the warehouse as `analyst` (database {db['id']})")
    return db["id"]


def rebuild_dashboard(session, db_id):
    for dash in api("GET", "/api/dashboard", session=session):
        if dash["name"] == DASHBOARD_NAME and not dash.get("archived"):
            for dashcard in api("GET", f"/api/dashboard/{dash['id']}", session=session).get("dashcards", []):
                if dashcard.get("card_id"):
                    api("PUT", f"/api/card/{dashcard['card_id']}", {"archived": True}, session=session)
            api("PUT", f"/api/dashboard/{dash['id']}", {"archived": True}, session=session)

    dashboard = api("POST", "/api/dashboard", {
        "name": DASHBOARD_NAME,
        "description": "Receipts published by dbt, what was quarantined and why, and how the pipeline runs went.",
    }, session=session)
    dashcards = []
    for i, (name, display, sql, viz, (col, row, width, height)) in enumerate(CARDS):
        card = api("POST", "/api/card", {
            "name": name, "display": display, "visualization_settings": viz,
            "dataset_query": {"type": "native", "native": {"query": sql}, "database": db_id},
        }, session=session)
        dashcards.append({"id": -(i + 1), "card_id": card["id"], "col": col, "row": row,
                          "size_x": width, "size_y": height})
    api("PUT", f"/api/dashboard/{dashboard['id']}", {"dashcards": dashcards}, session=session)
    return dashboard["id"]


def main():
    session = login()
    db_id = warehouse_id(session)
    dashboard_id = rebuild_dashboard(session, db_id)
    print(f"Dashboard: {URL}/dashboard/{dashboard_id}  (login {ADMIN_EMAIL} / {ADMIN_PASSWORD})")


if __name__ == "__main__":
    main()
