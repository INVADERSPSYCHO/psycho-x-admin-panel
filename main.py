import asyncio
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from datetime import datetime, timedelta
import secrets

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Response, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

# 🔥 Master API key for admin login
MASTER_KEY = os.environ.get("MASTER_KEY", "admin123")
API_KEY = os.environ.get("API_KEY", "tnum1906")

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

IDX_PHONE = "idx_phone"
IDX_AADHAR = "idx_aadhar"

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── In-memory storage for generated keys ──────────────────────────────────
_generated_keys = {}
_admin_session = {}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con

def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid

def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]

# ── Dedup & Connected Records ───────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()

def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        elif field == "otherNumber":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if is_num:
        all_rows = []
        searched = []
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q, "searched_fields": searched,
            "count": len(all_rows), "results": all_rows,
        }
    else:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

# ── FastAPI (Branded as PSYCHOMC OSINT) ──────────────────────────────────────
fastapi_app = FastAPI(title="PSYCHOMC OSINT API", docs_url="/docs")   # ✅ ICMR HATA DIYA

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

# ---------- Existing Endpoints ----------
@fastapi_app.get("/")
def root():
    return {
        "app": "PSYCHOMC OSINT API",   # ✅ ICMR HATA DIYA
        "records": 2_504_793_870,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@psychopathmc",   # 🔥 TERA CREDIT
    }

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "raw_database_required": False,
            "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
            "index_source": INDEX_SOURCE}

@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
    key: str = Query(..., description="API Key required"),
):
    if key == API_KEY:
        pass
    elif key in _generated_keys:
        expiry = _generated_keys[key].get("expiry")
        if expiry and datetime.now() > expiry:
            raise HTTPException(status_code=401, detail="API key expired")
        usage = _generated_keys[key].get("usage", 0)
        limit_val = _generated_keys[key].get("limit", 0)
        if limit_val > 0 and usage >= limit_val:
            raise HTTPException(status_code=429, detail="API key limit exceeded")
        _generated_keys[key]["usage"] = usage + 1
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    result = {"success": bool(data["count"]), **data, "number": q_val,
              "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _run_field_search,
                             item.get("field", "phoneNumber"),
                             item.get("value", ""),
                             item.get("mode", "exact"),
                             int(item.get("limit", req.limit)))
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(content=json.dumps({"searches": len(req.queries), "results": list(results)},
                                       indent=2, ensure_ascii=False),
                    media_type="application/json")

# ── API Key Management System ──────────────────────────────────────────────
@fastapi_app.get("/apikey/login", response_class=HTMLResponse)
async def login_page():
    return """
    <html>
        <head><title>Admin Login</title></head>
        <body>
            <h2>🔐 Admin Login</h2>
            <form action="/apikey/login" method="post">
                <label>Master Key: <input type="password" name="master_key" /></label><br/>
                <input type="submit" value="Login" />
            </form>
            <p><i>Default master key: admin123</i></p>
        </body>
    </html>
    """

@fastapi_app.post("/apikey/login")
async def login_post(request: Request):
    form = await request.form()
    master_key = form.get("master_key")
    if master_key != MASTER_KEY:
        return HTMLResponse("<h3>❌ Invalid master key. <a href='/apikey/login'>Try again</a></h3>", status_code=401)
    session_id = secrets.token_urlsafe(16)
    _admin_session[session_id] = datetime.now() + timedelta(hours=1)
    response = RedirectResponse(url="/apikey/dashboard", status_code=302)
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@fastapi_app.get("/apikey/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _admin_session or datetime.now() > _admin_session[session_id]:
        return RedirectResponse(url="/apikey/login", status_code=302)

    rows = ""
    for k, v in _generated_keys.items():
        expiry = v.get("expiry", "N/A")
        if expiry and isinstance(expiry, datetime):
            expiry = expiry.strftime("%Y-%m-%d %H:%M")
        rows += f"""
        <tr>
            <td><code>{k}</code></td>
            <td>{v.get('limit', '∞')}</td>
            <td>{expiry}</td>
            <td>{v.get('usage', 0)}</td>
            <td>
                <form action="/apikey/delete" method="post" style="display:inline;">
                    <input type="hidden" name="key" value="{k}" />
                    <input type="submit" value="Delete" onclick="return confirm('Delete this key?')" />
                </form>
            </td>
        </tr>
        """
    html = f"""
    <html>
        <head><title>API Key Management</title></head>
        <body>
            <h2>📋 Generated API Keys</h2>
            <table border="1" cellpadding="8">
                <tr><th>Key</th><th>Limit (req/day)</th><th>Expiry</th><th>Usage</th><th>Action</th></tr>
                {rows if rows else "<tr><td colspan='5'>No keys created yet.</td></tr>"}
            </table>
            <hr/>
            <h3>🔑 Create New Key</h3>
            <form action="/apikey/create" method="post">
                <label>Daily Limit (0 = unlimited): <input type="number" name="limit" value="0" /></label><br/>
                <label>Expiry (days from now): <input type="number" name="days" value="30" /></label><br/>
                <input type="submit" value="Generate Key" />
            </form>
            <p><a href="/apikey/logout">Logout</a></p>
        </body>
    </html>
    """
    return HTMLResponse(html)

@fastapi_app.post("/apikey/create")
async def create_key(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _admin_session or datetime.now() > _admin_session[session_id]:
        return RedirectResponse(url="/apikey/login", status_code=302)
    form = await request.form()
    limit = int(form.get("limit", 0))
    days = int(form.get("days", 30))
    new_key = f"ps_{secrets.token_urlsafe(12)}"
    expiry = datetime.now() + timedelta(days=days)
    _generated_keys[new_key] = {
        "limit": limit,
        "expiry": expiry,
        "usage": 0,
        "created": datetime.now()
    }
    return RedirectResponse(url="/apikey/dashboard", status_code=302)

@fastapi_app.post("/apikey/delete")
async def delete_key(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _admin_session or datetime.now() > _admin_session[session_id]:
        return RedirectResponse(url="/apikey/login", status_code=302)
    form = await request.form()
    key_to_delete = form.get("key")
    if key_to_delete in _generated_keys:
        del _generated_keys[key_to_delete]
    return RedirectResponse(url="/apikey/dashboard", status_code=302)

@fastapi_app.get("/apikey/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id in _admin_session:
        del _admin_session[session_id]
    response = RedirectResponse(url="/apikey/login", status_code=302)
    response.delete_cookie("session_id")
    return response

# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"[Pinger] OK")
                else:
                    print(f"[Pinger] Unexpected status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())

# ── Gradio UI (Branding Updated) ──────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found** for this number."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(
        title="PSYCHOMC OSINT API",   # ✅ ICMR HATA DIYA
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """
    ) as demo:
        gr.Markdown("# 🔍 PSYCHOMC OSINT API", elem_classes="main-title")   # ✅ ICMR HATA DIYA
        gr.Markdown("Search **2.5 billion records** — phone, Aadhaar, name, address & more", elem_classes="subtitle")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone number, Aadhaar, ya name daalo...",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results",
                )

        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output,
        )

        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints** (via FastAPI):
- `GET /search?q=<number>&key=YOUR_KEY` — Phone/Aadhaar search
- `GET /search?mobile=<number>&key=YOUR_KEY` — Phone search (alias)
- `GET /health` — Health check
- `GET /docs` — Swagger UI

**Data Source:** [Public HF Dataset](https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed)
            """)

        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "👨‍💻 **Developer:** @psychopathmc  |  📢 **Channel:** @psychodagoated"
            "</div>",
            elem_classes="footer"
        )

    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
