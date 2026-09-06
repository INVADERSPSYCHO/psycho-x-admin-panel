import os
import secrets
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from starlette.responses import JSONResponse
import requests
import pyarrow.parquet as pq
import io

# ── CONFIG ──────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "psychoxd")      # Master key
MASTER_KEY = os.environ.get("MASTER_KEY", "admin123") # Admin password for /apikey
DEVELOPER = "@psychopathmc"
SUPPORT_MSG = "For API purchase, contact @psychopathmc"
BASE_URL = "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main"
CACHE_TTL = 300

app = FastAPI(title="PsychopathMC OSINT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(FastAPIHTTPException)
async def custom_http_exception_handler(request: Request, exc: FastAPIHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "developer": DEVELOPER,
            "support": SUPPORT_MSG,
        }
    )

# ── Helper: Circle Lookup ──────────────────────────────
def get_circle(num: str) -> str:
    prefixes = {
        "9810": "AIRTEL DELHI", "9871": "AIRTEL DELHI", "9818": "AIRTEL DELHI",
        "9910": "VI DELHI", "8826": "JIO DELHI", "9999": "AIRTEL DELHI",
        "9971": "AIRTEL DELHI", "9883": "JIO WB", "9564": "JIO WB",
    }
    pref = num[:4]
    return prefixes.get(pref, "UNKNOWN CIRCLE")

# ── Cache ────────────────────────────────────────────────
_cache = {}
_cache_time = {}

def get_cache(url, column, value):
    key = f"{url}|{column}|{value}"
    if key in _cache and (datetime.now() - _cache_time[key]).seconds < CACHE_TTL:
        return _cache[key]
    return None

def set_cache(url, column, value, data):
    key = f"{url}|{column}|{value}"
    _cache[key] = data
    _cache_time[key] = datetime.now()
    if len(_cache) > 100:
        oldest = min(_cache_time, key=_cache_time.get)
        del _cache[oldest]
        del _cache_time[oldest]

# ── Fetch Data ───────────────────────────────────────────
def fetch_data(url: str, column: str, value: str, limit: int = 15):
    cached = get_cache(url, column, value)
    if cached is not None:
        return cached

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        needed_cols = ["name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber", "address"]
        table = pq.read_table(io.BytesIO(resp.content), columns=needed_cols)
        df = table.to_pandas()
        if column not in df.columns:
            return []
        filtered = df[df[column] == value]
        results = filtered.head(limit).to_dict(orient="records")
        set_cache(url, column, value, results)
        return results
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

# ── In-Memory Key Storage ──────────────────────────────
_generated_keys = {}   # key -> { limit, expiry, usage, created }
_admin_session = {}    # session_id -> expiry

# ── Endpoints ────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "PsychoAPI is live. Use /search?q=number&key=psychoxd",
        "developer": DEVELOPER,
        "support": SUPPORT_MSG,
    }

@app.get("/health")
def health():
    return {"status": "ok", "developer": DEVELOPER, "support": SUPPORT_MSG}

@app.get("/search")
def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    key: str = Query(..., description="API Key required"),
    limit: int = Query(5, ge=1, le=20)
):
    # Check master key or generated keys
    if key == API_KEY:
        pass
    elif key in _generated_keys:
        expiry = _generated_keys[key].get("expiry")
        if expiry and datetime.now() > expiry:
            raise HTTPException(status_code=401, detail="API key expired")
        usage = _generated_keys[key].get("usage", 0)
        lim = _generated_keys[key].get("limit", 0)
        if lim > 0 and usage >= lim:
            raise HTTPException(status_code=429, detail="API key limit exceeded")
        _generated_keys[key]["usage"] = usage + 1
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

    query = (q or mobile or "").strip()
    if not query:
        raise HTTPException(422, "Provide q or mobile")

    last_digit = query[-1]
    shard = int(last_digit) % 7

    phone_url = f"{BASE_URL}/idx_phone.{shard}.parquet"
    results = fetch_data(phone_url, "phoneNumber", query, limit)

    if not results:
        aadhar_url = f"{BASE_URL}/idx_aadhar.{shard}.parquet"
        results = fetch_data(aadhar_url, "aadharNumber", query, limit)

    # Deduplicate
    seen = set()
    unique_results = []
    for row in results:
        aadhar_val = row.get("aadharNumber")
        if aadhar_val not in seen:
            seen.add(aadhar_val)
            unique_results.append(row)
    results = unique_results

    return {
        "success": len(results) > 0,
        "query": query,
        "count": len(results),
        "results": results,
        "developer": DEVELOPER,
        "support": SUPPORT_MSG,
    }

# ── API Key Management Dashboard ─────────────────────────
@app.get("/apikey/login", response_class=HTMLResponse)
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

@app.post("/apikey/login")
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

@app.get("/apikey/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _admin_session or datetime.now() > _admin_session[session_id]:
        return RedirectResponse(url="/apikey/login", status_code=302)

    rows = ""
    for k, v in _generated_keys.items():
        expiry = v.get("expiry")
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

@app.post("/apikey/create")
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

@app.post("/apikey/delete")
async def delete_key(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in _admin_session or datetime.now() > _admin_session[session_id]:
        return RedirectResponse(url="/apikey/login", status_code=302)
    form = await request.form()
    key_to_delete = form.get("key")
    if key_to_delete in _generated_keys:
        del _generated_keys[key_to_delete]
    return RedirectResponse(url="/apikey/dashboard", status_code=302)

@app.get("/apikey/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id in _admin_session:
        del _admin_session[session_id]
    response = RedirectResponse(url="/apikey/login", status_code=302)
    response.delete_cookie("session_id")
    return response
