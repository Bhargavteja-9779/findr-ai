from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from pydantic import BaseModel
from server import db
import qrcode
import io

app = FastAPI()

# Init DB (creates tables and seeds demo data if empty)
db.init_db(seed=True)

# Static + templates
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

# Health
@app.get("/health")
def health():
    return {"ok": True}

# ----- Web pages -----
@app.get("/billboard", response_class=HTMLResponse)
def billboard(request: Request):
    items = db.list_items(states=["RED"])  # only RED items
    return templates.TemplateResponse(
        "billboard.html",
        {"request": request, "items": items, "title": "Billboard"}
    )

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    items = db.list_items()
    return templates.TemplateResponse("dashboard.html", {"request": request, "items": items, "title": "Dashboard"})

@app.get("/i/{shortid}", response_class=HTMLResponse)
def item_page(shortid: str, request: Request):
    it = db.get_item(shortid)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    return templates.TemplateResponse("item.html", {"request": request, "item": it, "title": f"Item {shortid}"})

@app.get("/q/{shortid}")
def qr(shortid: str):
    """PNG QR code for /i/{shortid} (local demo)."""
    url = f"http://127.0.0.1:8123/i/{shortid}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# ----- Minimal APIs (Vision worker calls) -----
class UpsertItem(BaseModel):
    shortid: str
    type: str
    color: Optional[str] = None
    zone: str
    state: str   # WITH_OWNER, AMBER, RED, RECOVERED
    reason: Optional[str] = None
    crop_path: Optional[str] = None

@app.get("/api/items")
def api_list_items(state: Optional[str] = None):
    items = db.list_items(states=[state] if state else None)
    return {"items": items}

@app.get("/api/items/{shortid}")
def api_get_item(shortid: str):
    it = db.get_item(shortid)
    if not it: 
        raise HTTPException(status_code=404, detail="Item not found")
    return it

@app.post("/api/items")
def api_upsert_item(payload: UpsertItem):
    db.upsert_item(payload.dict())
    return {"ok": True}

@app.get("/q/{shortid}")
def qr(shortid: str):
    """
    PNG QR code for /i/{shortid}.
    Uses FINDR_BASE_URL env var if set, else localhost.
    """
    import os, io, qrcode
    base = os.getenv("FINDR_BASE_URL", "http://127.0.0.1:8123")
    url = f"{base}/i/{shortid}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")