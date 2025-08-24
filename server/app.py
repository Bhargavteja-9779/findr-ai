from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse

from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from pydantic import BaseModel
from server import db
import qrcode, io, os
from pathlib import Path

app = FastAPI()

# Init DB (NO seeding now)
#db.init_db(seed=False)

@app.on_event("startup")
def on_startup():
    db.init_db(seed=False)


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
    items = db.list_items(states=["RED"])  # only RED items on public screen
    return templates.TemplateResponse("billboard.html", {"request": request, "items": items, "title": "Billboard"})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    items = db.list_items()
    # discover local videos
    vids_dir = Path("server/static/videos")
    videos = []
    if vids_dir.exists():
        for p in sorted(vids_dir.glob("*.mp4")):
            videos.append({"name": p.stem, "url": f"/static/videos/{p.name}"})
    return templates.TemplateResponse("dashboard.html", {"request": request, "items": items, "videos": videos, "title": "Dashboard"})

@app.get("/i/{shortid}", response_class=HTMLResponse)
def item_page(shortid: str, request: Request):
    it = db.get_item(shortid)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")
    return templates.TemplateResponse("item.html", {"request": request, "item": it, "title": f"Item {shortid}"})

@app.get("/q/{shortid}")
def qr(shortid: str):
    """PNG QR code for /i/{shortid} (uses FINDR_BASE_URL if set)."""
    base = os.getenv("FINDR_BASE_URL", "http://127.0.0.1:8123")
    url = f"{base}/i/{shortid}"
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

@app.post("/api/items/{shortid}/resolve")
def api_resolve_item(shortid: str, reason: str = Body(default="resolved by staff")):
    ok = db.resolve_item(shortid, reason=reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True, "shortid": shortid}

@app.post("/api/preview/{cam}")
async def api_preview(cam: str, request: Request):
    """
    Receive a JPEG (body bytes) and write to /static/previews/{cam}.jpg.
    Sent by the vision worker. Returns 'ok' text.
    """
    try:
        data = await request.body()
        if not data or len(data) < 100:  # sanity check
            raise HTTPException(status_code=400, detail="empty frame")
        out = Path("server/static/previews") / f"{cam}.jpg"
        out.write_bytes(data)
        return PlainTextResponse("ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))