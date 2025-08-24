from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from pydantic import BaseModel
from server import db

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
    items = db.list_items(states=["AMBER", "RED", "RECOVERED"])  # demo: show all three
    return templates.TemplateResponse("billboard.html", {"request": request, "items": items, "title": "Billboard"})

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

# ----- Minimal APIs (Vision worker will call these later) -----
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
    if not it: raise HTTPException(status_code=404, detail="Item not found")
    return it

@app.post("/api/items")
def api_upsert_item(payload: UpsertItem):
    db.upsert_item(payload.dict())
    return {"ok": True}

@app.post("/api/items/{shortid}/resolve")
def api_resolve_item(shortid: str, reason: str = Body(default="resolved by staff")):
    ok = db.resolve_item(shortid, reason=reason)
    if not ok: raise HTTPException(status_code=404, detail="Item not found")
    return {"ok": True, "shortid": shortid}
