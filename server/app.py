from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from pydantic import BaseModel
from server import db
from pathlib import Path
import qrcode, io, os, json, requests, traceback

app = FastAPI()

# Ollama config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

@app.on_event("startup")
def on_startup():
    db.init_db(seed=False)
    Path("server/static/previews").mkdir(parents=True, exist_ok=True)
    Path("server/static/crops").mkdir(parents=True, exist_ok=True)

# Static + templates
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

# Health
@app.get("/health")
def health():
    return {"ok": True}

# ----- Pages -----
@app.get("/billboard", response_class=HTMLResponse)
def billboard(request: Request):
    items = db.list_items(states=["RED"])
    return templates.TemplateResponse("billboard.html", {"request": request, "items": items, "title": "Billboard"})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    items = db.list_items()
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
    base = os.getenv("FINDR_BASE_URL", "http://127.0.0.1:8123")
    url = f"{base}/i/{shortid}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# ----- APIs -----
class UpsertItem(BaseModel):
    shortid: str
    type: str
    color: Optional[str] = None
    zone: str
    state: str
    reason: Optional[str] = None
    crop_path: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None

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
    try:
        data = await request.body()
        if not data or len(data) < 100:
            raise HTTPException(status_code=400, detail="empty frame")
        out = Path("server/static/previews") / f"{cam}.jpg"
        out.write_bytes(data)
        return PlainTextResponse("ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Clear All (DB + files), returns counts ----------
@app.post("/api/clear_all")
def api_clear_all():
    try:
        items_before, events_before = db.hard_reset()
        # clear crops & previews
        cleared = {"crops": 0, "previews": 0}
        for name, d in (("crops", Path("server/static/crops")), ("previews", Path("server/static/previews"))):
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*"):
                try:
                    f.unlink()
                    cleared[name] += 1
                except Exception:
                    pass
        return {
            "ok": True,
            "db_dropped": True,
            "items_before": items_before,
            "events_before": events_before,
            "files_cleared": cleared,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"clear_all failed: {e}")

# -------- Describe via Ollama ----------
@app.post("/api/items/{shortid}/describe")
def api_describe(shortid: str):
    it = db.get_item(shortid)
    if not it:
        raise HTTPException(status_code=404, detail="Item not found")

    sys_prompt = (
      "You are generating a very short lost-and-found description and keywords.\n"
      "Return STRICT JSON: {\"description\": \"...\", \"keywords\": [\"...\", \"...\"]}\n"
      "Description <= 18 words. 5-8 lowercase keywords, no punctuation, single words if possible."
    )
    user_input = f"type={it['type']}; color={it.get('color') or ''}; zone={it['zone']}; reason={it.get('reason') or ''}"

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{sys_prompt}\nINPUT: {user_input}\nJSON:",
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=25
        )
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        # try to extract JSON
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            txt = txt[start:end+1]
        data = json.loads(txt) if txt else {}
        desc = (data.get("description") or "").strip()
        kws = data.get("keywords") or []
        keywords_str = " ".join([str(k).strip().lower().replace(",", "") for k in kws if str(k).strip()])
        if not desc:
            desc = f"{it['type'].title()} possibly left in {it['zone']}"
        ok = db.set_description(shortid, desc, keywords_str)
        if not ok:
            raise HTTPException(status_code=404, detail="Item not found while updating")
        return {"ok": True, "shortid": shortid, "description": desc, "keywords": keywords_str.split() if keywords_str else []}
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Describe failed: {e}")

# -------- Search page/API ----------
@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: Optional[str] = None):
    results = db.search_items(q) if q else []
    return templates.TemplateResponse("search.html", {"request": request, "q": q or "", "results": results, "title": "Search"})

@app.get("/api/search")
def api_search(q: str):
    return {"results": db.search_items(q)}
