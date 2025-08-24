import cv2
import time
import math
import uuid
import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from PIL import Image
import requests
from ultralytics import YOLO
from privacy import face_blur_bgr

# ---------------- Config ----------------
VIDEO_SOURCE = 0  # or RTSP/file path
API_BASE = "http://127.0.0.1:8123"

# Project paths for saving crops
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "server" / "static" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Load zones (normalized coords)
ZONES_CFG_PATH = PROJECT_ROOT / "vision_worker" / "zones.json"
try:
    ZONES_CFG = json.loads(ZONES_CFG_PATH.read_text()) if ZONES_CFG_PATH.exists() else []
except Exception as e:
    print("[Zones] Failed to load zones.json:", e)
    ZONES_CFG = []

def zone_name_for_box(box: Tuple[int,int,int,int], W: int, H: int) -> str:
    """Return the first zone name whose rect contains the box centroid."""
    if not ZONES_CFG:
        return "Zone"
    x1,y1,x2,y2 = box
    cx = (x1 + x2) / 2.0 / max(W, 1)
    cy = (y1 + y2) / 2.0 / max(H, 1)
    for z in ZONES_CFG:
        if z.get("shape") == "rect":
            rx1, ry1, rx2, ry2 = z.get("rect", [0,0,1,1])
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return z.get("name", "Zone")
    return "Zone"

# COCO class IDs we care about
CLS_PERSON = 0
TARGET_ITEM_IDS = {24, 39, 63, 67, 73}  # backpack, bottle, laptop, phone, book
CLASS_NAMES = {0: "person", 24: "BACKPACK", 39:"BOTTLE", 63:"LAPTOP", 67:"PHONE", 73:"BOOK"}

# Ownership Score params (light)
D_MAX_NORM = 0.35   # distance normalization upper bound (fraction of frame diagonal)
THETA1 = 0.45       # to enter AMBER
THETA2 = 0.70       # to enter RED
T_AMBER = 3.0       # seconds OS >= THETA1 to go AMBER
T_RED   = 7.0       # seconds OS >= THETA2 to go RED
LEAVING_FRAMES = 6  # frames of "moving away" after being close

# Drawing
BOX_COLOR = (0, 140, 255)

# -------------- Helpers -----------------
def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-6
    return inter / union

def centroid(box: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def l2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------- Tracking structs ----------
@dataclass
class ItemTrack:
    shortid: str
    cls_name: str
    color_hint: Optional[str] = None
    zone: str = "Zone"
    box: Tuple[int,int,int,int] = (0,0,0,0)
    last_seen_ts: float = 0.0
    # ownership features
    nearest_dist_norm: float = 1.0
    dwell_nearby_sec: float = 0.0
    last_nearby_sec: float = 999.0
    leaving_flag: bool = False
    # internal timers
    os_ge_t1_since: Optional[float] = None
    os_ge_t2_since: Optional[float] = None
    state: str = "WITH_OWNER"  # WITH_OWNER, AMBER, RED
    # motion for leaving
    _prev_nearest_person: Optional[Tuple[float,float]] = None
    _away_counter: int = 0
    # UI crop
    crop_relpath: Optional[str] = None

    def compute_os(self) -> float:
        f_dist = 1.0 - max(0.0, min(self.nearest_dist_norm / D_MAX_NORM, 1.0))  # closer person => higher score
        f_dwell = max(0.0, min(self.dwell_nearby_sec / 6.0, 1.0))
        f_last_touch = math.exp(-self.last_nearby_sec / 8.0)
        f_leaving = 1.0 if self.leaving_flag else 0.0
        raw = 0.45*f_dist + 0.25*f_dwell + 0.20*f_last_touch + 0.10*f_leaving
        return max(0.0, min(raw, 1.0))

# -------------- API calls -----------------
def api_upsert(it: ItemTrack, reason: str):
    payload = {
        "shortid": it.shortid,
        "type": it.cls_name,
        "color": it.color_hint,
        "zone": it.zone,
        "state": it.state,
        "reason": reason,
        "crop_path": it.crop_relpath
    }
    try:
        r = requests.post(f"{API_BASE}/api/items", json=payload, timeout=2)
        r.raise_for_status()
    except Exception as e:
        print("[API] upsert failed:", e)

# -------------- Main loop -----------------
def main():
    print("[Vision] Loading YOLO...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    tracks: Dict[int, ItemTrack] = {}
    next_id = 1
    last_ts = time.time()

    print("[Vision] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue
        ts = time.time()
        dt = ts - last_ts
        last_ts = ts

        H, W = frame.shape[:2]
        diag = math.hypot(W, H)

        # Run detection
        res = model(frame, verbose=False)[0]
        people: List[Tuple[int,int,int,int]] = []
        items: List[Tuple[int,int,int,int,int,float]] = []  # (x1,y1,x2,y2, cls_id, conf)

        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                if cls == CLS_PERSON:
                    people.append((x1,y1,x2,y2))
                elif cls in TARGET_ITEM_IDS:
                    items.append((x1,y1,x2,y2, cls, conf))

        # --- match detections to existing tracks via IOU ---
        used_track_ids = set()
        candidates = []
        for tid, tr in tracks.items():
            best_iou = 0.0
            best_j = -1
            for j, det in enumerate(items):
                x1,y1,x2,y2,cls,conf = det
                if CLASS_NAMES.get(cls, "ITEM") != tr.cls_name:
                    continue
                i = iou(tr.box, (x1,y1,x2,y2))
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0:
                candidates.append((best_iou, tid, best_j))
        candidates.sort(reverse=True, key=lambda t: t[0])
        assigned_det = set()
        for iou_score, tid, j in candidates:
            if iou_score < 0.1:  # too small overlap
                continue
            if tid in used_track_ids or j in assigned_det:
                continue
            used_track_ids.add(tid)
            assigned_det.add(j)
            x1,y1,x2,y2,cls,conf = items[j]
            tr = tracks[tid]
            tr.box = (x1,y1,x2,y2)
            tr.last_seen_ts = ts
            tr.zone = zone_name_for_box(tr.box, W, H)

        # create new tracks for unassigned detections
        for j, det in enumerate(items):
            if j in assigned_det:
                continue
            x1,y1,x2,y2,cls,conf = det
            shortid = f"{CLASS_NAMES.get(cls,'ITEM')[:2].lower()}{uuid.uuid4().hex[:6]}"
            tr = ItemTrack(
                shortid=shortid,
                cls_name=CLASS_NAMES.get(cls, "ITEM"),
                box=(x1,y1,x2,y2),
                last_seen_ts=ts,
                zone=zone_name_for_box((x1,y1,x2,y2), W, H)
            )
            tracks[next_id] = tr
            used_track_ids.add(next_id)
            next_id += 1

        # remove stale tracks not seen for 2s
        for tid in list(tracks.keys()):
            if ts - tracks[tid].last_seen_ts > 2.0:
                del tracks[tid]

        # compute ownership features per track
        person_centroids = [centroid(p) for p in people]
        for tid, tr in tracks.items():
            c_item = centroid(tr.box)
            if person_centroids:
                dists = [l2(c_item, pc) for pc in person_centroids]
                min_px = min(dists)
                tr.nearest_dist_norm = min_px / max(diag, 1.0)
                if tr.nearest_dist_norm < D_MAX_NORM:
                    tr.dwell_nearby_sec += dt
                    tr.last_nearby_sec = 0.0
                    nearest_idx = dists.index(min_px)
                    nearest_now = person_centroids[nearest_idx]
                    if tr._prev_nearest_person is not None:
                        prev = tr._prev_nearest_person
                        if l2(c_item, nearest_now) > l2(c_item, prev) + 2.0:
                            tr._away_counter += 1
                        else:
                            tr._away_counter = max(0, tr._away_counter - 1)
                    tr._prev_nearest_person = nearest_now
                    tr.leaving_flag = tr._away_counter >= LEAVING_FRAMES
                else:
                    tr.last_nearby_sec += dt
                    tr.dwell_nearby_sec = max(0.0, tr.dwell_nearby_sec - dt*0.5)
                    tr._away_counter = max(0, tr._away_counter - 1)
                    tr.leaving_flag = False
            else:
                tr.nearest_dist_norm = 1.0
                tr.last_nearby_sec += dt
                tr.dwell_nearby_sec = max(0.0, tr.dwell_nearby_sec - dt*0.5)
                tr._away_counter = max(0, tr._away_counter - 1)
                tr.leaving_flag = False

            # Ownership Score + state timers
            os = tr.compute_os()
            now = ts

            if os >= THETA1:
                tr.os_ge_t1_since = tr.os_ge_t1_since or now
            else:
                tr.os_ge_t1_since = None

            if os >= THETA2:
                tr.os_ge_t2_since = tr.os_ge_t2_since or now
            else:
                tr.os_ge_t2_since = None

            prev_state = tr.state
            if tr.state == "WITH_OWNER":
                if tr.os_ge_t1_since and (now - tr.os_ge_t1_since) >= T_AMBER:
                    tr.state = "AMBER"
            if tr.state == "AMBER":
                if tr.os_ge_t2_since and (now - tr.os_ge_t2_since) >= T_RED:
                    tr.state = "RED"

            reason = f"dist:{tr.nearest_dist_norm*diag:.0f}px; dwell:{tr.dwell_nearby_sec:.1f}s; leaving:{'yes' if tr.leaving_flag else 'no'}"

            # if state changed → push to API (and save crop on RED)
            if tr.state != prev_state:
                print(f"[STATE] {tr.shortid}: {prev_state} → {tr.state} | {reason}")
                if tr.state == "RED":
                    x1,y1,x2,y2 = tr.box
                    pad_x = int(0.10 * (x2 - x1)); pad_y = int(0.10 * (y2 - y1))
                    sx1 = max(0, x1 - pad_x); sy1 = max(0, y1 - pad_y)
                    sx2 = min(W, x2 + pad_x);  sy2 = min(H, y2 + pad_y)
                    crop = frame[sy1:sy2, sx1:sx2]
                    if crop.size:
                        fname = f"{tr.shortid}.jpg"
                        fpath = CROPS_DIR / fname
                        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(fpath, quality=90)
                        tr.crop_relpath = f"crops/{fname}"
                api_upsert(tr, reason)

        # draw
        for (x1,y1,x2,y2) in people:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (80,255,80), 2)
            cv2.putText(frame, "person", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        for tr in tracks.values():
            x1,y1,x2,y2 = tr.box
            cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, 2)
            cv2.putText(frame, f"{tr.cls_name} | {tr.state}", (x1, max(0,y1-22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"os~{tr.compute_os():.2f}  {tr.zone}", (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # privacy: blur faces in preview
        frame_display = face_blur_bgr(frame, people_boxes=people)
        cv2.imshow("Findr.AI — Ownership & States", frame_display)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
