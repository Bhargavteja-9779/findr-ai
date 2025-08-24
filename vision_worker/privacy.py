import cv2

# Try to import mediapipe; many macOS/Python 3.13 setups won't have wheels yet.
try:
    import mediapipe as mp
    _MP_OK = True
    _FACE = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
except Exception:
    _MP_OK = False
    _FACE = None

_PRINTED_MODE = False

def _log_mode(msg: str):
    global _PRINTED_MODE
    if not _PRINTED_MODE:
        print("[Privacy]", msg)
        _PRINTED_MODE = True

def face_blur_bgr(frame_bgr, people_boxes=None):
    """
    Return a blurred-copy for privacy.
    Priority 1: MediaPipe face detection (if available).
    Priority 2: Approximate head blur using YOLO person boxes (upper 35% of each person bbox).
    If nothing available, return the original frame.
    """
    h, w = frame_bgr.shape[:2]

    # 1) MediaPipe path
    if _MP_OK and _FACE is not None:
        _log_mode("Face blur mode: MediaPipe FaceDetection")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = _FACE.process(rgb)
        if res and res.detections:
            out = frame_bgr.copy()
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w)); y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w)); y2 = min(h, int((bb.ymin + bb.height) * h))
                if x2 > x1 and y2 > y1:
                    roi = out[y1:y2, x1:x2]
                    if roi.size:
                        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (35,35), 0)
            return out
        # fall through to person-box head blur if no detections

    # 2) Person-box head blur (approximate)
    if people_boxes:
        _log_mode("Face blur mode: Approx head blur from person boxes")
        out = frame_bgr.copy()
        for (x1,y1,x2,y2) in people_boxes:
            # clamp
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1: 
                continue
            head_h = int((y2 - y1) * 0.35)  # top 35% as head region
            hy2 = y1 + max(1, head_h)
            roi = out[y1:hy2, x1:x2]
            if roi.size:
                out[y1:hy2, x1:x2] = cv2.GaussianBlur(roi, (35,35), 0)
        return out

    # 3) Nothing to blur with → return original
    _log_mode("Face blur mode: Disabled (no mediapipe & no person boxes)")
    return frame_bgr
