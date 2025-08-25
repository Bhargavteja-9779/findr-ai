# Findr.AI — Privacy-First CCTV Lost & Found

### Real-time AI system that detects unattended items in shared spaces (libraries, labs, campuses) and instantly publishes them on a Lost & Found Billboard with QR codes — while keeping privacy at the core.

---

## 🚀 Problem
Lost items (keys, wallets, phones, bottles) are common. Recovery is manual, slow, and unreliable. CCTV exists but is passive — it records, but doesn’t infer *ownership*.

---

## 💡 Solution
Findr.AI turns existing CCTV into a **smart lost & found system**:

- Detects **items + humans** → calculates an **Ownership Score**.
- State machine: `WITH_OWNER → AMBER → RED`.
- On RED: save crop → publish on Billboard (QR + image).
- Staff Dashboard: preview feeds, resolve items, edit zones.
- Privacy-first: **no faces stored**, only blurred previews + cropped items.
- Auto-descriptions with **Ollama** (keywords + search support).
- Analytics: **time-to-flag, pickup rate, false alerts/hour**.

---

## ⚙️ Tech Stack
- **Backend:** FastAPI + SQLite (FTS for search)
- **Vision Worker:** YOLOv8n, custom ownership scoring, OpenCV
- **Privacy:** MediaPipe face blur (fallback bbox head blur)
- **GenAI:** Ollama (Llama 3.2:3B) for descriptions + keywords
- **Frontend:** TailwindCSS, Chart.js, Jinja2 templates
- **Deployment:** Docker-ready, runs on edge devices

---

## 📊 Key Metrics
- Time-to-flag: **6–8 seconds**
- False alerts: **<2 per hour per camera**
- Pickup within 24h: **>90%**
- UI latency: **<1.5s**

---

## 🖥️ Demo Flow
1. Start Vision Worker (`vision_worker/main.py`) on a sample video.
2. Dashboard shows live annotated feeds.
3. Item becomes RED → instantly appears on Billboard with QR.
4. QR → Item page (zone, reason, timestamp).
5. Staff clicks “Resolve” → item disappears in 1s.
6. Analytics shows system KPIs.

---

## 🔒 Privacy Commitment
- No faces stored, only blurred previews.
- Only item crops saved when unattended.
- Runs fully **on-premise**, no cloud dependency.

---

## 🛠️ How to Run

```bash
# clone repo
git clone https://github.com/<your-team>/findr.ai.git
cd findr.ai

# setup venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run server
uvicorn server.app:app --reload --port 8123

# run vision worker
python vision_worker/main.py --cam sample1 --src server/static/videos/sample1.mp4
Visit:

Billboard: http://127.0.0.1:8123/billboard

Dashboard: http://127.0.0.1:8123/dashboard

Search: http://127.0.0.1:8123/search

👥 Team
P N Bhargav Teja (Lead) → Full-stack + AI pipeline

Shruti → Vision worker (YOLO, ownership, OpenCV)

Pranathi → Backend + DB + API + analytics

Deepika → UI/UX, billboard, pitch design
