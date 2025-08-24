import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_PATH = Path("findr.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shortid TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    color TEXT,
    zone TEXT NOT NULL,
    state TEXT NOT NULL, -- WITH_OWNER, AMBER, RED, RECOVERED
    reason TEXT,
    ts TEXT NOT NULL,    -- ISO string
    crop_path TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_shortid TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT,
    ts TEXT NOT NULL
);
"""

SEED_ITEMS = [
    dict(shortid="ph1", type="PHONE", color="black",
         zone="Floor 1 · Zone A · Table 3", state="RED",
         reason="dist:1.6m; dwell:7.2s; leaving:yes"),
    dict(shortid="bt1", type="BOTTLE", color="blue",
         zone="Floor 1 · Zone B · Table 7", state="AMBER",
         reason="dist:0.9m; dwell:2.5s"),
    dict(shortid="bp1", type="BACKPACK", color="red",
         zone="Floor 2 · Reading · Table 1", state="RECOVERED",
         reason="claimed by owner"),
]

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(seed: bool = True) -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        if seed:
            cur = conn.execute("SELECT COUNT(*) as c FROM items")
            count = cur.fetchone()["c"]
            if count == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for it in SEED_ITEMS:
                    conn.execute("""
                        INSERT INTO items (shortid, type, color, zone, state, reason, ts, crop_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (it["shortid"], it["type"], it.get("color"), it["zone"], it["state"], it.get("reason"), now, None))
                    conn.execute("""
                        INSERT INTO events (item_shortid, state, reason, ts)
                        VALUES (?, ?, ?, ?)
                    """, (it["shortid"], it["state"], it.get("reason"), now))
                conn.commit()
    finally:
        conn.close()

def list_items(states: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        if states:
            qmarks = ",".join("?" for _ in states)
            cur = conn.execute(f"SELECT * FROM items WHERE state IN ({qmarks}) ORDER BY id DESC", states)
        else:
            cur = conn.execute("SELECT * FROM items ORDER BY id DESC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def get_item(shortid: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.execute("SELECT * FROM items WHERE shortid = ?", (shortid,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def upsert_item(item: Dict[str, Any]) -> None:
    """Insert or update an item by shortid."""
    conn = connect()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = conn.execute("SELECT id FROM items WHERE shortid = ?", (item["shortid"],)).fetchone()
        if existing:
            conn.execute("""
                UPDATE items SET type=?, color=?, zone=?, state=?, reason=?, ts=?, crop_path=?
                WHERE shortid=?
            """, (item["type"], item.get("color"), item["zone"], item["state"], item.get("reason"), now, item.get("crop_path"), item["shortid"]))
        else:
            conn.execute("""
                INSERT INTO items (shortid, type, color, zone, state, reason, ts, crop_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (item["shortid"], item["type"], item.get("color"), item["zone"], item["state"], item.get("reason"), now, item.get("crop_path")))
        conn.execute("""
            INSERT INTO events (item_shortid, state, reason, ts)
            VALUES (?, ?, ?, ?)
        """, (item["shortid"], item["state"], item.get("reason"), now))
        conn.commit()
    finally:
        conn.close()

def resolve_item(shortid: str, reason: str = "resolved by staff") -> bool:
    conn = connect()
    try:
        cur = conn.execute("SELECT id FROM items WHERE shortid = ?", (shortid,))
        if not cur.fetchone():
            return False
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE items SET state='RECOVERED', reason=?, ts=? WHERE shortid=?", (reason, now, shortid))
        conn.execute("INSERT INTO events (item_shortid, state, reason, ts) VALUES (?,?,?,?)", (shortid, "RECOVERED", reason, now))
        conn.commit()
        return True
    finally:
        conn.close()
