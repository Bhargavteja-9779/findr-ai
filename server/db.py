import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
    crop_path TEXT,
    description TEXT,
    keywords TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_shortid TEXT NOT NULL,
    state TEXT NOT NULL,
    reason TEXT,
    ts TEXT NOT NULL
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    shortid, type, color, zone, description, keywords, content='items', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, shortid, type, color, zone, description, keywords)
  VALUES (new.id, new.shortid, new.type, new.color, new.zone, coalesce(new.description,''), coalesce(new.keywords,''));
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, shortid, type, color, zone, description, keywords)
  VALUES('delete', old.id, old.shortid, old.type, old.color, old.zone, coalesce(old.description,''), coalesce(old.keywords,''));
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, shortid, type, color, zone, description, keywords)
  VALUES('delete', old.id, old.shortid, old.type, old.color, old.zone, coalesce(old.description,''), coalesce(old.keywords,''));
  INSERT INTO items_fts(rowid, shortid, type, color, zone, description, keywords)
  VALUES (new.id, new.shortid, new.type, new.color, new.zone, coalesce(new.description,''), coalesce(new.keywords,''));
END;
"""

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(seed: bool = False) -> None:
    """Create tables if missing (idempotent)."""
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.executescript(FTS_SCHEMA)
        conn.commit()
    finally:
        conn.close()

def hard_reset() -> Tuple[int, int]:
    """
    Drop and recreate ALL tables (items, events, items_fts).
    Returns (items_before, events_before) for visibility.
    """
    conn = connect()
    try:
        # counts before
        try:
            c_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        except sqlite3.Error:
            c_items = 0
        try:
            c_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except sqlite3.Error:
            c_events = 0

        conn.executescript("""
        DROP TRIGGER IF EXISTS items_ai;
        DROP TRIGGER IF EXISTS items_ad;
        DROP TRIGGER IF EXISTS items_au;
        DROP TABLE IF EXISTS items_fts;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS items;
        """)

        # recreate
        conn.executescript(SCHEMA)
        conn.executescript(FTS_SCHEMA)
        conn.commit()
        return (c_items, c_events)
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
    conn = connect()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = conn.execute("SELECT id FROM items WHERE shortid = ?", (item["shortid"],)).fetchone()
        if existing:
            conn.execute("""
                UPDATE items SET type=?, color=?, zone=?, state=?, reason=?, ts=?, crop_path=?,
                                description=coalesce(?,description), keywords=coalesce(?,keywords)
                WHERE shortid=?
            """, (item["type"], item.get("color"), item["zone"], item["state"], item.get("reason"), now,
                  item.get("crop_path"), item.get("description"), item.get("keywords"), item["shortid"]))
        else:
            conn.execute("""
                INSERT INTO items (shortid, type, color, zone, state, reason, ts, crop_path, description, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (item["shortid"], item["type"], item.get("color"), item["zone"], item["state"], item.get("reason"),
                  now, item.get("crop_path"), item.get("description"), item.get("keywords")))
        conn.execute("""
            INSERT INTO events (item_shortid, state, reason, ts)
            VALUES (?, ?, ?, ?)
        """, (item["shortid"], item["state"], item.get("reason"), now))
        conn.commit()
    finally:
        conn.close()

def set_description(shortid: str, description: str, keywords: str) -> bool:
    conn = connect()
    try:
        cur = conn.execute("UPDATE items SET description=?, keywords=? WHERE shortid=?", (description, keywords, shortid))
        conn.commit()
        return cur.rowcount > 0
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

def search_items(query: str) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        try:
            cur = conn.execute("""
                SELECT items.* FROM items_fts
                JOIN items ON items.id = items_fts.rowid
                WHERE items_fts MATCH ? ORDER BY items.id DESC
            """, (query,))
            rows = cur.fetchall()
        except sqlite3.Error:
            like = f"%{query}%"
            cur = conn.execute("""
                SELECT * FROM items
                WHERE shortid LIKE ? OR type LIKE ? OR color LIKE ? OR zone LIKE ?
                      OR ifnull(description,'') LIKE ? OR ifnull(keywords,'') LIKE ?
                ORDER BY id DESC
            """, (like, like, like, like, like, like))
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
