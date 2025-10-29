import sqlite3, pathlib, os

APP_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_FILE = APP_DIR / "db" / "panel.db"

def get_conn():
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema():
    conn = get_conn(); c = conn.cursor()
    # settings key-value
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    # schedules
    c.execute("""CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        days_mask TEXT NOT NULL, -- 7 chars, L..D => '1111100'
        start_time TEXT NOT NULL, -- '21:00'
        end_time TEXT NOT NULL,   -- '05:00'
        enabled INTEGER NOT NULL DEFAULT 1
    )""")
    conn.commit(); conn.close()

def get_setting(key: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    r = c.fetchone()
    conn.close()
    return r["value"] if r else None

def set_setting(key: str, value: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit(); conn.close()

def dict_rows(rows):
    return [dict(r) for r in rows]
