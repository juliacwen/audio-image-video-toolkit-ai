#!/usr/bin/env python3
# Multimodal/db/image_graph_db.py
"""
image_graph_db.py
Author: Julia Wen (wendigilane@gmail.com)
Date: 11-06-2025
------------------------------------------------------------------------------
SQLite backend for the Bayesian A* image-graph demo.

Purpose
-------
Persist image metadata, graph edges, and Monte Carlo path result summaries.

Design notes
------------
- Database file is stored in the same folder as this module:
    Multimodal/db/image_graph.db
- All configuration constants are defined in the constants block (no magic numbers).
- Indexes are created for common lookup columns for performance.
- Safe, simple SQL compatible with SQLite and easy to port to other RDBMS.

Tables
------
- images(id, name, path, hash, crescent_prob, timestamp)
- edges(id, src_name, dst_name, mean_weight, std_weight)
- paths(id, run_timestamp, source, target, path_json, probability, sample_count)
Python: 3.9+
------------------------------------------------------------------------------
"""
import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

DB_FILENAME = Path(__file__).parent / "image_graph.db"

# ------------------ Schema Management / Init ------------------

def _connect(db_path: Path = DB_FILENAME):
    os.makedirs(db_path.parent, exist_ok=True)
    return sqlite3.connect(db_path)

def _table_columns(conn: sqlite3.Connection, table_name: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return [r[1] for r in rows]  # column names

def init_db(db_path: Path = DB_FILENAME) -> None:
    """
    Initialize DB and add missing columns/tables if necessary.
    Safe to call repeatedly; preserves existing data.
    """
    conn = _connect(db_path)
    c = conn.cursor()

    # Base table required columns (we'll add others via ALTER if missing)
    c.execute("""
    CREATE TABLE IF NOT EXISTS graph_runs (
        run_name TEXT PRIMARY KEY,
        source_image TEXT,
        target_image TEXT,
        timestamp TEXT
    );
    """)

    # Ensure top_paths table exists (legacy and useful)
    c.execute("""
    CREATE TABLE IF NOT EXISTS top_paths (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT NOT NULL,
        path_index INTEGER NOT NULL,
        path_json TEXT NOT NULL,
        probability REAL NOT NULL,
        FOREIGN KEY(run_name) REFERENCES graph_runs(run_name)
    );
    """)

    conn.commit()

    # Now check columns and add if missing
    existing = set(_table_columns(conn, "graph_runs"))

    # Desired extra columns
    desired_cols = {
        "graph_most_probable": "BLOB",
        "graph_top10": "BLOB",
        "top_paths_json": "TEXT",
        "image_map_json": "TEXT",
        "edges_json": "TEXT",
        "notes": "TEXT"
    }

    for col, coltype in desired_cols.items():
        if col not in existing:
            try:
                c.execute(f"ALTER TABLE graph_runs ADD COLUMN {col} {coltype}")
                conn.commit()
                print(f"[DB] Added missing column '{col}' to graph_runs.")
            except sqlite3.OperationalError as e:
                # If ALTER fails for some reason, log and continue (do not crash)
                print(f"[DB] Warning: could not add column {col}: {e}")

    conn.close()

# Backwards-compatible alias name
def init_image_graph_db(db_path: Path = DB_FILENAME) -> None:
    init_db(db_path)

# ------------------ Primary Save / Retrieve API ------------------

def save_run(run_name: str,
             source_image: str,
             target_image: str,
             graph_most_probable_bytes: bytes,
             graph_top10_bytes: bytes,
             top_paths: List[Tuple[List[str], float]],
             sample_count: int = 0,
             db_path: Path = DB_FILENAME) -> None:
    """
    Save a full run into the DB. Creates or replaces the graph_runs row and
    writes structured top_paths rows as well.

    FIXED: added sample_count argument to match Streamlit app call
    """
    init_db(db_path)
    conn = _connect(db_path)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()

    # store structured top_paths as JSON in graph_runs for quick export / compatibility
    top_paths_json = json.dumps(top_paths)

    # Insert or replace run row
    c.execute("""
        INSERT OR REPLACE INTO graph_runs
        (run_name, source_image, target_image, timestamp, graph_most_probable, graph_top10, top_paths_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (run_name, source_image, target_image, timestamp,
          graph_most_probable_bytes, graph_top10_bytes, top_paths_json))

    # Remove previous top_paths rows for this run and insert new ones (structured table)
    c.execute("DELETE FROM top_paths WHERE run_name=?", (run_name,))
    for idx, (path_seq, prob) in enumerate(top_paths):
        path_json = json.dumps(path_seq)
        c.execute("""
            INSERT INTO top_paths (run_name, path_index, path_json, probability)
            VALUES (?, ?, ?, ?)
        """, (run_name, idx, path_json, float(prob)))

    conn.commit()
    conn.close()
    print(f"[DB] save_run: saved run '{run_name}' (source={source_image}, target={target_image})")

def list_runs(db_path: Path = DB_FILENAME):
    """Return list of (run_name, source_image, target_image, timestamp) ordered newest first."""
    init_db(db_path)
    conn = _connect(db_path)
    c = conn.cursor()
    c.execute("SELECT run_name, source_image, target_image, timestamp FROM graph_runs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_run(run_name: str, db_path: Path = DB_FILENAME):
    """
    Retrieve run data. Returns dict or None.
    """
    init_db(db_path)
    conn = _connect(db_path)
    c = conn.cursor()
    c.execute("SELECT source_image, target_image, timestamp, graph_most_probable, graph_top10, top_paths_json FROM graph_runs WHERE run_name=?", (run_name,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return None
    source_image, target_image, timestamp, g1, g10, tp_json = row
    # load structured top_paths table as well (if present)
    c.execute("SELECT path_json, probability FROM top_paths WHERE run_name=? ORDER BY path_index", (run_name,))
    paths_table = [(json.loads(r[0]), float(r[1])) for r in c.fetchall()]
    conn.close()

    # If top_paths_json exists but structured table empty, prefer JSON
    if (not paths_table) and tp_json:
        try:
            paths_table = [(list(p[0]), float(p[1])) for p in json.loads(tp_json)]
        except Exception:
            paths_table = []

    return {
        "run_name": run_name,
        "source_image": source_image,
        "target_image": target_image,
        "timestamp": timestamp,
        "graph_most_probable_bytes": g1,
        "graph_top10_bytes": g10,
        "top_paths": paths_table
    }

def export_run_files(run_name: str, output_dir: str = None, db_path: Path = DB_FILENAME):
    """
    Export a run's PNGs and CSV from DB to disk.
    Returns dict { 'png_most_probable': path, 'png_top10': path, 'csv_top_paths': path }
    """
    data = get_run(run_name, db_path)
    if data is None:
        raise ValueError(f"Run '{run_name}' not found in DB.")
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "retrieved_runs")
    os.makedirs(output_dir, exist_ok=True)

    src = os.path.splitext(os.path.basename(data["source_image"] or "source"))[0]
    tgt = os.path.splitext(os.path.basename(data["target_image"] or "target"))[0]
    png1_path = os.path.join(output_dir, f"graph_most_probable_{run_name}.png")
    png10_path = os.path.join(output_dir, f"graph_top10_paths_{run_name}.png")
    csv_path = os.path.join(output_dir, f"top_paths_{run_name}.csv")

    # Write PNG blobs (handle None)
    if data["graph_most_probable_bytes"]:
        with open(png1_path, "wb") as f:
            f.write(data["graph_most_probable_bytes"])
    else:
        png1_path = None

    if data["graph_top10_bytes"]:
        with open(png10_path, "wb") as f:
            f.write(data["graph_top10_bytes"])
    else:
        png10_path = None

    # Write CSV from structured top_paths
    with open(csv_path, "w", newline="") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(["Path (filenames)", "Probability"])
        for path_seq, prob in data["top_paths"]:
            writer.writerow([" -> ".join(path_seq), f"{prob:.4f}"])

    print(f"[DB] Exported run '{run_name}': PNGs -> {png1_path}, {png10_path}; CSV -> {csv_path}")
    return {"png_most_probable": png1_path, "png_top10": png10_path, "csv_top_paths": csv_path}

# Ensure DB exists on import
init_db()

# ------------------ Backwards-Compatible helper functions ------------------

def save_images(hash_to_path: dict, crescent_probs: dict = None, db_path: Path = DB_FILENAME):
    """
    Compatibility shim: stores a single 'images_metadata' run row containing
    the image map and crescent probs in the graph_runs.image_map_json column.
    """
    init_db(db_path)
    conn = _connect(db_path)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    run_name = "images_metadata"

    payload = {"hash_to_path": hash_to_path, "crescent_probs": crescent_probs or {}}
    payload_json = json.dumps(payload)

    c.execute("""
        INSERT OR REPLACE INTO graph_runs
        (run_name, source_image, target_image, timestamp, image_map_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_name, "", "", timestamp, payload_json, "images metadata (compat shim)"))

    conn.commit()
    conn.close()
    print("[DB] save_images: stored image metadata under run 'images_metadata' (compat)")

def save_edges(G, clear_existing: bool = True, db_path: Path = DB_FILENAME):
    """
    Compatibility shim: serializes graph edges into edges_json column under run 'edges_metadata'.
    """
    init_db(db_path)
    conn = _connect(db_path)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    run_name = "edges_metadata"

    edges_serial = []
    for u, v, data in G.edges(data=True):
        edges_serial.append({"u": u, "v": v, "data": data})

    payload_json = json.dumps(edges_serial)

    c.execute("""
        INSERT OR REPLACE INTO graph_runs
        (run_name, source_image, target_image, timestamp, edges_json, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_name, "", "", timestamp, payload_json, "edges metadata (compat shim)"))

    conn.commit()
    conn.close()
    print("[DB] save_edges: stored edges under run 'edges_metadata' (compat shim)")

def save_paths(source_hash: str, target_hash: str, normalized_top_paths: List[Tuple[List[str], float]], sample_count: int = 0, db_path: Path = DB_FILENAME):
    """
    Compatibility shim: save_paths previously expected to record paths.
    We create a run row with generated run_name and store top_paths.
    """
    init_db(db_path)
    run_name = f"compat_paths_{source_hash[:8]}_{target_hash[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        save_run(run_name=run_name,
                 source_image=source_hash,
                 target_image=target_hash,
                 graph_most_probable_bytes=None,
                 graph_top10_bytes=None,
                 top_paths=normalized_top_paths,
                 sample_count=sample_count,
                 db_path=db_path)
        print(f"[DB] save_paths: created compat run '{run_name}' with {len(normalized_top_paths)} paths.")
    except Exception as e:
        print(f"[DB] save_paths: error saving compat paths: {e}")

# Expose public API names expected by newer code
__all__ = [
    "init_db", "save_run", "list_runs", "get_run", "export_run_files",
    "init_image_graph_db", "save_images", "save_edges", "save_paths"
]
