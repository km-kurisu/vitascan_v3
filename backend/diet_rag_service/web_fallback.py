"""
WebFallback - live IFCT2017 data retrieval used when curated-KB coverage is thin.

Fetches https://raw.githubusercontent.com/nodef/ifct2017/main/compositions/index.csv
once, caches it in .cache/ifct_index.csv with a 7-day TTL, and performs a lenient
parse for the foods matching a query. Any failure returns [] silently.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vitascan.diet_web")

IFCT_URL = "https://raw.githubusercontent.com/nodef/ifct2017/main/compositions/index.csv"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "ifct_index.csv"
TTL_SECONDS = 7 * 24 * 3600

IFCT_NUT = {
    "fe": ("fe", 1000.0, "mg"),
    "folsum": ("folsum", 1e6, "ug"),
    "vitc": ("vitc", 1000.0, "mg"),
    "retol": ("retol", 1e6, "ug"),
    "vitd": ("vitd", 1e6, "ug"),
    "thia": ("thia", 1000.0, "mg"),
    "ribf": ("ribf", 1000.0, "mg"),
    "nia": ("nia", 1000.0, "mg"),
    "vitb6c": ("vitb6c", 1000.0, "mg"),
    "ca": ("ca", 1000.0, "mg"),
    "mg": ("mg", 1000.0, "mg"),
    "zn": ("zn", 1000.0, "mg"),
    "p": ("p", 1000.0, "mg"),
    "k": ("k", 1000.0, "mg"),
    "na": ("na", 1000.0, "mg"),
    "protcnt": ("protcnt", 1.0, "g"),
    "fatce": ("fatce", 1.0, "g"),
    "fibtg": ("fibtg", 1.0, "g"),
}


def _fresh_cache() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < TTL_SECONDS


def _download() -> bool:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(IFCT_URL, timeout=15) as resp:
            data = resp.read()
        CACHE_FILE.write_bytes(data)
        return True
    except Exception as e:
        logger.warning(f"IFCT web fallback download failed: {e}")
        return False


def _rows() -> List[Dict[str, str]]:
    if not _fresh_cache() and not _download():
        if not CACHE_FILE.exists():
            return []
    try:
        with open(CACHE_FILE, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.warning(f"IFCT cache parse failed: {e}")
        return []


def search_foods(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Best-effort keyword search over IFCT raw rows."""
    rows = _rows()
    if not rows:
        return []
    q = (query or "").strip().lower()
    results = []
    for r in rows:
        name = (r.get("name") or "") + " " + (r.get("tags") or "")
        if not q or q in name.lower():
            results.append(r)
            if len(results) >= limit:
                break
    return results


def find_nutrients_by_code(code: str) -> Optional[Dict[str, Any]]:
    """Return converted nutrient dict for an exact IFCT code, or None."""
    rows = _rows()
    if not rows:
        return None
    for r in rows:
        if r.get("code") == code:
            nutrients = {}
            for nid, (col, mult, unit) in IFCT_NUT.items():
                raw = r.get(col)
                val = 0.0
                if raw:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = 0.0
                nutrients[nid] = round(val * mult, 4)
            return {"code": code, "name": r.get("name"), "nutrients_per_100g": nutrients}
    return None


def lookup_food(name_or_code: str) -> Optional[Dict[str, Any]]:
    """Cached in-memory lookup for the README/runtime path; returns None when unavailable."""
    if name_or_code in os.environ.get("IFCT_LOOKUP_DISABLED", ""):
        return None
    return find_nutrients_by_code(name_or_code)