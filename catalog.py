"""
catalog.py — loads SHL Individual Test Solutions catalog and provides
lightweight keyword-based retrieval to fit relevant items into context.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Load catalog once at import time
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

def _load() -> List[Dict[str, Any]]:
    with open(_CATALOG_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data

CATALOG: List[Dict[str, Any]] = _load()

# Name → item index for O(1) lookup during validation
_NAME_INDEX: Dict[str, Dict[str, Any]] = {item["name"].lower(): item for item in CATALOG}
_URL_SET: set = {item["url"] for item in CATALOG}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_full_catalog_text() -> str:
    """
    Returns the entire catalog serialised as a compact string suitable for
    inclusion in an LLM system prompt.
    """
    lines = ["=== SHL INDIVIDUAL TEST SOLUTIONS CATALOG ===\n"]
    for item in CATALOG:
        jl = ", ".join(item.get("job_levels", []))
        lang = ", ".join(item.get("languages", [])[:4])  # cap languages for brevity
        lines.append(
            f"NAME: {item['name']}\n"
            f"URL: {item['url']}\n"
            f"TEST_TYPE: {item['test_type']}\n"
            f"DURATION: ~{item.get('duration_minutes', '?')} min\n"
            f"JOB_LEVELS: {jl}\n"
            f"LANGUAGES: {lang}\n"
            f"DESCRIPTION: {item['description']}\n"
            f"REMOTE: {item.get('remote_testing', True)}\n"
            "---"
        )
    return "\n".join(lines)


def validate_recommendations(recommendations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filters out any recommendation whose URL is not in the real catalog.
    Also corrects the test_type if the agent hallucinated a wrong one.
    Returns only catalog-verified items.
    """
    clean: List[Dict[str, str]] = []
    for rec in recommendations:
        url = rec.get("url", "")
        name = rec.get("name", "")
        # Accept if URL matches
        if url in _URL_SET:
            clean.append(rec)
            continue
        # Try to match by name (case-insensitive)
        matched = _NAME_INDEX.get(name.lower())
        if matched:
            clean.append({
                "name": matched["name"],
                "url": matched["url"],
                "test_type": matched["test_type"],
            })
    # Deduplicate by URL, preserving order
    seen: set = set()
    deduped: List[Dict[str, str]] = []
    for rec in clean:
        key = rec.get("url") or rec.get("name")
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    return deduped[:10]  # cap at 10
