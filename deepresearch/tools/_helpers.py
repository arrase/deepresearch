"""Search helper functions shared across search backends."""

from __future__ import annotations

import re
import unicodedata


def is_duckduckgo_anomaly_page(content: str) -> bool:
    lowered = content.lower()
    return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered


def normalize_search_query(query: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9\s.-]", " ", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    if not ascii_text:
        return ""
    terms = ascii_text.split()
    return " ".join(terms[:12])
