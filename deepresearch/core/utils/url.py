"""URL canonicalization and domain extraction."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid",
}


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    filtered_query = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunparse(parsed._replace(
        scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/", query=urlencode(filtered_query), fragment="",
    ))


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()
