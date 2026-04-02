"""Deterministic internal workers — re-export façade.

All public helpers are available directly via ``from deepresearch.core.utils import …``
for backwards compatibility.  The implementations live in the sub-modules:

- ``url``      – URL canonicalization and domain extraction
- ``text``     – chunking, token estimation, excerpts
- ``ranking``  – candidate scoring, deduplication, classification
- ``evidence`` – evidence dedup, selection, dossier updates
- ``coverage`` – summarization and rendering helpers
"""

from .coverage import (
    render_markdown_report,
    summarize_evidence,
    summarize_gaps,
    summarize_search_candidates,
    summarize_source_visit,
    summarize_subqueries,
)
from .evidence import (
    build_report_sources,
    compute_minimum_coverage,
    deduplicate_evidence,
    select_evidence_for_context,
    stable_evidence_key,
    total_evidence_tokens,
    update_working_dossier,
)
from .ranking import (
    classify_browser_payload,
    deduplicate_candidates,
    enrich_gaps_with_search_terms,
    prune_queue_by_domain,
    rank_subqueries_for_source,
    reformulate_queries,
    score_candidate,
)
from .text import (
    estimate_tokens,
    select_relevant_chunks,
    short_excerpt,
    split_text,
)
from .url import (
    TRACKING_QUERY_KEYS,
    canonicalize_url,
    extract_domain,
)

__all__ = [
    # url
    "TRACKING_QUERY_KEYS",
    "canonicalize_url",
    "extract_domain",
    # text
    "estimate_tokens",
    "select_relevant_chunks",
    "short_excerpt",
    "split_text",
    # ranking
    "classify_browser_payload",
    "deduplicate_candidates",
    "enrich_gaps_with_search_terms",
    "prune_queue_by_domain",
    "rank_subqueries_for_source",
    "reformulate_queries",
    "score_candidate",
    # evidence
    "build_report_sources",
    "compute_minimum_coverage",
    "deduplicate_evidence",
    "select_evidence_for_context",
    "stable_evidence_key",
    "total_evidence_tokens",
    "update_working_dossier",
    # coverage
    "render_markdown_report",
    "summarize_evidence",
    "summarize_gaps",
    "summarize_search_candidates",
    "summarize_source_visit",
    "summarize_subqueries",
]
