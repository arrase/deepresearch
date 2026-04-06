"""Markdown helpers for topic briefs and final report parsing."""

from __future__ import annotations

import re

from ...state import ReportSection


def strip_markdown_fence(text: str) -> str:
    markdown = text.strip()
    if markdown.startswith("```"):
        markdown = re.sub(r"^```(?:markdown)?\n", "", markdown)
        markdown = re.sub(r"\n```$", "", markdown)
    return markdown.strip()


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def extract_markdown_section(markdown: str, heading: str) -> str:
    normalized_heading = _normalize_heading(heading)
    matches = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        title = _normalize_heading(match.group(2))
        if title != normalized_heading:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        return markdown[start:end].strip()
    return ""


def extract_markdown_bullets(markdown: str, heading: str) -> list[str]:
    section = extract_markdown_section(markdown, heading)
    bullets: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            bullets.append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s+", stripped):
            bullets.append(re.sub(r"^\d+\.\s+", "", stripped).strip())
    return bullets


def markdown_to_report_sections(markdown: str) -> list[ReportSection]:
    sections: list[ReportSection] = []
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if not body:
            continue
        first_paragraph = next((part.strip() for part in body.split("\n\n") if part.strip()), "")
        sections.append(ReportSection(title=title, summary=first_paragraph, body=body))
    return sections
