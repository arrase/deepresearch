"""Utility functions for report output generation (PDF, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import markdown
from weasyprint import CSS, HTML


def generate_pdf(markdown_text: str, output_path: Path | None = None) -> bytes:
    """
    Converts markdown text to a PDF.
    If output_path is provided, writes to disk.
    Always returns the PDF bytes.
    """

    css = """
    @page {
        margin: 2.5cm;
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 9pt;
            color: #777;
        }
    }
    body {
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    h1 {
        color: #2c3e50;
        font-size: 28pt;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2 {
        color: #e67e22;
        font-size: 20pt;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    h3 {
        color: #2980b9;
        font-size: 16pt;
        margin-top: 20px;
    }
    a {
        color: #3498db;
        text-decoration: none;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 10pt;
    }
    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e1e4e8;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    blockquote {
        border-left: 5px solid #bdc3c7;
        margin: 20px 0;
        padding: 10px 20px;
        background-color: #f9f9f9;
        color: #7f8c8d;
        font-style: italic;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #dcdde1;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #fbfbfb;
    }
    """

    html_text = markdown.markdown(markdown_text, extensions=['extra', 'codehilite', 'toc'])
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Research Report</title>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """

    pdf_bytes = HTML(string=full_html).write_pdf(output_path, stylesheets=[CSS(string=css)])
    return cast(bytes, pdf_bytes)
