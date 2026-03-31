"""Discord notification subagent."""

from __future__ import annotations

import datetime
import json
import logging
import re

import httpx

from ..config import DiscordConfig
from ..state import FinalReport
from ..output_utils import generate_pdf

logger = logging.getLogger(__name__)


async def send_discord_report(config: DiscordConfig, report: FinalReport) -> bool:
    """Sends the research report to a Discord user via DM."""
    if not config.token or not config.user_id:
        logger.error("Discord token or user_id not configured")
        return False

    headers = {
        "Authorization": f"Bot {config.token}",
        "User-Agent": "DeepResearchBot (https://github.com/google/deepresearch, 0.1.0)",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Create DM channel
        try:
            resp = await client.post(
                "https://discord.com/api/v10/users/@me/channels",
                headers=headers,
                json={"recipient_id": config.user_id},
            )
            resp.raise_for_status()
            channel_id = resp.json()["id"]
        except Exception as e:
            logger.error(f"Failed to create Discord DM channel: {e}")
            return False

        # 2. Prepare the report content
        clean_query = re.sub(r"[^a-zA-Z0-9]", "_", report.query)[:50]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if config.output == "pdf":
            filename = f"report_{clean_query}_{timestamp}.pdf"
            file_content = generate_pdf(report.markdown_report)
            content_type = "application/pdf"
        else:
            filename = f"report_{clean_query}_{timestamp}.md"
            file_content = report.markdown_report.encode("utf-8")
            content_type = "text/markdown"

        intro_parts = [
            f"🚀 **Research Report Generated**",
            f"**Query:** {report.query}",
            f"**Confidence:** {report.confidence.value.upper()}",
            f"**Sources Cited:** {len(report.cited_sources)}",
        ]

        if report.key_findings:
            intro_parts.append("\n**Key Findings:**")
            for finding in report.key_findings[:3]:
                intro_parts.append(f"• {finding}")
            if len(report.key_findings) > 3:
                intro_parts.append(f"• ... and {len(report.key_findings) - 3} more")

        intro_content = "\n".join(intro_parts)

        # Send as file
        try:
            # Multipart/form-data for files
            files = {"file": (filename, file_content, content_type)}
            payload_json = json.dumps({"content": intro_content})
            resp = await client.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                headers=headers,
                data={"payload_json": payload_json},
                files=files,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord file: {e}")
            return False
