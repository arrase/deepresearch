import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from deepresearch.config import DiscordConfig
from deepresearch.subagents.discord import send_discord_report
from deepresearch.state import FinalReport, ConfidenceLevel

def create_mock_report(query: str, markdown: str) -> FinalReport:
    return FinalReport(
        query=query,
        executive_answer="Summary",
        key_findings=["Finding 1", "Finding 2"],
        confidence=ConfidenceLevel.HIGH,
        markdown_report=markdown,
        cited_sources=[]
    )

@pytest.mark.asyncio
async def test_send_discord_report_short_message() -> None:
    config = DiscordConfig(token="test_token", user_id="test_user")
    report = create_mock_report("Test query", "Short report")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mock create channel response
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"id": "channel_123"}
        
        # Mock send message response
        resp2 = MagicMock()
        resp2.status_code = 200

        mock_post.side_effect = [resp1, resp2]

        success = await send_discord_report(config, report)
        
        assert success is True
        assert mock_post.call_count == 2
        
        # Check second call (send message)
        args, kwargs = mock_post.call_args_list[1]
        assert "channel_123" in args[0]
        assert "json" in kwargs
        assert "Short report" in kwargs["json"]["content"]

@pytest.mark.asyncio
async def test_send_discord_report_long_message_as_file() -> None:
    config = DiscordConfig(token="test_token", user_id="test_user")
    report = create_mock_report("Test query", "Long report" * 200)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mock create channel
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"id": "channel_123"}
        
        resp2 = MagicMock()
        resp2.status_code = 200

        mock_post.side_effect = [resp1, resp2]

        success = await send_discord_report(config, report)
        
        assert success is True
        assert mock_post.call_count == 2
        # Check second call (send file)
        args, kwargs = mock_post.call_args_list[1]
        assert "channel_123" in args[0]
        assert "files" in kwargs
        assert "data" in kwargs
        assert "payload_json" in kwargs["data"]

@pytest.mark.asyncio
async def test_send_discord_report_failure_to_create_channel() -> None:
    config = DiscordConfig(token="test_token", user_id="test_user")
    report = create_mock_report("Query", "Report")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = Exception("Forbidden")
        mock_post.return_value = resp

        success = await send_discord_report(config, report)
        
        assert success is False
        assert mock_post.call_count == 1
