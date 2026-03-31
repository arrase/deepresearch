---
name: deepresearch
description: Use this skill when you need to perform an extensive web search, verify information across multiple sources, or generate a detailed report with citations. The application crawls the web, extracts evidence, and sends a synthesized report to Discord.
---

# deepresearch

Use this skill for any request that requires in-depth information gathering, market research, technical analysis, or factual verification using multiple online sources.

## Instructions

1. Formulate a clear, descriptive research query based on the user's request.
2. Execute the `deepresearch` command with the query as the first argument.
3. Always include the `--discord` flag to ensure the final report and findings are delivered to the user via Discord.
4. Set `background: true` as the process involves multi-stage web browsing and synthesis which may take several minutes.

## Example Command

`deepresearch "Comprehensive analysis of the current state of solid-state battery technology and its main competitors" --discord`
