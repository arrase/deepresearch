# DeepResearch 🔍

**DeepResearch** is an autonomous, auditable research agent that performs deep-dives into complex, open-ended questions. Unlike simple RAG systems, it iteratively plans, searches, browses, and evaluates evidence until it finds a comprehensive answer—all powered by local LLMs via Ollama.

## ✨ Key Features

- **Autonomous Research:** Decomposes complex queries into subqueries and search intents.
- **Local-First:** Optimized for local models (like Qwen 2.5 or Llama 3) via Ollama. No data leaves your machine except for web searches.
- **Web-Scale Browsing:** Uses [Lightpanda](https://lightpanda.io/), a high-performance headless browser, to navigate and extract content from the real web.
- **Traceable & Auditable:** Every claim in the final report is backed by atomic evidence, specific URLs, and direct quotations.
- **Iterative Refinement:** Evaluates its own progress, identifies knowledge gaps or contradictions, and performs follow-up searches.
- **Rich Output:** Generates structured Markdown, professional PDF reports, or sends findings directly to Discord.
- **Agentic Skill:** Compatible with agents like OpenClaw and Gemini CLI as a specialized research skill.

## 🚀 Getting Started

### 📋 Prerequisites

1.  **Python 3.11+**
2.  **Docker (Mandatory):** Required to run the Lightpanda browser instance for web scraping.
3.  **Ollama:** Installed and running with your preferred model (e.g., `ollama pull qwen2.5:7b`).

### 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/deepresearch.git
   cd deepresearch
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Pull the browser image:**
   ```bash
   docker pull lightpanda/browser:nightly
   ```

## 🛠 Usage

You can run the research pipeline directly from your terminal:

```bash
# Basic research (outputs report.md)
deepresearch "What are the latest breakthroughs in solid-state battery technology in 2024?"

# Generate a PDF report
deepresearch "Compare Lightpanda vs Playwright for LLM-based web scraping" --pdf comparative_analysis.pdf

# Send the final report to Discord
deepresearch "The impact of Llama 3 on local AI" --discord
```

### ⌨️ CLI Arguments

| Argument | Description |
| :--- | :--- |
| `query` | The open-ended research question (required). |
| `--markdown PATH` | Save the final report as a Markdown file at the specified path. |
| `--pdf PATH` | Save the final report as a PDF file at the specified path. |
| `--discord` | Send the final report and executive summary to a configured Discord user via DM. |
| `--model NAME` | Override the default Ollama model (e.g., `qwen2.5:14b`). |
| `--num-ctx N` | Override the LLM context window size (default: 100,000+). |
| `--max-iterations N` | Limit the number of research cycles (default: 8). |
| `--config-root PATH` | Path to an editable configuration directory. |
| `-v, --verbose` | Enable real-time telemetry to watch the agent's internal process. |

## 🤖 Skill Integration (OpenClaw / Gemini CLI)

DeepResearch can be used as a "skill" by AI agents. When activated, the agent uses the `deepresearch` command to perform exhaustive background research before responding to the user.

**Example Agent Command:**
```bash
# The agent will run this in the background
deepresearch "Comprehensive analysis of solid-state battery tech and its competitors" --discord
```

## ⚙️ Configuration

On the first run, the application creates a configuration directory at `~/.deepresearch/config/`.

### Discord Setup
To use the `--discord` flag, update `config.toml` with your bot credentials:
```toml
[discord]
token = "YOUR_BOT_TOKEN"
user_id = "YOUR_DISCORD_USER_ID"
output = "pdf" # or "markdown"
```

### Search Backend
You can switch between search providers in `config.toml`:
- **Tavily (Recommended):** High-quality search for LLMs. Requires an `api_key`.
- **DuckDuckGo:** Free, lite backend with no API key required.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
