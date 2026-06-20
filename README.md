# Company Intelligence Report Generator

An AI-powered system that generates comprehensive business intelligence reports for companies by researching information, analyzing business challenges, and identifying AI opportunities.

## Overview

This project uses a multi-agent architecture with LLMs to:
1. **Research** - Autonomously search and gather company information from the web
2. **Analyze** - Generate business insights and identify AI implementation opportunities
3. **Report** - Produce polished markdown reports with actionable recommendations

## Features

- **Autonomous Web Research**: Uses DuckDuckGo to search for company information
- **Vector Database**: ChromaDB stores company knowledge with semantic embeddings
- **Multi-Agent System**: Specialized agents for research, analysis, and orchestration
- **AI Opportunity Identification**: Analyzes business challenges and suggests AI solutions
- **Executive Summaries**: CEO pitch and strategic recommendations

## Tech Stack

- **LLM Framework**: [smolagents](https://github.com/agentic-ai/smolagents)
- **Language Model**: Google Gemini 3.5 Flash (via LiteLLM)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Web Search**: DuckDuckGo Search API
- **Text Processing**: LangChain text splitters

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Setup

1. Clone/navigate to the project directory:
```bash
cd UnadaLabs_assesment
```

2. Install dependencies:
```bash
pip install smolagents langchain-huggingface sentence-transformers chromadb duckduckgo-search langchain-chroma langchain-text-splitters python-dotenv litellm
```

3. Set up environment variables:
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

Run the program:
```bash
python app.py
```

You'll be prompted to enter a company name:
```
Enter company name: Apple
```

The system will:
1. **Research Phase**: Search the web for company information
2. **Analysis Phase**: Generate insights and AI opportunities
3. **Report Generation**: Output a markdown report containing:
   - Company Overview
   - Key Business Information
   - Business Challenges (operational, sales, customer experience)
   - AI Opportunities (automation, analytics, engagement, efficiency)
   - CEO Pitch

## Project Structure

- **app.py** - Main application with agent definitions and orchestration
- **tools.py** - Tool definitions for company research and knowledge management
- **chroma_db/** - Persistent vector database storage

## How It Works

### Agent Architecture

**Research Agent**
- Creates/loads company knowledge base
- Searches for company information
- Stores findings in ChromaDB

**Analysis Agent**
- Retrieves stored company context
- Generates comprehensive business reports
- Identifies AI opportunities

**Manager Agent**
- Orchestrates workflow between research and analysis agents
- Ensures quality and completeness of final report

### Data Flow

```
User Input (Company Name)
         ↓
  Manager Agent
  ├─→ Research Agent
  │   ├─ Search Web
  │   ├─ Parse Results
  │   └─ Store in ChromaDB
  └─→ Analysis Agent
      ├─ Retrieve Context
      ├─ Generate Report
      └─ Format Output
         ↓
   Markdown Report
```

## Key Components

### Tools

- **get_or_create_company()** - Initialize company knowledge base
- **research_company()** - Web search for company information
- **append_company_research()** - Store research data with embeddings
- **retrieve_company_context()** - Semantic search of company data

### Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim embeddings)
- **Chunk Size**: 1000 tokens with 200-token overlap
- **Search Results**: 15 max results per query
- **Report Length**: Under 500 words

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM inference |

## Output Example

```
# Company Overview
[Company details and general information]

# Key Business Information
[Products, services, industry position]

# Business Challenges
- Operational challenges
- Sales challenges
- Customer experience challenges

# AI Opportunities
- Automation opportunities
- Analytics solutions
- Customer engagement tools
- Operational efficiency improvements

# CEO Pitch
[Executive summary and strategic recommendations]
```

## License

MIT

## Notes

- Reports are generated in real-time based on latest web information
- ChromaDB persists company knowledge for faster subsequent queries
- Customize report depth by modifying agent descriptions in `app.py`
