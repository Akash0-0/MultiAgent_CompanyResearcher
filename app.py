from smolagents import CodeAgent,DuckDuckGoSearchTool,GoogleSearchTool,Tool,FinalAnswerTool,tool,LiteLLMModel,InferenceClientModel
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
from smolagents import tool
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import uuid
from smolagents import OpenAIServerModel
import litellm
from tools import (
    get_or_create_company,
    research_company,
    append_company_research,
    retrieve_company_context
)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

from dotenv import load_dotenv
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

model = LiteLLMModel(
    model_id="gemini/gemini-3.5-flash",
    api_key=key,
    
)
# =========================
# RESEARCH AGENT
# =========================

ResearchAgent = CodeAgent(
    name="research_agent",
    description="""
Your ONLY responsibility is building the company knowledge base.

Workflow:
1. Create or load the company collection.
2. Search for company information.
3. Gather:
   - Company overview
   - Products and services
   - Industry
   - Recent developments
   - Business operations
   - Expansion plans
4. Store findings in ChromaDB.

Do NOT generate reports.
Do NOT perform business analysis.
Only collect and store information.
""",
    model=model,
    tools=[
        get_or_create_company,
        research_company,
        append_company_research
    ]
)

# =========================
# ANALYSIS AGENT
# =========================

AnalysisAgent = CodeAgent(
    name="analysis_agent",
    description="""
Use retrieve_company_context to generate a complete business report.

Generate:

1. Company Overview
2. Key Business Information
3. Business Challenges
   - Operational Challenges
   - Sales Challenges
   - Customer Experience Challenges

4. AI Opportunities
   - Automation
   - Analytics
   - Customer Engagement
   - Operational Efficiency

5. CEO Pitch

Requirements:
- Use retrieved context.
- Explain reasoning.
- Avoid generic recommendations.
- Keep output concise.
- Prioritize practical AI solutions.
""",
    model=model,
    tools=[
        retrieve_company_context
    ]
)

# =========================
# MANAGER AGENT
# =========================

manager_agent = CodeAgent(
    name="company_intelligence_manager",
    description="""
Create a complete company intelligence report.

Workflow:

Step 1:
Ask research_agent to:
- create company knowledge base
- gather information
- store information

Step 2:
Ask analysis_agent to:
- retrieve company information
- generate company report

Final report must contain:

# Company Overview

# Key Business Information

# Business Challenges

# AI Opportunities

# CEO Pitch

Return polished markdown.
Keep report under 500 words.
""",
    model=model,
    managed_agents=[
        ResearchAgent,
        AnalysisAgent
    ],
    tools=[]
)

if __name__ == "__main__":

    company_name = input("Enter company name: ")

    result = manager_agent.run(
        f"""
    Generate a company intelligence report for {company_name}.

    The research agent must first collect and store information.

    The analysis agent must then generate:

    1. Company Overview
    2. Key Business Information
    3. Business Challenges
    4. AI Opportunities
    5. CEO Pitch

    Return markdown.
    """
    )