from smolagents import CodeAgent,DuckDuckGoSearchTool,GoogleSearchTool,Tool,FinalAnswerTool,tool
from langchain_huggingface import HuggingFaceEmbeddings
import os
from smolagents import tool
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
from duckduckgo_search import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import uuid


#defining Toolto make chroma doc for comapnies
@tool
def get_or_create_company(company_name: str) -> str:
    """
    Create or load a company knowledge base.
    Args:
        company_name: Name of the company.
    Returns:
        Collection name.
    """
    collection_name = company_name.lower().replace(" ", "_")
    chroma_client.get_or_create_collection(name=collection_name)

    return f"Collection ready: {collection_name}"

@tool
def research_company(company_name: str, max_results: int = 15) -> str:
    """
    Search web for company information.
    Args:
        company_name: Company name.
        max_results: Number of search results.
    Returns:
        Search results as text.
    """
    query = f"""
    {company_name}
    company overview
    recent news
    expansion plans
    business operations
    """
    output = []
    with DDGS() as ddgs:
        results = ddgs.text(query,max_results=max_results)
        for i, item in enumerate(results, 1):
            output.append(
                f"""
            SOURCE {i}
                TITLE:
                {item.get('title','')}
                    BODY:
                    {item.get('body','')}
                    URL:
                {item.get('href','')}
            """
            )
    return "\n".join(output)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

@tool
def append_company_research(
    company_name: str,
    research_text: str
) -> str:
    """
    Store company research in ChromaDB.
    Args:
        company_name: Company name.
        research_text: Research text.
    Returns:
        Success message.
    """
    collection_name = company_name.lower().replace(" ", "_")
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(research_text)
    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]
    vectordb.add_texts(
        texts=chunks,
        ids=ids,
        metadatas=[
        {"company": company_name}
        for _ in chunks
        ]
    )
    return f"Stored {len(chunks)} chunks."


@tool
def retrieve_company_context(
    company_name: str,
    query: str,
    k: int = 8
) -> str:
    """
    Retrieve relevant company information.
    Args:
        company_name: Company name.
        query: Search query.
        k: Number of chunks.
    Returns:
        Retrieved context.
    """
    collection_name = company_name.lower().replace(" ", "_")
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    docs = vectordb.similarity_search(
    query=query,
    k=k,
    filter={
        "company": company_name
    }
    )   
    return "\n\n".join(
        [doc.page_content for doc in docs]
    )
