import streamlit as st
import sys
from langchain.tools import Tool
from typing import Union, Dict, Type

import json
import re
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from os import getenv
from textwrap import dedent
from time import sleep
from dotenv import load_dotenv
from agno.models.openrouter import OpenRouter
from agno.agent import Agent
from firecrawl_fapi import (
    scrape_website, crawl_website, search_website,
    map_links, extract_content, deep_analysis,
    ScrapWebsite, CrawlWebsite, SearchWebsite,
    MapUrls, ExtractContent, DeepResearch
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




load_dotenv()
# Disable torch monitoring warning
sys.modules['torch.classes'] = None

# ---------------------- Streamlit Layout ----------------------
st.set_page_config(page_title="Firecrawl Agent", layout="centered")
st.title("🔥 Firecrawl Agent")
st.caption("Scrape · Crawl · Extract · Research")

# ---------------------- Session State ----------------------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------- Typing Stream Simulation ----------------------
def simulate_typewriter(text: str):
    for word in text.split():
        yield word + " "
        sleep(0.05)


# ---------------------- Agent LLM Configuration ----------------------
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = getenv("GOOGLE_API_KEY")

# -------------------- Tool Definitions --------------------
tools = [
    Tool(
        name="scrape_website",
        func=scrape_website,
        description=dedent("""
            Scrape a single website URL. Parameters:
            - url: The website URL to scrape (required)
            - formats: Content formats to extract (markdown, html, etc.)
            - onlyMainContent: Whether to extract only main content
        """)
    ),
    Tool(
        name="crawl_website",
        func=crawl_website,
        description=dedent("""
            Crawl a website. Parameters:
            - url: The website URL to crawl (required)
            - limit: Maximum pages to crawl (required)
            - formats: Content formats to extract
            - onlyMainContent: Whether to extract only main content
        """)
    ),
    Tool(
        name="search_website",
        func=search_website,
        description=dedent("""
            Search the web. Parameters:
            - query: Search terms (required)
            - limit: Maximum results to return (required)
            - formats: Content formats to retrieve
            - onlyMainContent: Whether to exclude navigation/boilerplate
        """)
    ),
    Tool(
        name="map_links",
        func=map_links,
        description=dedent("""
            Map links on a webpage. Parameters:
            - url: Base URL to start mapping (required)
            - limit: Maximum links to return (required)
            - search: Filter links containing this text
        """)
    ),
    Tool(
        name="extract_content",
        func=extract_content,
        description=dedent("""
            Extract content from URLs. Parameters:
            - urls: List of target URLs (required)
            - prompt: Natural language instructions for extraction
            - content_schema: Optional JSON schema for output
        """)
    ),
    Tool(
        name="deep_analysis",
        func=deep_analysis,
        description=dedent("""
            Conduct deep research. Parameters:
            - query: Research topic (required)
            - max_depth: Link recursion depth (default=3)
            - time_limit: Maximum research time in seconds (default=300)
        """)
    )
]

# ---------------------- Response Processing ----------------------
llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.7
)
llm_with_tools= llm.bind_tools(tools= tools)

def gemini_llm_response(raw_output: Union[dict, list]) -> str:
    """Formatter for large/complex responses using Google GenAI"""
    try:
        if isinstance(raw_output, (dict, list)):
            content = json.dumps(raw_output, indent=2)
        else:
            content = str(raw_output)

        format_prompt = dedent(f"""
        You are a professional AI assistant trained to transform raw web data into polished, user-friendly outputs. Follow this structured approach to format the content effectively:

        ---

        ### 🌟 **Executive Summary**  
        [Provide a 2–3 line overview answering:  
        1. What is the core topic/purpose of this data?  
        2. Why is it relevant to the user?  
        3. Key takeaway at a glance.]  

        ---

        ### 📂 **Content Formatting Rules**  
        *Apply these based on the input type:*

        #### 🔍 **For Search/Crawl/Scrape Results (Lists)**  
        **→ Section Title:** `### � Top Results`  
        - Format each entry as:  
          `1. 🔍 **<Title>** — <Description> (Max 1 line, extract key intent/utility).`  
        - *Limit to 5–7 most relevant items.*  
        - **Links Section:**  
          `### 🔗 Useful Links`  
          - Markdown format: `[<Title>](<URL>)`  
          - Include ALL valid URLs from the data.  

        #### 📜 **For Long-Form Content (Articles, Research, etc.)**  
        **Structure:**  
        - **Overview** (2–3 bullet points)  
        - **Key Insights** (Bulleted list of 3–5 core ideas)  
        - **Important Facts** (Data points, stats, or critical details)  
        - **Actionable Recommendations** (If applicable)  
        *Use subheadings (`###`), bold text, and line breaks for readability.*  

        #### ❓ **For Q&A or FAQ Content**  
        **→ Section Title:** `### ❓ Key Questions Answered`  
        - Format each as:  
          `**Q:** <Question>  
          **A:** <Concise answer (1–3 lines)>`  

        ---

        ### 🧹 **Data Cleaning Guidelines**  
        - **Remove:**  
          - Noise: `"svg"`, `"bubbles"`, `"Sponsored"`, ads, pagination text.  
          - Redundant metadata (e.g., `"last updated"` unless critical).  
          - Broken/empty fields or duplicate entries.  
        - **Preserve:**  
          - Valid hyperlinks, key statistics, and named entities.  
          - Hierarchical structure (e.g., H1/H2 headings as sub-sections).  

        ---

        ### ✨ **Final Output Requirements**  
        - Language: Clear, concise, neutral tone.  
        - Format: Strict markdown (headings, bold, lists).  
        - Length: Condense without losing meaning (avoid walls of text).  

        ---

        **Apply this to the following data and return ONLY the formatted output:**  

        {content}  
        """)

        return llm.invoke(format_prompt).content

    except Exception as e:
        return f"Error formatting complex response: {str(e)}"


# ---------------------- Main Processing Function ----------------------
class ToolCall(BaseModel):
    tool_name: str
    params: dict


# OutputParser mapping
tool_schemas: Dict[str, Type[BaseModel]] = {
    "crawl_website": CrawlWebsite,
    "scrap_website": ScrapWebsite,
    "search_website": SearchWebsite,
    "map_links": MapUrls,
    "extract_content": ExtractContent,
    "deep_analysis": DeepResearch
}

tool_parsers: Dict[str, PydanticOutputParser] = {
    tool_name: PydanticOutputParser(pydantic_object=model)
    for tool_name, model in tool_schemas.items()
}

# Tool Mapping:
tool_mapping= {
    'scrape_website': scrape_website,
    'crawl_website': crawl_website,
    'search_website': search_website,
    'map_links': map_links,
    'extract_content': extract_content,
    'deep_analysis': deep_analysis
}

def process_user_prompt(json_text: str):
    base_parser = PydanticOutputParser(pydantic_object=ToolCall)
    tool_call = base_parser.parse(json_text)

    # Pick correct parser based on tool name
    selected_parser = tool_parsers[tool_call.tool_name]
    parsed_params = selected_parser.pydantic_object(**tool_call.params)

    # Now you can invoke dynamically:
    tool_output = tool_mapping[tool_call.tool_name].invoke({"params": parsed_params})

    try:
        parsed_output = json.loads(tool_output)
    except Exception:
        parsed_output = tool_output  # Keep it raw if not JSON

    return parsed_output

# ---------------------- Cleaning of Data ----------------------
def clean_web_output(data):
    def clean_text(text):
        if not isinstance(text, str):
            return text
        blacklist = ["svg+xml", r"\bSponsored\b", r"\d+ of \d+ bubbles"]  # Regex for exact matches
        for pattern in blacklist:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    # Handle multiple response structures
    if isinstance(data, dict):
        if "data" in data:  # Firecrawl-style
            items = data["data"]
        elif "results" in data:  # Alternative API
            items = data["results"]
        else:
            items = [data]  # Single-item wrap

        return [
            {
                "title": clean_text(item.get("title", "Untitled")),
                "description": clean_text(item.get("description", "")),
                "url": item.get("url", "")
            }
            for item in items
        ]
    elif isinstance(data, list):
        return [clean_web_output(item) for item in data]
    return data


# ---------------------- Parse Prompt Template ----------------------
parse_prompt = PromptTemplate(
            template="""
        Extract the tool name and its required arguments from the user request.

        Respond only in JSON as:
        {{
          "tool_name": "search_website",
          "params": {{
            "query": "...",
            "limit": ...
          }}
        }}

        User request: {user_input}
        """,
            input_variables=["user_input"]
        )

# ---------------------- Chat Interface ----------------------
st.markdown("<h3 style='text-align: center;'>🧠 What can I help with?</h3>", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new prompt
if prompt := st.chat_input("💬 Type your prompt here"):
    # Add user message to chat
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Process and display response
    chain = parse_prompt | llm
    response = chain.invoke({"user_input": prompt})

    with st.status(label="Selecting Tool and Parsing User Prompt", expanded=False) as status:
        output_text = response.content
        llm_output = process_user_prompt(output_text)
        status.success("Finished Tool Selection and Parsing Prompt")

    with st.status(label="Cleaning User", expanded=False) as status:
        clean_data = clean_web_output(llm_output)
        final_llm_response = gemini_llm_response(clean_data)
        status.success("Finished Formatting LLM Output")

    with st.chat_message("assistant"):
        st.write(final_llm_response)
        st.session_state.chat_history.append({"role": "assistant", "content": final_llm_response})