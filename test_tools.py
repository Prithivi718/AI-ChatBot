import sys
from langchain.tools import Tool
from typing import Union, Dict, Type
import re
import json

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from os import getenv
from textwrap import dedent, indent
from time import sleep
from dotenv import load_dotenv

from firecrawl_fapi import (
    scrape_website, crawl_website, search_website,
    map_links, extract_content, deep_analysis,
    ScrapWebsite, CrawlWebsite, SearchWebsite,
    MapUrls, ExtractContent, DeepResearch
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




load_dotenv()
# Disable torch monitoring warning
sys.modules['torch.classes'] = None



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
    "scrape_website": ScrapWebsite,
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
    print(f"\nParsed Params:\n{parsed_params}\n")
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
        blacklist = ["svg+xml", "Sponsored", "of 5 bubbles"]
        for word in blacklist:
            text = text.replace(word, "")
        return text.strip()

    # Case 1: Firecrawl-style dictionary (contains "data" key)
    if isinstance(data, dict) and "data" in data:
        return [
            {
                "title": clean_text(item.get("title", "Untitled")),
                "description": clean_text(item.get("description", "")),
                "url": item.get("url", "")
            }
            for item in data["data"]
        ]

    # Case 2: Plain string or non-dictionary output (e.g., from deep_analysis)
    elif isinstance(data, str):
        return {"title": "Analysis Result", "description": clean_text(data), "url": ""}

    # Case 3: List of strings (e.g., ["result1", "result2"])
    elif isinstance(data, list):
        return [
            {"title": f"Result {i}", "description": clean_text(item), "url": ""}
            for i, item in enumerate(data)
        ]

    # Case 4: Other dictionaries (fallback)
    elif isinstance(data, dict):
        if "markdown" in data:
            title = extract_title_from_markdown(data["markdown"])
            return [{
                "title": title,
                "description": clean_text(data["markdown"]),
                "url": ""
            }]
        elif "content" in data:
            title = extract_title_from_markdown(data["content"])
            return [{
                "title": title,
                "description": clean_text(data["content"]),
                "url": ""
            }]
        elif "html" in data:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(data["html"], "html.parser")
            title = soup.title.string if soup.title else "Extracted Page"
            text = soup.get_text()
            return [{
                "title": title,
                "description": clean_text(text),
                "url": ""
            }]
        else:
            # Fallback for anything else
            raw_text = json.dumps(data, indent=2)
            return [{
                "title": "Extracted Content",
                "description": clean_text(raw_text),
                "url": ""
            }]

    # Default: Return as-is (e.g., numbers, None)
    return data

def extract_title_from_markdown(md_text: str) -> str:
    lines = md_text.splitlines()
    for line in lines:
        if line.startswith("#"):
            return line.strip("# ").strip()
    return "Untitled"



# ---------------------- Parse Prompt Template ----------------------
parse_prompt = PromptTemplate(
    template="""
        You are an intelligent agent tasked with selecting the best tool to execute a user request.
        
        ---
        
        📦 Available Tools:
        1. `search_website` — Search the web using a query (needs `query`, `limit`)
        2. `scrape_website` — Scrape a specific website (needs `url`, optional `formats`, `onlyMainContent`)
        3. `crawl_website` — Crawl a site deeply (needs `url`, `limit`, optional `formats`)
        4. `deep_analysis` — Perform deep research on a topic (needs `query`, optional `max_depth`, `time_limit`)
        5. `map_links` — Extract links from a base page (needs `url`, `limit`, optional `search`)
        6. `extract_content` — Extract specific content from URLs (needs `urls`, `prompt`, optional `content_schema`)
        
        ---
        
        📥 Based on the following user request, choose the most appropriate tool and return its required parameters in this format:
        
        ```json
        {{
          "tool_name": "<name_of_tool>",
          "params": {{
            "<param1>": "...",
            "<param2>": ...
          }}
        }}
        
        Respond only with JSON.
        
        User request: {user_input}
        """,
        input_variables=["user_input"]
)

# Handle new prompt
if prompt := input("💬 Type your prompt here: "):

    # Process and display response
    chain = parse_prompt | llm
    response = chain.invoke({"user_input": prompt})

    output_text = response.content
    print(f"\n---Input Context--\n{output_text}\n")
    llm_output = process_user_prompt(output_text)
    print(f"---LLM Output---:\n{llm_output}\n")
    structure_data = clean_web_output(llm_output)
    # print(f"---Cleaned LLM Output---:\n{structure_data}\n")
    # final_llm_response = gemini_llm_response(structure_data)
    #
    # print(f"---Final LLM Response:---\n{final_llm_response}")
