import json
from os import getenv
from typing import Any, Dict, List, Optional,Literal

from dotenv import load_dotenv
from pydantic import BaseModel,Field
from mcp.server.fastmcp import FastMCP
load_dotenv()

try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    raise ImportError("`firecrawl-py` not installed. Please install using `pip install firecrawl-py`")

api_url= "https://api.firecrawl.dev"
api_key= getenv("FIRECRAWL_API_KEY")

# Create Object for FirecrawlApp
app= FirecrawlApp(api_key=api_key, api_url=api_url)

# MCP Tool Call
mcp= FastMCP("FireCrawl Agent")

class ScrapWebsite(BaseModel):
    url: str
    formats: Optional[List[
        Literal["markdown", "html", "rawHtml", "content", "links", "screenshot", "screenshot@fullPage", "extract", "json", "changeTracking"]]] = (
        Field(
            default=["markdown"],
            description="Content formats to extract (default: ['markdown'])")
        )
    onlyMainContent: Optional[bool] = Field(
        default=None,
        description="Extract only the main content, filtering out navigation, footer, etc."
    )
    actions: Optional[List[dict]] = Field(
        default=None,
        description="Custom actions like clicks or scrolls to perform before scraping"
    )


@mcp.tool()
def scrape_website(params: ScrapWebsite):

    try:
        scrap = app.scrape_url(
            url=params.url,
            formats= params.formats,
            only_main_content= params.onlyMainContent,
            actions= params.actions
        )
        return scrap.model_dump_json()
    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)


class CrawlWebsite(BaseModel):
    url: str
    limit: int
    formats: Optional[List[
        Literal[
            "markdown", "html", "rawHtml", "content", "links", "screenshot", "screenshot@fullPage", "extract", "json", "changeTracking"]]] = (
        Field(
            default=["markdown"],
            description="Content formats to extract (default: ['markdown'])")
        )

    onlyMainContent: Optional[bool] = Field(
        default=None,
        description="Extract only the main content, filtering out navigation, footer, etc."
    )

@mcp.tool()
def crawl_website(params: CrawlWebsite):
    url= params.url
    limit= params.limit
    formats= params.formats
    maincontent = params.onlyMainContent

    try:
        crawl = app.crawl_url(
            url=url,
            limit=limit,
            scrape_options=ScrapeOptions(formats=formats, onlyMainContent=maincontent),
        )

        return crawl.model_dump_json()
    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)


class SearchWebsite(BaseModel):
    query: str
    limit: int
    formats: Optional[List[
        Literal[
            "markdown", "html", "rawHtml", "content", "links", "screenshot", "screenshot@fullPage", "extract", "json", "changeTracking"]]] = (
        Field(
            default=["markdown"],
            description="Content formats to extract (default: ['markdown'])")
    )

    onlyMainContent: Optional[bool] = Field(
        default=None,
        description="Extract only the main content, filtering out navigation, footer, etc."
    )

@mcp.tool()
def search_website(params: SearchWebsite):
    query= params.query
    limit= params.limit

    try:
        search_result = app.search(
            query= query,
            limit=limit,
            scrape_options=ScrapeOptions(formats=params.formats, onlyMainContent=params.onlyMainContent),
        )

        return search_result.model_dump_json()
    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)


class MapUrls(BaseModel):
    url: str
    limit: int
    search: str

@mcp.tool()
def map_links(params: MapUrls):

    try:
        search_result = app.map_url(
            url= params.url,
            limit= params.limit,
            search= params.search
        )
        return search_result.model_dump_json()

    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)


class ExtractContent(BaseModel):
    urls: List[str]
    prompt: Optional[str]
    content_schema: Optional[Dict[str, Any]] = None  # Accept dict or Pydantic class

@mcp.tool()
def extract_content(params: ExtractContent):

    try:
        extract_result= app.extract(
            urls= params.urls,
            prompt= params.prompt,
            schema= params.content_schema
        )

        return extract_result.model_dump_json()
    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)


class DeepResearch(BaseModel):
    query: str

    max_depth: Optional[int] = Field(
        default= 7,
        description= "Maximum depth to follow links when recursively crawling a website."
    )

    time_limit: Optional[int]= Field(
        default=270,
        description= "Maximum time limit in seconds for the entire crawl process."
    )


@mcp.tool()
def deep_analysis(params: DeepResearch):

    try:
        research_result= app.deep_research(
            query= params.query,
            max_depth= params.max_depth,
            time_limit= params.time_limit,
            analysis_prompt=(
                "Analyze the gathered information for the key ways AI is influencing the education sector. "
                "Focus on learning outcomes, personalized education, teacher support, and ethical implications."
            ),
            system_prompt=(
                "You are a research assistant specializing in the intersection of AI and education. "
                "Your job is to analyze web-scraped content and summarize key themes and findings relevant to education."
            )

        )
        return json.dumps(research_result, indent=2)

    except Exception as e:
        # ❌ Handle and return error in JSON format
        return json.dumps({"error": str(e)}, indent=2)



if __name__ == "__main__":
    mcp.run()
