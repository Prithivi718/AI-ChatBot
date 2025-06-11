import json
import os
from os import getenv
from typing import Any, Dict, List, Optional,Literal

from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain.tools import tool

load_dotenv()

try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    raise ImportError("`firecrawl-py` not installed. Please install using `pip install firecrawl-py`")

api_url= "https://api.firecrawl.dev"
api_key= getenv("FIRECRAWL_API_KEY")

# Create Object for FirecrawlApp
app= FirecrawlApp(api_key=api_key, api_url=api_url)


class ScrapWebsite(BaseModel):
    url: str
    formats: Optional[List[
        Literal["markdown", "html", "rawHtml", "content", "links", "screenshot", "screenshot@fullPage", "extract", "json", "changeTracking"]]] = (
        Field(
            default=["markdown"],
            description="Content formats to extract (default: ['markdown'])")
        )
    onlyMainContent: Optional[bool] = Field(
        default=False,
        description="Extract only the main content, filtering out navigation, footer, etc."
    )

@tool
def scrape_website(params: ScrapWebsite):

    """
    This tool will Scrape a Website by the URL prompted by User

    Args: From the Pydantic BaseModel
        url: str -> The address of the Website. Ex: "https://example.com/"
        formats: Optional[List[Literal["markdown","html"]]] -> The format that user recognization. Default is 'markdown'
        onlyMainContent: Optional[bool] -> Used to extract Main Content of the Results or the Whole content

    Returns:
        A (str) of the Scraped Contents

    """

    try:
        scrap = app.scrape_url(
            url=params.url,
            formats= params.formats,
            only_main_content= params.onlyMainContent
        )
        return scrap.model_dump()
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
            default= ["markdown"],
            description="Content formats to extract (default: ['markdown'])")
        )

    onlyMainContent: Optional[bool] = Field(
        default=None,
        description="Extract only the main content, filtering out navigation, footer, etc."
    )

@tool
def crawl_website(params: CrawlWebsite):

    """
    This tool will Crawl-Over a Website given the User in URL

    Args: From the Pydantic BaseModel
        url: str -> The address of the Website. Ex: "https://example.com/"
        limit: int -> The no. of search results for the User
        formats: Optional[List[Literal["markdown","html"]]] -> The format that user recognization. Default is 'markdown'
        onlyMainContent: Optional[bool] -> Used to extract Main Content of the Results or the Whole content

    Returns:
        A (str) of the Crawled contents

    """

    try:
        url = params.url
        limit = params.limit
        formats = params.formats
        maincontent = params.onlyMainContent

        crawl = app.crawl_url(
            url=url,
            limit=limit,
            scrape_options=ScrapeOptions(formats=formats, onlyMainContent=maincontent),
        )

        return crawl.model_dump_json(indent=2)
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

@tool
def search_website(params: SearchWebsite):
    """
    This tool will search for a User Query on the Website

    Args: From the Pydantic BaseModel
        query: str -> The search content of the user to be performed.
        limit: int -> The no. of search results for the User
        formats: Optional[List[Literal["markdown","html"]]] -> The format that user recognization. Default is 'markdown'
        onlyMainContent: Optional[bool] -> Used to extract Main Content of the Results or the Whole content

    Returns:
        A (str) of the Searched Results

    """

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

@tool
def map_links(params: MapUrls):

    """
    This tool will Group a List of URLs in the Website

    Args: From the Pydantic BaseModel
        url: str -> The address of the Website. Ex: "https://example.com/"
        limit: int -> The no. of search results for the User
        search: str -> Which the User wants map-links about a specific search content on Website provided by URL

    Returns:
        A List(str) of the Mapped URLs

    """

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

@tool
def extract_content(params: ExtractContent):

    """
    This tool will Extract a User-Format from the given Website by URLs

    Args: From the Pydantic BaseModel
        urls: str -> The address of the Website. Ex: "https://example.com/"
        prompt: Optional[str] -> A user-request for the extraction
        content_schema: A schema that must be extracted

    Returns:
        A (str) of the Extracted Results

    """

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


@tool
def deep_analysis(params: DeepResearch):

    """
    This tool will search for a User Query on the Website

    Args: From the Pydantic BaseModel
        query: str -> The question from user needs to be deeply-researched!
        max_depth: Optional[int] -> Maximum depth to follow links when recursively crawling a website, where default is 7.
        time_limit: Optional[int] -> Maximum time limit in seconds for the entire crawl process, where default is 270s


    Returns:
        A (str) of the Analyzed results

    """

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

