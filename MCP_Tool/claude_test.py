"""
Firecrawl Agent with Tool Calling and Parameter Extraction
This file implements the main agent that processes natural language and calls appropriate tools.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from firecrawl_tools import get_firecrawl_tools


class FirecrawlAgent:
    """
    Main agent class that handles natural language processing and tool calling
    """

    def __init__(self, firecrawl_api_key: str, google_api_key: str):
        """Initialize the agent with API keys"""
        self.firecrawl_api_key = firecrawl_api_key
        self.google_api_key = google_api_key

        # Initialize Google Gen-AI LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=google_api_key,
            temperature=0.1
        )

        # Get Firecrawl tools
        self.tools = get_firecrawl_tools(firecrawl_api_key)

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

        # Define tool selection patterns
        self.tool_patterns = {
            'scrape_website': [
                r'scrape?\s+(?:the\s+)?(?:website|page|url|site)',
                r'get\s+content\s+from',
                r'extract\s+(?:data|content|text)\s+from\s+(?:website|url|page)',
                r'fetch\s+(?:data|content)\s+from'
            ],
            'crawl_website': [
                r'crawl\s+(?:the\s+)?(?:website|site)',
                r'get\s+multiple\s+pages',
                r'crawl\s+\d+\s+pages',
                r'crawl.*limit.*\d+',
                r'spider\s+(?:the\s+)?website'
            ],
            'search_website': [
                r'search\s+(?:for\s+)?(?:websites?|web|internet)',
                r'find\s+(?:websites?|pages?)\s+(?:about|with|containing)',
                r'search\s+(?:the\s+)?(?:web|internet)\s+for',
                r'web\s+search'
            ],
            'map_links': [
                r'(?:map|find|get|list)\s+(?:all\s+)?links',
                r'discover\s+links',
                r'find\s+(?:all\s+)?(?:urls?|links)\s+(?:on|from)',
                r'map\s+(?:the\s+)?(?:website|site)\s+structure'
            ],
            'extract_content': [
                r'extract\s+(?:structured\s+)?(?:data|content|information)',
                r'get\s+specific\s+(?:data|information)',
                r'parse\s+(?:data|content)\s+from',
                r'extract.*using.*(?:schema|structure|format)'
            ],
            'deep_analysis': [
                r'(?:deep|thorough|comprehensive)\s+(?:analysis|research)',
                r'research\s+(?:about|on)',
                r'analyze\s+(?:deeply|thoroughly)',
                r'in-depth\s+(?:analysis|research|study)'
            ]
        }

    def identify_tool(self, user_input: str) -> Optional[str]:
        """
        Identify which tool to use based on user input patterns
        """
        user_input_lower = user_input.lower()

        for tool_name, patterns in self.tool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return tool_name

        # Default fallback logic
        if any(keyword in user_input_lower for keyword in ['http', 'www', '.com', '.org', '.net']):
            if any(keyword in user_input_lower for keyword in ['crawl', 'multiple', 'pages', 'limit']):
                return 'crawl_website'
            elif any(keyword in user_input_lower for keyword in ['links', 'map', 'discover']):
                return 'map_links'
            else:
                return 'scrape_website'

        if any(keyword in user_input_lower for keyword in ['search', 'find']):
            return 'search_website'

        return None

    def extract_parameters(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameters from user input based on the selected tool
        """
        params = {}
        user_input_lower = user_input.lower()

        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, user_input)

        if tool_name == 'scrape_website':
            if urls:
                params['url'] = urls[0]

            # Extract format preferences
            if 'html' in user_input_lower:
                params['formats'] = ['html']
            elif 'json' in user_input_lower:
                params['formats'] = ['json']
            elif 'markdown' in user_input_lower or 'md' in user_input_lower:
                params['formats'] = ['markdown']

            # Extract main content preference
            if any(phrase in user_input_lower for phrase in ['main content', 'only content', 'no navigation']):
                params['onlyMainContent'] = True

        elif tool_name == 'crawl_website':
            if urls:
                params['url'] = urls[0]

            # Extract limit
            limit_match = re.search(r'(?:limit|max|maximum|up to)\s*(\d+)', user_input_lower)
            if limit_match:
                params['limit'] = int(limit_match.group(1))
            else:
                # Look for standalone numbers that might indicate limit
                number_match = re.search(r'\b(\d+)\s*(?:pages?|links?|results?)', user_input_lower)
                if number_match:
                    params['limit'] = int(number_match.group(1))
                else:
                    params['limit'] = 10  # default

        elif tool_name == 'search_website':
            # Extract search query (remove common command words)
            query = user_input
            for phrase in ['search for', 'find', 'look for', 'search']:
                query = re.sub(rf'\b{phrase}\b', '', query, flags=re.IGNORECASE)
            params['query'] = query.strip()

            # Extract limit
            limit_match = re.search(r'(?:limit|max|maximum|up to)\s*(\d+)', user_input_lower)
            if limit_match:
                params['limit'] = int(limit_match.group(1))
            else:
                params['limit'] = 10

        elif tool_name == 'map_links':
            if urls:
                params['url'] = urls[0]

            # Extract search term for filtering links
            search_terms = []
            if 'about' in user_input_lower:
                about_match = re.search(r'about\s+([^,.\n]+)', user_input_lower)
                if about_match:
                    search_terms.append(about_match.group(1).strip())

            params['search'] = ' '.join(search_terms) if search_terms else ''

            # Extract limit
            limit_match = re.search(r'(?:limit|max|maximum|up to)\s*(\d+)', user_input_lower)
            params['limit'] = int(limit_match.group(1)) if limit_match else 20

        elif tool_name == 'extract_content':
            params['urls'] = urls if urls else []

            # Extract extraction prompt
            prompt_indicators = ['extract', 'get', 'find', 'pull out']
            for indicator in prompt_indicators:
                if indicator in user_input_lower:
                    prompt_part = user_input[user_input_lower.find(indicator):]
                    params['prompt'] = prompt_part
                    break

            if 'prompt' not in params:
                params['prompt'] = user_input

        elif tool_name == 'deep_analysis':
            # Extract research query
            query_indicators = ['research', 'analyze', 'analysis of', 'study']
            query = user_input
            for indicator in query_indicators:
                if indicator in user_input_lower:
                    query = user_input[user_input_lower.find(indicator) + len(indicator):].strip()
                    break

            params['query'] = query

            # Extract depth and time limits
            depth_match = re.search(r'(?:depth|levels?)\s*(\d+)', user_input_lower)
            if depth_match:
                params['max_depth'] = int(depth_match.group(1))

            time_match = re.search(r'(?:time|timeout|limit)\s*(\d+)\s*(?:seconds?|minutes?)', user_input_lower)
            if time_match:
                seconds = int(time_match.group(1))
                if 'minute' in user_input_lower:
                    seconds *= 60
                params['time_limit'] = seconds

        return params

    def process_request(self, user_input: str) -> str:
        """
        Main method to process user requests
        """
        try:
            # First, try to identify the specific tool and extract parameters
            tool_name = self.identify_tool(user_input)

            if tool_name:
                params = self.extract_parameters(user_input, tool_name)

                # Find the corresponding tool
                selected_tool = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        selected_tool = tool
                        break

                if selected_tool:
                    # Validate required parameters
                    if self._validate_parameters(tool_name, params):
                        # Execute the tool directly
                        result = selected_tool._run(**params)
                        return f"Tool: {tool_name}\nParameters: {json.dumps(params, indent=2)}\nResult: {result}"
                    else:
                        # Fall back to agent if parameters are incomplete
                        return self.agent.run(user_input)
                else:
                    return f"Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}"
            else:
                # Use the agent for general processing
                return self.agent.run(user_input)

        except Exception as e:
            return f"Error processing request: {str(e)}"

    @staticmethod
    def _validate_parameters(tool_name: str, params: Dict[str, Any]) -> bool:
        """
        Validate that required parameters are present for each tool
        """
        required_params = {
            'scrape_website': ['url'],
            'crawl_website': ['url', 'limit'],
            'search_website': ['query', 'limit'],
            'map_links': ['url', 'limit', 'search'],
            'extract_content': ['urls', 'prompt'],
            'deep_analysis': ['query']
        }

        if tool_name not in required_params:
            return False

        for param in required_params[tool_name]:
            if param not in params or not params[param]:
                return False

        return True

    def get_available_tools(self) -> List[str]:
        """Return list of available tool names"""
        return [tool.name for tool in self.tools]


def main():
    """Example usage of the Firecrawl Agent"""

    # Load API keys from environment variables
    firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')

    if not firecrawl_api_key or not google_api_key:
        print("Please set FIRECRAWL_API_KEY and GOOGLE_API_KEY environment variables")
        return

    # Initialize the agent
    agent = FirecrawlAgent(firecrawl_api_key, google_api_key)

    print("Firecrawl Agent initialized!")
    print(f"Available tools: {agent.get_available_tools()}")
    print("\nYou can now interact with the agent. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nAgent: Processing your request...")
        response = agent.process_request(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()