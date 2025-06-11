from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.openrouter import OpenRouter
from config import OPENROUTER_API_KEY

agent = Agent(
    name="Task Agent",
    role="To perform User Requested Task",

    tools=[GoogleSearchTools()],
    model=OpenRouter(id="gpt-4o-mini", api_key=OPENROUTER_API_KEY),

    description="You are a Task Agent that helps users search the web for relevant results based on their requests.",
    instructions = [
    "Given a request by the user, perform a web search to find relevant and accurate results.",
    "Search for up to 10 web items and return the top 5 most relevant and unique results.",
    "Ensure the search is performed in English only.",
    "It would be Better if the URL of the documentation is given for each Search Results"
    "Return concise and user-friendly summaries with valid URLs where possible."],

    show_tool_calls=False,
    debug_mode=True,
)
agent.print_response("List the Updates by 'Marvel Studios' for 'Doomsday' ")