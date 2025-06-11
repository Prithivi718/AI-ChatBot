from agno.agent import Agent
from agno.tools.discord import DiscordTools
from agno.models.openrouter import OpenRouter
from config import OPENROUTER_API_KEY
from config import DISCORD_TOKEN

discord= DiscordTools(
    bot_token= DISCORD_TOKEN,
    enable_messaging=True,
    enable_message_management=True,
    enable_channel_management=True,
    enable_history=True
)

agent = Agent(
    name="Discord Agent",
    role="Role is to send Message to a Discord Channel",
    tools=[discord],
    model=OpenRouter(id="gpt-4o-mini", api_key=OPENROUTER_API_KEY),

    description="You are a Discord Agent that automatically sends messages to a specific Discord channel based on user input or triggers.",
    instructions = [
    "Always send the provided message content to the specified Discord channel without modification.",
    "Ensure the message is delivered only to the channel ID that is explicitly configured or passed as a parameter.",
    "Do not attempt to perform web searches or respond to users directly.",
    "Do not send messages to any other channels or users except the specified one.",
    "Format messages clearly and ensure any URLs, mentions, or emojis are properly rendered in Discord.",
    "Handle message sending errors (e.g., invalid token, missing permissions) gracefully and log them if needed."
    ],

    show_tool_calls=False,
    markdown=True,
)

#agent.print_response("Send message, 'Hello Interns' to direct message 1325845338287898748")
