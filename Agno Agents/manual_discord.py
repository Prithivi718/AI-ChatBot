import discord

from config import DISCORD_TOKEN
from discord_github import DiscordTools
from reasoning_agent import response_reason_agents

intents = discord.Intents.default()
intents.message_content = True  # Needed to read user messages

client = discord.Client(intents=intents)

agent = DiscordTools()

# This function will use the LLM to get a response based on user message
async def response_model(user_request):
    try:
        # Use OpenRouter or OpenAI here based on your setup
        final_response = response_reason_agents(user_request)

        return final_response

    except Exception as e:
        return f"\n❌ Error during LLM response: {e}"

@client.event
async def on_ready():
    print(f"Bot is online as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    # Step 1: Read user message
    user_request = message.content
    print(f"User message detected: {user_request}")

    # Step 2: Get AI response
    reply = await response_model(user_request)

    # Step 3: Send it using your agent's function
    await agent.send_message(message.channel.id, reply)

# Start the bot
client.run(DISCORD_TOKEN)

