from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from config import OPENROUTER_API_KEY
from textwrap import dedent

def response_reason_agents(user_chat):
    # 🧠 Reasoning Agent: Thinks deeply and step-by-step
    reasoning_agent = Agent(
        name="Reasoning Agent",
        role="Logical problem solver that breaks tasks into clear reasoning steps.",

        model=OpenRouter(
            id="openai/gpt-4o-mini",  # or use gemini-1.5-flash if you want speed
            api_key=OPENROUTER_API_KEY
        ),

        description="This agent analyzes the problem, thinks step-by-step, and produces structured solutions.",

        instructions=dedent
        ("""\ [Step-1]
            Understand the user's request deeply.,
            Think through the task step-by-step using your own reasoning.,
            Focus on clarity, correctness, and structure.,
            Do not consider tone or emotion — focus only on logical clarity.,
            Your output will be passed to a tone-aware agent next.,
            
            [Step-2]
            Rephrase it to match the emotional tone of the user’s original question.
            Use emojis, casual phrasing, or professional tone depending on the user's vibe.
            Do not change the logic or steps — only adapt tone, formatting, and clarity.
            Be friendly and human-like, unless the user is being formal.
        """),

        tools=[ReasoningTools
            (
            think=True,
            analyze=True,
            add_instructions=True,
            add_few_shot=True
        )
        ],
        debug_mode=False,
        show_tool_calls=False
    )

    # # 💬 Response Agent: Adapts the tone based on user's vibe
    # response_agent = Agent(
    #     name="Response Agent",
    #     role="Friendly, emotionally-aware responder that adapts the reasoning output to the user's tone.",
    #
    #     model=OpenRouter(
    #         id="google/gemini-flash-1.5-8b",  # or "mistral" or "zephyr" if cost is a concern
    #         api_key=OPENROUTER_API_KEY
    #     ),
    #
    #     description="This agent reformats the raw reasoning output into a response that matches the user's emotional tone (e.g., confused, casual, formal).",
    #
    #     instructions=dedent("""\
    #             You will receive a structured reasoning output.
    #             Rephrase it to match the emotional tone of the user’s original question.
    #             Use emojis, casual phrasing, or professional tone depending on the user's vibe.
    #             Do not change the logic or steps — only adapt tone, formatting, and clarity.
    #             Be friendly and human-like, unless the user is being formal.
    #         """),
    #
    #     tools=[],
    #     show_tool_calls=False
    # )

    # Step 1: Reasoning Agent generates raw logic
    reasoning_output= reasoning_agent.run(user_chat)
    # Step 2: Response Agent generates the proper response format
    #response =response_agent.run(f"Raw Output: {reasoning_output}")

    return reasoning_output.content



