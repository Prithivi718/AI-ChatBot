import sys
from agno.models.openrouter import OpenRouter
from agno.agent import Agent
from textwrap import dedent
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from config import OPENROUTER_API_KEY

load_dotenv()



# Initialize MCP Server
mcp= FastMCP("Interview Question Asker 🎤")

ask_qn_agent= Agent(
        name= "Interview Agent",
        role= "To ask Technical questions to the user",
        model= OpenRouter(id= "openai/gpt-4o-mini", api_key= OPENROUTER_API_KEY),
        instructions=dedent("""
            You are an intelligent technical interviewer.
            Your job is to ask one technical question at a time to the user.

            Consider the user's background in programming, data structures, web development, or system design.
            Keep your questions relevant, professional, and progressively more challenging.

            Start with basic-level questions but don't repeat same questions and gradually increase difficulty based on the user's responses.
            Ask only one question per turn. Wait for the user's answer before proceeding to the next question.

            Avoid casual talk. Keep the focus on the technical interview process.
        """),
        show_tool_calls=False,
        debug_mode=False,
        markdown=True
    )

analyze_agent= Agent(
        name= "Analyzer Agent",
        role= "Should analyze the Answer by the User for the Asked Question",
        model=  OpenRouter(id= "openai/gpt-4o-mini", api_key= OPENROUTER_API_KEY),
        instructions=dedent("""
            You are a smart and fair interview answer evaluator.
            Your job is to carefully analyze the answer provided by the user based on the question asked.

            Consider the following while analyzing:
            - Is the answer factually correct?
            - Is the explanation clear and logical?
            - Does the answer demonstrate understanding of the topic?
            - Are there any key points missing or wrongly explained?

            Your response should include:
            - A brief analysis of the answer
            - What the user did well
            - What could be improved (if any)

            Be professional, constructive, and encouraging in your tone.
        """),
        show_tool_calls=False,
        debug_mode=False,
        markdown=True
    )

reward_score_agent = Agent(
    name="RewardScoreAgent",
    role="Should reward scores for the user's answer based on the asked question",
    model=OpenRouter(id="openai/gpt-4o-mini", api_key= OPENROUTER_API_KEY),
    instructions=dedent("""
        You are an intelligent scoring agent in a technical interview system.
        Your job is to evaluate the user's answer for a given question and assign a fair score between 0 and 10.

        Consider the following while scoring:
        - Accuracy and completeness of the answer
        - Clarity and depth of explanation
        - Use of correct terminology and examples (if any)
        - Whether the answer directly addresses the question

        Your response should include:
        - The score (out of 10)
        - A short justification for the score

        Be objective and fair. Use full scores only for fully correct and well-explained answers.
    """),
    show_tool_calls=False,
    debug_mode=False,
    markdown=True
)

@mcp.tool()
def ask_qns(user_knowledge: list) -> str:
    print("🧠 [ask_qns] Tool triggered", file=sys.stderr)
    qn_prompt = f"Ask questions to user based on user knowledge: {user_knowledge}."
    qn_response = ask_qn_agent.run(qn_prompt)
    return qn_response.content

@mcp.tool()
def analyze_ans(user_ans: str) -> str:
    print(f"🔍 [analyze_ans] Analyzing: {user_ans}", file=sys.stderr)
    prompt = f"Analyse the user's answer: {user_ans}."
    analyse_response= analyze_agent.run(prompt)
    return analyse_response.content

@mcp.tool()
def reward_score(user_ans: str, ask_qn: str) -> str:
    print(f"🏆 [reward_score] Evaluating: {user_ans} against question: {ask_qn}", file=sys.stderr)
    prompt = f"Reward score based on the user's answer: {user_ans} and the question: {ask_qn}."
    reward_response = reward_score_agent.run(prompt)
    return reward_response.content


if __name__ == "__main__":
    print("🚀 MCP Interview Agent starting...", file=sys.stderr)
    mcp.run()


