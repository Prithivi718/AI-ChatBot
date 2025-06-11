from openai import OpenAI
from config import OPENROUTER_API_KEY


# Initiating Client to the OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY  # This must be valid
)

def token_stream(prompt, model):
    try:
        response = client.chat.completions.create(
        model= model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": '''Your name is Friday! You're an Excellent Document Expert! 
                            You should answer the Questions based on the Given Document provided!''',
            },
        ],
        stream=True,
    )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"\n❌ Error during LLM response: {e}"