from qdrant_client import QdrantClient
from config import QDRANT_API_URL, QDRANT_API_KEY, DEFAULT_MODEL
from studbot_response import token_stream


# Q-Drant Initialization Process
qdrant_client = QdrantClient(url=QDRANT_API_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2")
qdrant_client.set_sparse_model("Qdrant/bm25")


def retrieve_and_respond(collection_name, question, model= DEFAULT_MODEL):
    try:
        points = qdrant_client.query(
            collection_name=collection_name,
            query_text=question,
            limit=7
        )

        if not points:
            print("❗ No relevant chunks found for the given question.")
            return None

        context_chunks = "\n\n".join(
            [f"[Chunk {i + 1}]\n{p.document}" for i, p in enumerate(points)]
        )

        lower_q = question.lower()

        if "study plan" in lower_q or "revision plan" in lower_q:
                format_prompt = f"""
        You're 'Friday', an AI Study Planner. Based on the following study material, create a **daily revision plan**.
        Include:
        - Day-wise breakdown (Day 1, Day 2, ...)
        - Key topics to cover
        - Estimated time per topic

        Context:
        {context_chunks}

        Task: {question}

        Give a clear, bullet-point plan.
        """
        elif "quiz" in lower_q or "mcq" in lower_q or "questions" in lower_q:
                format_prompt = f"""
        You're 'Friday', a quiz generation expert. Based on the content below, generate a quiz with:
        - 10 multiple choice questions (MCQs)
        - 4 options each
        - Highlight the correct option

        Context:
        {context_chunks}

        Task: {question}

        Format:
        Q1. Question text?
        A. Option 1  
        B. Option 2  
        C. Option 3 ✅  
        D. Option 4  

        Repeat for 10 questions.
        """
        else:
                format_prompt = f"""
        You're 'Friday', an AI tutor. Based on the content below, answer the user's question clearly.

        Context:
        {context_chunks}

        Question: {question}
        """

            # print("\n🤖 Answer:\n")
            # Typing effect: Stream output into a single markdown container
            # Collect and return the full response as a string

        # Stream and collect the response
        response = token_stream(format_prompt, model)
        final_output = ""
        for char in response:
            final_output += char

        return final_output

    except Exception as e:
       return f"\n❌ Error during LLM response: {e}"


