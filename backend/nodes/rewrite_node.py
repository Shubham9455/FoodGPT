import ollama
from state import GraphState

REWRITE_PROMPT = """
You are a restaurant search optimizer.

Convert the user's request into a retrieval-friendly sentence.

Rules:
- Preserve meaning
- Expand vague phrases
- Include cuisine, mood, dining style if implied
- Keep under 20 words
- Return ONLY the rewritten sentence

Examples:

User: cheap dosa place
Output: Affordable South Indian restaurants serving dosa

User: date night place with good vibes
Output: Romantic restaurants with cozy ambience suitable for date nights

User query:
{query}
"""


def rewrite_query_node(state: GraphState) -> GraphState:
    """Rewrite the user's query into a retrieval-friendly sentence."""
    query = state["original_query"]

    try:
        response = ollama.chat(
            model="phi",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}]
        )
        rewritten = response["message"]["content"].strip()
    except Exception:
        rewritten = query

    print(f"\nOriginal: {query}")
    print(f"Optimized: {rewritten}\n")

    return {"rewritten_query": rewritten}