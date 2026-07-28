from backend.state import GraphState


def build_prompt_node(state: GraphState) -> GraphState:
    """Build the prompt for the LLM using retrieved restaurant data."""
    query = state["original_query"]
    results = state["retrieved_docs"]

    context = "\n".join(
        f"{i+1}. {r['name']} ({r['location']}) | {r['cuisines']} | Rating: {r['rate']}"
        for i, r in enumerate(results)
    )

    prompt = f"""
You are FoodGPT.

STRICT RULES:
- Use ONLY the restaurants below
- Do NOT add extra text
- Do NOT explain locations
- Keep output SHORT

User query: "{query}"

Restaurants:
{context}

Return EXACTLY this format (no extra lines before/after):

1. Name (Location) - short reason
2. Name (Location) - short reason
3. Name (Location) - short reason
"""

    return {"prompt": prompt}