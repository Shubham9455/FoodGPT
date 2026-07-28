import ollama
from backend.state import GraphState


def generate_node(state: GraphState) -> GraphState:
    """Generate the final answer using Ollama and the built prompt."""
    prompt = state["prompt"]

    try:
        response = ollama.chat(
            model="phi",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"]
        return {"answer": answer}
    except Exception as e:
        return {"answer": "Something went wrong while generating response."}