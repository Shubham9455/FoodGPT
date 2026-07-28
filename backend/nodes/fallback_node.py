from backend.state import GraphState


def fallback_node(state: GraphState) -> GraphState:
    """Return a fallback message when no restaurants are found."""
    return {
        "answer": "No relevant restaurants found.",
        "results": [],
    }