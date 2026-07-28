from state import GraphState
from typing import Literal


def check_results_node(state: GraphState) -> Literal["build_prompt", "fallback"]:
    """Check if any documents were retrieved after reranking. Route to build_prompt or fallback."""
    docs = state.get("reranked_docs", [])
    if docs and len(docs) > 0:
        return "build_prompt"
    return "fallback"
