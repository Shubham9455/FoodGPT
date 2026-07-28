from langgraph.graph import StateGraph, END
from backend.state import GraphState
from backend.nodes.rewrite_node import rewrite_query_node
from backend.nodes.retrieve_node import retrieve_node
from backend.nodes.check_results_node import check_results_node
from backend.nodes.build_prompt_node import build_prompt_node
from backend.nodes.generate_node import generate_node
from backend.nodes.fallback_node import fallback_node


def build_foodgpt_graph() -> StateGraph:
    """Build and compile the FoodGPT LangGraph pipeline."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("rewrite", rewrite_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("build_prompt", build_prompt_node)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback", fallback_node)

    # Set entry point
    builder.set_entry_point("rewrite")

    # Define edges
    builder.add_edge("rewrite", "retrieve")

    # Conditional edge: check if results exist
    builder.add_conditional_edges(
        "retrieve",
        check_results_node,
        {
            "build_prompt": "build_prompt",
            "fallback": "fallback",
        }
    )

    builder.add_edge("build_prompt", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)

    # Compile
    graph = builder.compile()
    return graph


# Create a singleton instance
foodgpt_graph = build_foodgpt_graph()