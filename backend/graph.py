from langgraph.graph import StateGraph, END
from state import GraphState
from nodes.rewrite_node import rewrite_query_node
from nodes.hybrid_retrieve_node import hybrid_retrieve_node
from nodes.rerank_node import rerank_node
from nodes.check_results_node import check_results_node
from nodes.build_prompt_node import build_prompt_node
from nodes.generate_node import generate_node
from nodes.fallback_node import fallback_node


def build_foodgpt_graph() -> StateGraph:
    """Build and compile the FoodGPT LangGraph pipeline with hybrid search + reranking."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("rewrite", rewrite_query_node)
    builder.add_node("hybrid_retrieve", hybrid_retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("build_prompt", build_prompt_node)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback", fallback_node)

    # Set entry point
    builder.set_entry_point("rewrite")

    # Define edges
    builder.add_edge("rewrite", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "rerank")

    # Conditional edge: check if reranked results exist
    builder.add_conditional_edges(
        "rerank",
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