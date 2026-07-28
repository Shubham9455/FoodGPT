"""
RAG engine for FoodGPT.

This module maintains backward compatibility with the original API
while internally using a LangGraph pipeline for orchestration.
"""

from backend.graph import foodgpt_graph
from backend.state import GraphState


def ask(query: str):
    """
    Main entry point. Invokes the LangGraph pipeline and returns (answer, results).

    Maintains backward compatibility with existing callers (app.py, main.py).
    """
    initial_state: GraphState = {
        "original_query": query,
        "rewritten_query": None,
        "retrieved_docs": None,
        "prompt": None,
        "answer": None,
        "results": None,
        "error": None,
    }

    final_state = foodgpt_graph.invoke(initial_state)

    answer = final_state.get("answer", "Something went wrong.")
    results = final_state.get("results", [])

    return answer, results


def stream_graph(query: str):
    """
    Stream the LangGraph execution step by step.
    Yields events with node name and state updates.
    """
    initial_state: GraphState = {
        "original_query": query,
        "rewritten_query": None,
        "retrieved_docs": None,
        "prompt": None,
        "answer": None,
        "results": None,
        "error": None,
    }

    for event in foodgpt_graph.stream(initial_state):
        yield event