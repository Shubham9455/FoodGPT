from typing import TypedDict, Optional, List, Dict, Any


class GraphState(TypedDict):
    """Typed state for the FoodGPT LangGraph pipeline."""
    original_query: str
    rewritten_query: Optional[str]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    prompt: Optional[str]
    answer: Optional[str]
    results: Optional[List[Dict[str, Any]]]
    error: Optional[str]
