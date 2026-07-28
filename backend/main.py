from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import ask, stream_graph

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "FoodGPT API running 🚀"}


def stream_response(query: str):
    """Legacy streaming endpoint - now backed by LangGraph pipeline."""
    answer, results = ask(query)

    # Simulate token streaming by yielding the full answer
    buffer = ""
    for token in answer.split():
        buffer += token + " "
        yield f"data: {json.dumps({'token': buffer.strip()})}\n\n"

    # Send structured results at end
    yield f"data: {json.dumps({'results': results})}\n\n"


@app.get("/stream")
def stream(query: str):
    return StreamingResponse(
        stream_response(query),
        media_type="text/event-stream"
    )


def langgraph_stream_response(query: str):
    """Stream LangGraph node-level events as SSE."""
    for event in stream_graph(query):
        # Each event is a dict like {"node_name": {"field": "value", ...}}
        yield f"data: {json.dumps({'event': event})}\n\n"


@app.get("/langgraph/stream")
def langgraph_stream(query: str):
    """New endpoint that streams LangGraph node execution events."""
    return StreamingResponse(
        langgraph_stream_response(query),
        media_type="text/event-stream"
    )