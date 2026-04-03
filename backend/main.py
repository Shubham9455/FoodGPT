from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

from rag_engine import retrieve, build_prompt

app = FastAPI()


@app.get("/")
def root():
    return {"message": "FoodGPT API running 🚀"}


def stream_response(query: str):
    results = retrieve(query)
    prompt = build_prompt(query, results)

    stream = ollama.chat(
        model="phi",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        yield f"data: {json.dumps({'token': token})}\n\n"

    # Send structured results at end
    yield f"data: {json.dumps({'results': results})}\n\n"


@app.get("/stream")
def stream(query: str):
    return StreamingResponse(
        stream_response(query),
        media_type="text/event-stream"
    )