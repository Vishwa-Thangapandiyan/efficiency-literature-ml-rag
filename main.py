import asyncio
import json
import threading
from contextlib import asynccontextmanager

import numpy as np
import ollama
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from pipeline import load_or_build_pipeline
from retrieval import reciprocal_rank_fusion


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading pipeline...")
    index, bm25, all_chunks = await asyncio.to_thread(load_or_build_pipeline)
    app.state.index = index
    app.state.bm25 = bm25
    app.state.all_chunks = all_chunks
    print(f"Pipeline ready — {len(all_chunks)} chunks loaded.")
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


def _retrieval_loop(query: str, index, bm25, all_chunks, k: int = 10):
    query_embedding = ollama.embed(model="nomic-embed-text", input=[query])["embeddings"]
    query_vec = np.array(query_embedding, dtype="float32")
    _, faiss_ranked = index.search(query_vec, k)
    faiss_ranked = faiss_ranked[0]

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    bm25_ranked = np.argsort(scores)[::-1][:k]

    ranks = reciprocal_rank_fusion(faiss_ranked, bm25_ranked)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [all_chunks[int(idx)] for idx, _ in sorted_ranks[:5]]


def _judge_sufficiency(query: str, chunks: list) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""Question: {query}

Context: {context}

Does the retrieved context contain enough information to fully answer the question?
if yes, respond with exactly: SUFFICIENT
if no, respond with exactly: SEARCH <a better search query to find the missing information>
"""
    return ollama.generate(model="llama3.1:8b", prompt=prompt)["response"].strip()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _stream_agentic_rag(query: str, index, bm25, all_chunks, max_iterations: int = 3):
    try:
        top_chunks_retrieved = []
        curr_query = query

        for i in range(max_iterations):
            status = "Searching..." if i == 0 else "Searching again..."
            yield _sse({"type": "status", "message": status})

            chunks = await asyncio.to_thread(_retrieval_loop, curr_query, index, bm25, all_chunks)
            top_chunks_retrieved.extend(chunks)

            yield _sse({"type": "status", "message": "Checking sufficiency..."})
            decision = await asyncio.to_thread(_judge_sufficiency, query, top_chunks_retrieved)

            if "SUFFICIENT" in decision:
                break
            curr_query = decision.replace("SEARCH", "").strip()

        yield _sse({"type": "status", "message": "Generating answer..."})

        context = "\n\n".join(c["text"] for c in top_chunks_retrieved)
        prompt = f"""Answer the question thoroughly using only the context below.
Structure your answer with:
- A brief direct answer first
- Supporting details organized under clear headings
- Cite which paper each point comes from, by title

Context:
{context}

Question: {query}

Answer:"""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_stream():
            try:
                for chunk in ollama.generate(model="llama3.1:8b", prompt=prompt, stream=True):
                    token = chunk["response"] if isinstance(chunk, dict) else getattr(chunk, "response", "")
                    if token:
                        asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield _sse({"type": "error", "message": str(item)})
                break
            yield _sse({"type": "token", "text": item})

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})


@app.post("/ask")
async def ask(body: QueryRequest, request: Request):
    return StreamingResponse(
        _stream_agentic_rag(
            body.query,
            request.app.state.index,
            request.app.state.bm25,
            request.app.state.all_chunks,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())
