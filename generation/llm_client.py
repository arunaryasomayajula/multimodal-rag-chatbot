import httpx
from generation.prompts import SYSTEM_PROMPT, RAG_TEMPLATE
from generation.citation import format_context
from config import settings

_TIMEOUT = 120.0


def chat(question: str, chunks: list[dict], history: list[dict] | None = None) -> dict:
    context, sources = format_context(chunks)
    user_message = RAG_TEMPLATE.format(context=context, question=question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])  # keep last 3 turns (6 messages)
    messages.append({"role": "user", "content": user_message})

    with httpx.Client(base_url=settings.ollama_base_url, timeout=_TIMEOUT) as client:
        resp = client.post("/api/chat", json={
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
        })
        resp.raise_for_status()

    answer = resp.json()["message"]["content"]
    return {"answer": answer, "sources": sources}
