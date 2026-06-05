from generation.prompts import SYSTEM_PROMPT, RAG_TEMPLATE
from generation.citation import format_context
import generation.model_registry as registry


def chat(question: str, chunks: list[dict], history: list[dict] | None = None,
         model_id: str | None = None) -> dict:
    context, sources = format_context(chunks)
    user_message = RAG_TEMPLATE.format(context=context, question=question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})

    entry = registry.get_llm_entry(model_id)
    answer = entry.provider.chat(messages, entry.model_id)
    return {"answer": answer, "sources": sources}
