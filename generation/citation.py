def format_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    parts = []
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        source_file = chunk.get("source_file", "unknown")
        page = chunk.get("page_number")
        page_str = f", p.{page}" if page else ""
        parts.append(f"[{i}] [{source_file}{page_str}]\n{chunk['content']}")
        sources.append({"ref": i, "file": source_file, "page": page})
    return "\n\n".join(parts), sources
