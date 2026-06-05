import generation.model_registry as registry


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider, model = registry.get_embed_provider()
    return provider.embed(texts, model)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
