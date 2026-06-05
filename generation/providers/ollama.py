import httpx

_CHAT_TIMEOUT = 120.0
_EMBED_TIMEOUT = 60.0


class OllamaProvider:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        with httpx.Client(base_url=self.base_url, timeout=_CHAT_TIMEOUT) as client:
            resp = client.post("/api/chat", json={
                "model": model,
                "messages": messages,
                "stream": False,
            })
            resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        vectors = []
        with httpx.Client(base_url=self.base_url, timeout=_EMBED_TIMEOUT) as client:
            for text in texts:
                resp = client.post("/api/embeddings", json={"model": model, "prompt": text})
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
                resp = client.get("/api/tags")
                resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
