"""
Adapter for any OpenAI-compatible server: vLLM, LM Studio, etc.
Speaks /v1/chat/completions and /v1/embeddings.
"""
import httpx

_CHAT_TIMEOUT = 120.0
_EMBED_TIMEOUT = 60.0


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key: str = "none"):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key != "none" else {}

    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        with httpx.Client(timeout=_CHAT_TIMEOUT, headers=self._headers) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json={
                "model": model,
                "messages": messages,
            })
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        with httpx.Client(timeout=_EMBED_TIMEOUT, headers=self._headers) as client:
            resp = client.post(f"{self.base_url}/embeddings", json={
                "model": model,
                "input": texts,
            })
            resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=10.0, headers=self._headers) as client:
                resp = client.get(f"{self.base_url}/models")
                resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []
