"""
Model registry — built at app startup from config.
Maps model_id → ModelEntry(provider_instance, type, provider_name).

Supports:
  • Ollama           (local, CPU/GPU, always-on)
  • vLLM - Qwen3     (GPU, port 8001, optional)
  • vLLM - Llama3.2  (GPU, port 8002, optional)
  • vLLM - Mistral7B (GPU, port 8003, optional)
  • OpenAI-compat    (LM Studio or any /v1 server, optional)

Each vLLM endpoint is registered only if its *_enabled flag is set AND the
endpoint is reachable at startup.  Missing endpoints are logged and skipped
so the API starts cleanly even when the GPU containers are down.
"""
import logging
from dataclasses import dataclass, field
from typing import Any
from config import settings
from generation.providers.ollama import OllamaProvider
from generation.providers.openai_compat import OpenAICompatProvider

log = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    provider_name: str    # "ollama" | "vllm-qwen" | "vllm-llama" | "vllm-mistral" | "openai_compat"
    type: str             # "llm" | "embed"
    provider: Any         # OllamaProvider | OpenAICompatProvider
    display_name: str = field(default="")


_registry: dict[str, ModelEntry] = {}
_default_llm_model: str = ""
_default_embed_model: str = ""
_embed_provider: Any = None


# ── vLLM endpoint descriptors ────────────────────────────────────────────────
# Each tuple: (label, enabled, base_url, api_key, configured_model_id)
def _vllm_endpoints():
    return [
        ("qwen",    settings.vllm_qwen_enabled,    settings.vllm_qwen_base_url,    settings.vllm_qwen_api_key,    settings.vllm_qwen_model),
        ("llama",   settings.vllm_llama_enabled,   settings.vllm_llama_base_url,   settings.vllm_llama_api_key,   settings.vllm_llama_model),
        ("mistral", settings.vllm_mistral_enabled, settings.vllm_mistral_base_url, settings.vllm_mistral_api_key, settings.vllm_mistral_model),
    ]


_DISPLAY_NAMES: dict[str, str] = {
    # vLLM models
    "qwen3:4b":        "🐉 Qwen3 · 4B",
    "qwen3:8b":        "🐉 Qwen3 · 8B",
    "qwen2.5:7b":      "🐉 Qwen2.5 · 7B",
    "llama3.2:3b":     "🦙 Llama 3.2 · 3B",
    "llama3.2:1b":     "🦙 Llama 3.2 · 1B",
    "llama3.1:8b":     "🦙 Llama 3.1 · 8B",
    "mistral:7b":      "🌊 Mistral · 7B",
    "mistral:7b-v0.3": "🌊 Mistral · 7B v0.3",
    # Ollama models
    "llama3.1:8b":     "🦙 Llama 3.1 · 8B",
    "llama3.2:3b":     "🦙 Llama 3.2 · 3B",
}


def _display(model_id: str) -> str:
    return _DISPLAY_NAMES.get(model_id, model_id)


def build_registry():
    global _registry, _default_llm_model, _default_embed_model, _embed_provider

    _registry = {}

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama = OllamaProvider(settings.ollama_base_url)
    for mid in ollama.list_models():
        _registry[mid] = ModelEntry(mid, "ollama", "llm", ollama, _display(mid))

    # Always register configured embed model (Ollama)
    _registry[settings.embed_model] = ModelEntry(
        settings.embed_model, "ollama", "embed", ollama, "🧲 " + settings.embed_model
    )
    _default_embed_model = settings.embed_model
    _embed_provider = ollama

    _default_llm_model = settings.llm_model
    if settings.llm_model not in _registry:
        _registry[settings.llm_model] = ModelEntry(
            settings.llm_model, "ollama", "llm", ollama, _display(settings.llm_model)
        )

    # ── vLLM endpoints (one per model family) ────────────────────────────────
    for label, enabled, base_url, api_key, cfg_model in _vllm_endpoints():
        if not enabled:
            continue
        prov_name = f"vllm-{label}"
        try:
            provider = OpenAICompatProvider(base_url, api_key)
            discovered = provider.list_models()
            if discovered:
                for mid in discovered:
                    _registry[mid] = ModelEntry(mid, prov_name, "llm", provider, _display(mid))
                log.info("Registered %d model(s) from %s (%s)", len(discovered), prov_name, base_url)
            else:
                # vLLM running but /v1/models returned empty — fall back to configured name
                _registry[cfg_model] = ModelEntry(cfg_model, prov_name, "llm", provider, _display(cfg_model))
                log.info("Registered configured model %s from %s (no /v1/models list)", cfg_model, prov_name)
        except Exception as exc:
            log.warning("vLLM-%s endpoint %s unreachable (%s) — skipping", label, base_url, exc)

    # ── Generic OpenAI-compat (LM Studio etc.) ────────────────────────────────
    if settings.openai_compat_enabled and settings.openai_compat_default_model:
        try:
            oc = OpenAICompatProvider(settings.openai_compat_base_url, settings.openai_compat_api_key)
            for mid in oc.list_models():
                _registry[mid] = ModelEntry(mid, "openai_compat", "llm", oc, mid)
            if settings.openai_compat_default_model not in _registry:
                _registry[settings.openai_compat_default_model] = ModelEntry(
                    settings.openai_compat_default_model, "openai_compat", "llm", oc,
                    settings.openai_compat_default_model,
                )
        except Exception as exc:
            log.warning("OpenAI-compat endpoint unreachable (%s) — skipping", exc)

    log.info("Model registry built: %d entries", len(_registry))


def get_llm_entry(model_id: str | None = None) -> ModelEntry:
    if not _registry:
        build_registry()
    mid = model_id or _default_llm_model
    if mid not in _registry:
        raise KeyError(f"Model '{mid}' not in registry. Available: {list(_registry)}")
    return _registry[mid]


def get_embed_provider():
    if _embed_provider is None:
        build_registry()
    return _embed_provider, _default_embed_model


def list_models() -> list[dict]:
    return [
        {
            "model_id":     e.model_id,
            "display_name": e.display_name,
            "provider":     e.provider_name,
            "type":         e.type,
        }
        for e in _registry.values()
    ]
