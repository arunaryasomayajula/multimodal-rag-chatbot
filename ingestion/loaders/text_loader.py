from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawChunk:
    content: str
    metadata: dict = field(default_factory=dict)


def load_text(path: str | Path) -> list[RawChunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return [RawChunk(content=text, metadata={"source_file": path.name, "modality": "text"})]
