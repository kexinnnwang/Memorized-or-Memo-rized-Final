from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class Document:
    doc_id: str         
    text: str             
    source: str        
    date: datetime | None = None 

class DocumentLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

    def load(self) -> list[Document]:
        files = sorted(self.data_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")

        docs: list[Document] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue  # skip empty files silently
            docs.append(Document(doc_id=path.stem, text=text, source=str(path)))
        return docs

