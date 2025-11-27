import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import uuid


@dataclass
class EmbeddingEntry:
    id: str
    text: str
    embedding: np.ndarray


class EmbeddingStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingStore, cls).__new__(cls)
            cls._instance.entries = []
        return cls._instance

    def add(self, text: str, embedding: np.ndarray) -> str:
        entry_id = str(uuid.uuid4())
        entry = EmbeddingEntry(id=entry_id, text=text, embedding=embedding)
        self.entries.append(entry)
        return entry_id

    def get_all(self) -> List[EmbeddingEntry]:
        return self.entries

    def clear(self):
        self.entries = []

    def count(self) -> int:
        return len(self.entries)
    
    def __contains__(self, text):
        return text in [item.text for item in self.get_all()]
