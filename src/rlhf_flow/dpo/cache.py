import hashlib
from typing import Dict, Optional, Tuple

class ReferenceLogProbCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[float, float]] = {}

    def store(self, prompt: str, chosen: str, rejected: str, c_logp: float, r_logp: float):
        k = hashlib.sha256(f"{prompt}|{chosen}|{rejected}".encode("utf-8")).hexdigest()
        self._cache[k] = (c_logp, r_logp)

    def get(self, prompt: str, chosen: str, rejected: str) -> Optional[Tuple[float, float]]:
        k = hashlib.sha256(f"{prompt}|{chosen}|{rejected}".encode("utf-8")).hexdigest()
        return self._cache.get(k)

    def size(self) -> int:
        return len(self._cache)

    def __len__(self) -> int:
        return len(self._cache)
