# src/pipeline/llm_cache.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFileCache:
    """
    Simple file-based cache:
      key -> SHA256(key) -> JSON file stored on disk.

    Use this to:
      - avoid re-paying for identical LLM requests
      - resume after interruptions
      - speed up demo reruns
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt cache entry; treat as miss
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path_for_key(key)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
