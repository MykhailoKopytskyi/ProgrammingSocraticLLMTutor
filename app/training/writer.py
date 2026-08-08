from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlRecordStore:
    """Minimal JSONL persistence for generic dictionary records."""

    def __init__(
        self,
        path: str | Path,
    ):
        self._path = Path(path)

    def append(
        self,
        record: dict[str, Any],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
