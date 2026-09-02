from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append_model(self, model: BaseModel) -> None:
        self._append_text(model.model_dump_json())

    def append_mapping(self, row: dict[str, Any]) -> None:
        self._append_text(json.dumps(row, ensure_ascii=False))

    def read_mappings(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.path}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object on line {line_number} of {self.path}"
                    )
                rows.append(row)
        return rows

    def read_models(self, model_type: type[ModelT]) -> list[ModelT]:
        if not self.path.exists():
            return []

        rows: list[ModelT] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(model_type.model_validate_json(line))
                except ValidationError as error:
                    raise ValueError(
                        f"Invalid {model_type.__name__} on line {line_number} of {self.path}"
                    ) from error
        return rows

    def _append_text(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
