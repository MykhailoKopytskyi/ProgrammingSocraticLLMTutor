from __future__ import annotations

from pathlib import Path

from .dialogue_models import GeneratedDialogue


class DialogueStore:
    def __init__(
        self,
        path: str | Path,
    ):
        self._path = Path(path)

    def append(
        self,
        dialogue: GeneratedDialogue,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("a", encoding="utf-8") as file:
            file.write(dialogue.model_dump_json())
            file.write("\n")

    def load(self) -> list[GeneratedDialogue]:
        if not self._path.exists():
            return []

        with self._path.open("r", encoding="utf-8") as file:
            return [
                GeneratedDialogue.model_validate_json(line)
                for line in file
                if line.strip()
            ]

    def completed_case_ids(self) -> set[str]:
        return {dialogue.case_id for dialogue in self.load()}
