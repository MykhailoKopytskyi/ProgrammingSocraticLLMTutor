from __future__ import annotations

from pathlib import Path

from .models import GeneratedDialogue


class DialogueStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    def append(self, dialogue: GeneratedDialogue) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("a", encoding="utf-8") as file:
            file.write(dialogue.model_dump_json())
            file.write("\n")

    def load(self) -> list[GeneratedDialogue]:
        if not self._path.exists():
            return []

        dialogues = []

        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dialogues.append(GeneratedDialogue.model_validate_json(line))

        return dialogues
