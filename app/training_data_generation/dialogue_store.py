from __future__ import annotations

from pathlib import Path
from threading import Lock

from .models import GeneratedDialogue


class DialogueStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._write_lock = Lock()

    def append(self, dialogue: GeneratedDialogue) -> None:
        line = dialogue.model_dump_json() + "\n"

        with self._write_lock:
            self._path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self._path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(line)

    def load(self) -> list[GeneratedDialogue]:
        if not self._path.exists():
            return []

        dialogues = []

        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dialogues.append(GeneratedDialogue.model_validate_json(line))

        return dialogues
