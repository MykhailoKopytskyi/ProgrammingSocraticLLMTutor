from __future__ import annotations

from pathlib import Path

from .models import GeneratedDialogue, PreparedDialogueCase


class PreparedDialogueCaseStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    def append(self, case: PreparedDialogueCase) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(case.model_dump_json())
            file.write("\n")

    def load(self) -> list[PreparedDialogueCase]:
        if not self._path.exists():
            return []

        cases: list[PreparedDialogueCase] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    cases.append(PreparedDialogueCase.model_validate_json(line))
        return cases

    def by_case_id(self) -> dict[str, PreparedDialogueCase]:
        cases_by_id: dict[str, PreparedDialogueCase] = {}
        for case in self.load():
            cases_by_id[case.case_id] = case
        return cases_by_id


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

        dialogues: list[GeneratedDialogue] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    dialogues.append(GeneratedDialogue.model_validate_json(line))
        return dialogues

    def completed_case_ids(self) -> set[str]:
        case_ids: set[str] = set()
        for dialogue in self.load():
            case_ids.add(dialogue.case_id)
        return case_ids
