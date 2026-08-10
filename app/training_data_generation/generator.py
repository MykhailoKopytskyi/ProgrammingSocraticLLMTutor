from __future__ import annotations

from typing import Any

from ..agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from ..agents.student_agent import StudentAgent
from ..agents.tutor_agent import TutorAgent
from ..common.benchmark_case_store import BenchmarkCaseStore
from .dialogue_pipeline import DialogueGenerationPipeline
from .models import PreparedDialogueCase
from .planning_pipeline import PlanningPipeline
from .stores import DialogueStore, PreparedDialogueCaseStore


class TrainingDataGenerator:
    """Prepares fixed case conditions once, then generates dialogues from them."""

    def __init__(
        self,
        *,
        llm: Any,
        student_model: str,
        tutor_model: str,
        case_stores: tuple[BenchmarkCaseStore, ...],
        planning_pipeline: PlanningPipeline,
        profile_agent: OfflineStudentProfileAgent,
        dialogue_pipeline: DialogueGenerationPipeline,
        prepared_store: PreparedDialogueCaseStore,
        dialogue_store: DialogueStore,
        limit: int | None = None,
    ):
        self._llm = llm
        self._student_model = student_model
        self._tutor_model = tutor_model
        self._case_stores = case_stores
        self._planning_pipeline = planning_pipeline
        self._profile_agent = profile_agent
        self._dialogue_pipeline = dialogue_pipeline
        self._prepared_store = prepared_store
        self._dialogue_store = dialogue_store
        self._limit = limit

    def generate(self) -> None:
        cases = []
        for store in self._case_stores:
            for case in store.load():
                cases.append(case)

        if self._limit is not None:
            cases = cases[: self._limit]

        prepared = self._prepared_store.by_case_id()
        for index, case in enumerate(cases, start=1):
            if case.case_id in prepared:
                continue

            print(f"[prepare {index}/{len(cases)}] {case.case_id}")
            try:
                verified_plan = self._planning_pipeline.generate(case)
                profile = self._profile_agent.generate(case)
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}")
                continue

            prepared_case = PreparedDialogueCase(
                case_id=case.case_id,
                source=case.source,
                student_profile=profile,
                planner_output=verified_plan.output,
            )
            self._prepared_store.append(prepared_case)
            prepared[case.case_id] = prepared_case
            print("prepared")

        completed_dialogues = self._dialogue_store.completed_case_ids()
        for index, case in enumerate(cases, start=1):
            if case.case_id in completed_dialogues:
                continue

            prepared_case = prepared.get(case.case_id)
            if prepared_case is None:
                continue

            print(f"[dialogue {index}/{len(cases)}] {case.case_id}")
            student = StudentAgent(
                llm=self._llm,
                model=self._student_model,
                case=case,
                profile=prepared_case.student_profile,
            )
            tutor = TutorAgent(
                llm=self._llm,
                model=self._tutor_model,
                case=case,
                plan=prepared_case.planner_output.plan,
            )

            try:
                dialogue = self._dialogue_pipeline.generate(
                    case=case,
                    planner_output=prepared_case.planner_output,
                    profile=prepared_case.student_profile,
                    student_agent=student,
                    tutor_agent=tutor,
                )
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}")
                continue

            self._dialogue_store.append(dialogue)
            print("accepted")
