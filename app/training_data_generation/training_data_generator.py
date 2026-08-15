from __future__ import annotations

from ..agents.agent import AgentResponseError
from ..agents.offline.offline_student_agent import StudentVariant
from ..agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from ..common.benchmark_case_store import BenchmarkCaseStore
from .dialogue_generation import DialogueGenerationError, DialogueGenerator
from .dialogue_store import DialogueStore
from .plan_generation import PlanGenerationError, PlanGenerator


class TrainingDataGenerator:
    def __init__(
        self,
        *,
        case_store: BenchmarkCaseStore,
        plan_generator: PlanGenerator,
        profile_agent: OfflineStudentProfileAgent,
        dialogue_generator: DialogueGenerator,
        dialogue_store: DialogueStore,
        limit: int | None = None,
    ):
        self._case_store = case_store
        self._plan_generator = plan_generator
        self._profile_agent = profile_agent
        self._dialogue_generator = dialogue_generator
        self._dialogue_store = dialogue_store
        self._limit = limit

    def generate(self) -> None:
        cases = self._case_store.load()

        if self._limit is not None:
            cases = cases[: self._limit]

        existing_dialogues = self._dialogue_store.load()
        completed = set()  # set of completed dialogue vase ids
        contexts = {}  # dictionary map case_id -> (verified plan, student profile) because those are generated per BenchmarkCase (and for each case we have 3 variants of students)

        for dialogue in existing_dialogues:
            completed.add(dialogue.dialogue_id)

            if dialogue.case_id not in contexts:
                contexts[dialogue.case_id] = (
                    dialogue.verified_plan,
                    dialogue.student_profile,
                )

        variants = (
            StudentVariant.RECEPTIVE,
            StudentVariant.PERSISTENT,
            StudentVariant.UNCERTAIN,
        )

        for index, case in enumerate(cases, start=1):
            missing_variants = []
            for variant in variants:
                dialogue_id = f"{case.case_id}__{variant.value}"

                if dialogue_id not in completed:
                    missing_variants.append(variant)

            if not missing_variants:
                continue

            context = contexts.get(case.case_id)
            # If none of dialogues have been generated for this case, then we first need to generate plan and student profile
            if context is None:
                print(f"[case {index}/{len(cases)}] preparing {case.case_id}")

                try:
                    verified_plan = self._plan_generator.generate(case)
                    profile = self._profile_agent.generate(case)
                except (PlanGenerationError, AgentResponseError) as error:
                    print(f"FAILED: {type(error).__name__}: {error}")
                    continue

                contexts[case.case_id] = (
                    verified_plan,
                    profile,
                )
            else:
                verified_plan, profile = context

            for variant in missing_variants:
                dialogue_id = f"{case.case_id}__{variant.value}"
                print(f"[dialogue {index}/{len(cases)}] {dialogue_id}")

                try:
                    dialogue = self._dialogue_generator.generate(
                        case=case,
                        verified_plan=verified_plan,
                        profile=profile,
                        variant=variant,
                    )
                except (DialogueGenerationError, AgentResponseError) as error:
                    print(f"FAILED: {type(error).__name__}: {error}")
                    continue

                self._dialogue_store.append(dialogue)
                completed.add(dialogue_id)
                print("accepted")
