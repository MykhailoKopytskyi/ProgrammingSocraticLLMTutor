from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ..agents.agent import AgentResponseError
from ..agents.offline.offline_student_agent import StudentVariant
from ..agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from ..common.benchmark_case_store import BenchmarkCaseStore
from .dialogue_generation import DialogueGenerationError, DialogueGenerator
from .dialogue_store import DialogueStore
from .plan_generation import PlanGenerationError, PlanGenerator

MAX_WORKERS = 20


class TrainingDataGenerator:
    def __init__(
        self,
        *,
        case_store: BenchmarkCaseStore,
        plan_generator: PlanGenerator,
        profile_agent: OfflineStudentProfileAgent,
        dialogue_generator: DialogueGenerator,
        dialogue_store: DialogueStore,
        failed_path: str | Path,
        limit: int | None = None,
    ):
        self._case_store = case_store
        self._plan_generator = plan_generator
        self._profile_agent = profile_agent
        self._dialogue_generator = dialogue_generator
        self._dialogue_store = dialogue_store
        self._failed_path = Path(failed_path)
        self._limit = limit
        self._state_lock = Lock()
        self._failed_file_lock = Lock()

    def generate(self) -> None:
        cases = self._case_store.load()

        if self._limit is not None:
            cases = cases[: self._limit]

        existing_dialogues = self._dialogue_store.load()

        completed = set()
        contexts = {}

        for dialogue in existing_dialogues:
            completed.add(dialogue.dialogue_id)

            if dialogue.case_id not in contexts:
                contexts[dialogue.case_id] = (
                    dialogue.verified_plan,
                    dialogue.student_profile,
                )

        failed = set()

        if self._failed_path.exists():
            with self._failed_path.open("r", encoding="utf-8") as file:
                for line in file:
                    dialogue_id = line.strip()

                    if dialogue_id:
                        failed.add(dialogue_id)

        variants = (
            StudentVariant.RECEPTIVE,
            StudentVariant.UNCERTAIN,
            StudentVariant.PERSISTENT,
        )

        for variant in variants:
            print(f"\n=== GENERATING {variant.value.upper()} DIALOGUES ===\n")

            pending = []

            for index, case in enumerate(cases, start=1):
                dialogue_id = f"{case.case_id}__{variant.value}"

                if dialogue_id not in completed and dialogue_id not in failed:
                    pending.append((index, case))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        self._generate_one,
                        case=case,
                        index=index,
                        total=len(cases),
                        variant=variant,
                        contexts=contexts,
                        completed=completed,
                        failed=failed,
                    )
                    for index, case in pending
                ]

                for future in as_completed(futures):
                    future.result()

    def _generate_one(
        self,
        *,
        case,
        index: int,
        total: int,
        variant: StudentVariant,
        contexts: dict,
        completed: set[str],
        failed: set[str],
    ) -> None:
        dialogue_id = f"{case.case_id}__{variant.value}"

        with self._state_lock:
            context = contexts.get(case.case_id)

        if context is None:
            print(f"[case {index}/{total}] preparing {case.case_id}")
            try:
                verified_plan = self._plan_generator.generate(case)
                profile = self._profile_agent.generate(case)
            except (
                PlanGenerationError,
                AgentResponseError,
            ) as error:
                print(f"FAILED: {type(error).__name__}: {error}")
                self._save_failed(dialogue_id, failed)
                return

            context = (verified_plan, profile)
            with self._state_lock:
                contexts[case.case_id] = context
        verified_plan, profile = context
        print(f"[{variant.value} {index}/{total}] {dialogue_id}")
        try:
            dialogue = self._dialogue_generator.generate(
                case=case,
                verified_plan=verified_plan,
                profile=profile,
                variant=variant,
            )
        except (
            DialogueGenerationError,
            AgentResponseError,
        ) as error:
            print(f"FAILED: {type(error).__name__}: {error}")
            self._save_failed(dialogue_id, failed)
            return

        self._dialogue_store.append(dialogue)
        with self._state_lock:
            completed.add(dialogue_id)
        print(f"accepted: {dialogue_id}")

    def _save_failed(
        self,
        dialogue_id: str,
        failed: set[str],
    ) -> None:
        with self._state_lock:
            if dialogue_id in failed:
                return

            failed.add(dialogue_id)

        try:
            with self._failed_file_lock:
                self._failed_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with self._failed_path.open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(f"{dialogue_id}\n")
        except Exception:
            with self._state_lock:
                failed.discard(dialogue_id)

            raise
