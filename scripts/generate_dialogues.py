from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_dialogue_verifier_agent import (
    OfflineDialogueVerifierAgent,
)
from app.agents.offline.offline_plan_verifier_agent import OfflinePlanVerifierAgent
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from app.agents.offline.offline_turn_verifier_agent import OfflineTurnVerifierAgent
from app.agents.student_agent import StudentAgent
from app.agents.tutor_agent import TutorAgent
from app.datasets.store import BenchmarkCaseStore
from app.execution.code_runner import DockerCodeRunner
from app.training.dialogue_pipeline import DialogueGenerationPipeline
from app.training.dialogue_store import DialogueStore
from app.training.plan_pipeline import PlanningPipeline
from dotenv import load_dotenv
from openai import OpenAI


class DialogueGenerationApp:
    def __init__(self):
        load_dotenv()

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        runner = DockerCodeRunner()

        self._planning_pipeline = PlanningPipeline(
            planner=OfflinePlannerAgent(
                llm=self._client,
                model=os.environ["PLANNER_LLM_MODEL"],
            ),
            verifier=OfflinePlanVerifierAgent(
                llm=self._client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
        )
        self._profile_agent = OfflineStudentProfileAgent(
            llm=self._client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        self._dialogue_pipeline = DialogueGenerationPipeline(
            turn_verifier=OfflineTurnVerifierAgent(
                llm=self._client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
            dialogue_verifier=OfflineDialogueVerifierAgent(
                llm=self._client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
            code_runner=runner,
        )
        self._cases = BenchmarkCaseStore(
            Path(
                os.getenv(
                    "BENCHMARK_DATA_PATH",
                    "data/processed/multi_debug.jsonl",
                )
            )
        )
        self._dialogues = DialogueStore(
            Path(
                os.getenv(
                    "DIALOGUE_OUTPUT_PATH",
                    "data/generated/dialogues.jsonl",
                )
            )
        )

    def run(self) -> None:
        cases = self._cases.load()
        limit = self._max_cases()

        if limit is not None:
            cases = cases[:limit]

        completed = self._dialogues.completed_case_ids()

        for index, case in enumerate(cases, start=1):
            if case.case_id in completed:
                continue

            print(f"[{index}/{len(cases)}] {case.case_id}")

            try:
                verified_plan = self._planning_pipeline.generate(case)
                profile = self._profile_agent.generate(case)

                student = StudentAgent(
                    llm=self._client,
                    model=os.environ["STUDENT_LLM_MODEL"],
                    case=case,
                    profile=profile,
                )
                tutor = TutorAgent(
                    llm=self._client,
                    model=os.environ["TUTOR_LLM_MODEL"],
                    case=case,
                    plan=verified_plan.output.plan,
                )

                dialogue = self._dialogue_pipeline.generate(
                    case=case,
                    verified_plan=verified_plan,
                    profile=profile,
                    student_agent=student,
                    tutor_agent=tutor,
                )
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}")
                continue

            self._dialogues.append(dialogue)
            print("accepted")

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv("MAX_CASES", "").strip()
        return int(value) if value else None


if __name__ == "__main__":
    DialogueGenerationApp().run()
