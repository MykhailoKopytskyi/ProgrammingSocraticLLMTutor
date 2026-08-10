from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.offline.offline_dialogue_verifier_agent import OfflineDialogueVerifierAgent
from app.agents.offline.offline_plan_verifier_agent import OfflinePlanVerifierAgent
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from app.agents.offline.offline_turn_verifier_agent import OfflineTurnVerifierAgent
from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.code_runner import DockerCodeRunner
from app.training_data_generation.dialogue_pipeline import DialogueGenerationPipeline
from app.training_data_generation.generator import TrainingDataGenerator
from app.training_data_generation.planning_pipeline import PlanningPipeline
from app.training_data_generation.stores import DialogueStore, PreparedDialogueCaseStore


class TrainingDataGenerationApp:
    def __init__(self):
        load_dotenv()
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        planning_pipeline = PlanningPipeline(
            planner=OfflinePlannerAgent(
                llm=client,
                model=os.environ["PLANNER_LLM_MODEL"],
            ),
            verifier=OfflinePlanVerifierAgent(
                llm=client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
        )
        dialogue_pipeline = DialogueGenerationPipeline(
            turn_verifier=OfflineTurnVerifierAgent(
                llm=client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
            dialogue_verifier=OfflineDialogueVerifierAgent(
                llm=client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
            code_runner=DockerCodeRunner(),
        )

        self._generator = TrainingDataGenerator(
            llm=client,
            student_model=os.environ["STUDENT_LLM_MODEL"],
            tutor_model=os.environ["TUTOR_LLM_MODEL"],
            case_stores=(BenchmarkCaseStore(Path("data/processed/multi_debug.jsonl")),),
            planning_pipeline=planning_pipeline,
            profile_agent=OfflineStudentProfileAgent(
                llm=client,
                model=os.environ["DATA_LLM_MODEL"],
            ),
            dialogue_pipeline=dialogue_pipeline,
            prepared_store=PreparedDialogueCaseStore(
                Path("data/generated/prepared_dialogue_cases.jsonl")
            ),
            dialogue_store=DialogueStore(Path("data/generated/dialogues.jsonl")),
            limit=self._max_cases(),
        )

    def run(self) -> None:
        self._generator.generate()

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv("MAX_CASES", "").strip()
        return int(value) if value else None


if __name__ == "__main__":
    TrainingDataGenerationApp().run()
