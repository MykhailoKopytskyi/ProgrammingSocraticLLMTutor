from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_dialogue_verifier_agent import (
    OfflineDialogueVerifierAgent,
)
from app.agents.offline.offline_plan_verifier_agent import OfflinePlanVerifierAgent
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.agents.offline.offline_student_profile_agent import OfflineStudentProfileAgent
from app.agents.offline.offline_student_turn_verifier_agent import (
    OfflineStudentTurnVerifierAgent,
)
from app.agents.offline.offline_tutor_turn_verifier_agent import (
    OfflineTutorTurnVerifierAgent,
)
from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.code_runner import DockerCodeRunner
from app.training_data_generation.dialogue_generation import DialogueGenerator
from app.training_data_generation.dialogue_store import DialogueStore
from app.training_data_generation.plan_generation import PlanGenerator
from app.training_data_generation.training_data_generator import TrainingDataGenerator
from dotenv import load_dotenv
from openai import OpenAI


def max_cases():
    value = os.getenv("MAX_CASES", "").strip()

    if not value:
        return None

    return int(value)


def student_state_seed():
    return int(os.getenv("STUDENT_STATE_SEED", "0"))


def main():
    load_dotenv()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180.0)

    plan_generator = PlanGenerator(
        planner=OfflinePlannerAgent(
            llm=client,
            model=os.environ["PLANNER_LLM_MODEL"],
        ),
        verifier=OfflinePlanVerifierAgent(
            llm=client,
            model=os.environ["VERIFIER_LLM_MODEL"],
        ),
    )

    dialogue_generator = DialogueGenerator(
        llm=client,
        student_model=os.environ["STUDENT_LLM_MODEL"],
        tutor_model=os.environ["TUTOR_LLM_MODEL"],
        student_turn_verifier=OfflineStudentTurnVerifierAgent(
            llm=client,
            model=os.environ["VERIFIER_LLM_MODEL"],
        ),
        tutor_turn_verifier=OfflineTutorTurnVerifierAgent(
            llm=client,
            model=os.environ["VERIFIER_LLM_MODEL"],
        ),
        dialogue_verifier=OfflineDialogueVerifierAgent(
            llm=client,
            model=os.environ["VERIFIER_LLM_MODEL"],
        ),
        code_runner=DockerCodeRunner(),
        student_state_seed=student_state_seed(),
    )

    generator = TrainingDataGenerator(
        case_store=BenchmarkCaseStore(
            Path("data/processed/splits/train/benchmark_cases.jsonl")
        ),
        plan_generator=plan_generator,
        profile_agent=OfflineStudentProfileAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        ),
        dialogue_generator=dialogue_generator,
        dialogue_store=DialogueStore(Path("data/generated/dialogues.jsonl")),
        limit=max_cases(),
    )

    generator.generate()


if __name__ == "__main__":
    main()
