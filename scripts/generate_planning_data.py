from __future__ import annotations

import os
from pathlib import Path

from app.datasets.multi_debug import load_multi_debug
from dotenv import load_dotenv
from openai import OpenAI

from app.agents.offline.offline_plan_verifier_agent import (
    OfflinePlanVerifierAgent,
)
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.training.plan_pipeline import (
    build_planning_record,
    generate_verified_plan,
)
from app.training.writer import write_jsonl

DATASET_ROOT = Path("data/multi_debug")
OUTPUT_PATH = Path("data/generated/planning.jsonl")


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Required environment variable {name} is not configured")

    return value


def main() -> None:
    load_dotenv()

    client = OpenAI(api_key=require_environment_variable("OPENAI_API_KEY"))

    teacher = OfflinePlannerAgent(
        llm=client,
        model=require_environment_variable("TEACHER_PLANNER_LLM_MODEL"),
    )

    verifier = OfflinePlanVerifierAgent(
        llm=client,
        model=require_environment_variable("PLAN_VERIFIER_LLM_MODEL"),
    )

    max_cases_text = os.getenv("MAX_CASES", "").strip()
    max_cases = int(max_cases_text) if max_cases_text else None

    cases = load_multi_debug(
        DATASET_ROOT,
        limit=max_cases,
    )

    records: list[dict[str, object]] = []

    print(f"Loaded {len(cases)} MULTI_DEBUG cases")

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Generating plan for {case.case_id}...")

        try:
            result = generate_verified_plan(
                case=case,
                teacher_planner=teacher,
                plan_verifier=verifier,
                observed_failure=case.observed_failure,
                max_attempts=3,
            )

            record = build_planning_record(
                case=case,
                verified_plan=result,
                observed_failure=case.observed_failure,
            )

            records.append(record)

            print(f"Accepted after {result.attempts} attempt(s)")

        except Exception as error:
            print(f"Failed {case.case_id}: {type(error).__name__}: {error}")

    write_jsonl(
        OUTPUT_PATH,
        records,
    )

    print(f"Wrote {len(records)} planning records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
