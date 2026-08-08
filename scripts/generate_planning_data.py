from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_plan_verifier_agent import OfflinePlanVerifierAgent
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.datasets.store import BenchmarkCaseStore
from app.training.plan_pipeline import PlanningPipeline
from app.training.writer import JsonlRecordStore
from dotenv import load_dotenv
from openai import OpenAI


class PlanningDataGenerationApp:
    """Optional planning-only export; dialogue generation uses the same pipeline."""

    def __init__(self):
        load_dotenv()

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        self._cases = BenchmarkCaseStore(
            Path(
                os.getenv(
                    "BENCHMARK_DATA_PATH",
                    "data/processed/multi_debug.jsonl",
                )
            )
        )
        self._records = JsonlRecordStore(
            Path(
                os.getenv(
                    "PLANNING_OUTPUT_PATH",
                    "data/generated/planning.jsonl",
                )
            )
        )
        self._pipeline = PlanningPipeline(
            planner=OfflinePlannerAgent(
                llm=client,
                model=os.environ["PLANNER_LLM_MODEL"],
            ),
            verifier=OfflinePlanVerifierAgent(
                llm=client,
                model=os.environ["VERIFIER_LLM_MODEL"],
            ),
        )

    def run(self) -> None:
        cases = self._cases.load()
        limit = self._max_cases()

        if limit is not None:
            cases = cases[:limit]

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id}")

            try:
                result = self._pipeline.generate(case)
            except Exception as error:
                print(f"FAILED: {type(error).__name__}: {error}")
                continue

            self._records.append(
                {
                    "task": "solve_and_plan",
                    "case_id": case.case_id,
                    "source": case.source,
                    "input": case.visible_context(),
                    "target": result.output.model_dump(mode="json"),
                    "metadata": {
                        "attempts": result.attempts,
                        "covered_bug_ids": result.verification.covered_bug_ids,
                    },
                }
            )
            print("accepted")

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv("MAX_CASES", "").strip()
        return int(value) if value else None


if __name__ == "__main__":
    PlanningDataGenerationApp().run()
