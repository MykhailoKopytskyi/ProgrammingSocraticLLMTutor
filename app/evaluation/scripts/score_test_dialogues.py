from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.evaluation.backends import OpenAIBackend
from app.evaluation.scoring.evaluator import (
    DialogueEvaluator,
)


class ScoreTestDialoguesCLI:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[3]

    def run(self) -> None:
        args = self._arguments()
        load_dotenv(self.root / ".env")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180.0)
        evaluator = DialogueEvaluator(
            backend=OpenAIBackend(
                client=client,
                model=args.judge_model,
                reasoning_effort=args.reasoning_effort,
            ),
            cases_path=args.cases,
        )

        stem = args.input.stem
        output_dir = args.output_dir / stem
        summary = evaluator.score_file(
            input_path=args.input,
            scores_path=output_dir / "turn_scores.jsonl",
            summary_path=output_dir / "summary.json",
        )
        print(json.dumps(summary, indent=2))
        print(f"Saved scores under: {output_dir}")

    def _arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument(
            "--cases",
            type=Path,
            default=self.root
            / "data"
            / "processed"
            / "splits"
            / "test"
            / "benchmark_cases.jsonl",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=self.root / "data" / "evaluation" / "scores",
        )
        parser.add_argument(
            "--judge-model",
            default=os.getenv("EVAL_JUDGE_MODEL", "gpt-5.6-terra"),
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            default="high",
        )
        return parser.parse_args()


if __name__ == "__main__":
    ScoreTestDialoguesCLI().run()
