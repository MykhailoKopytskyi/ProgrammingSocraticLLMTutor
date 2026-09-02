from __future__ import annotations

import argparse
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from app.agents.offline.offline_student_agent import StudentVariant
from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.code_runner import DockerCodeRunner
from app.common.models import BenchmarkCase
from dotenv import load_dotenv
from openai import OpenAI

from app.evaluation.backends import ModalBackend, OpenAIBackend, StructuredModelBackend
from app.evaluation.generation.session import EvaluationSession, EvaluationStore
from app.evaluation.generation.student import ControlledStudentSystem, StudentProfileStore
from app.evaluation.generation.tutor import (
    GPT4oTutorSystem,
    SocraticRepairTutorSystem,
    TutorSystem,
)
from app.evaluation.models import EvaluationSessionResult


class GenerateTestDialoguesCLI:
    DEFAULT_WORKERS = 20

    def __init__(self):
        self.root = Path(__file__).resolve().parents[3]

    def run(self) -> None:
        args = self._arguments()
        load_dotenv(self.root / ".env")

        if not DockerCodeRunner.is_available():
            raise RuntimeError("Docker is required. Start Docker Desktop first.")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180.0)
        profile_store = StudentProfileStore(
            path=args.output_dir / "student_profiles.jsonl",
            backend=OpenAIBackend(
                client=client,
                model=os.getenv(
                    "EVAL_STUDENT_PROFILE_MODEL", os.environ["STUDENT_PROFILE_MODEL"]
                ),
                reasoning_effort="minimal",
            ),
        )
        student_generator = OpenAIBackend(
            client=client,
            model=os.getenv("EVAL_STUDENT_MODEL", os.environ["STUDENT_LLM_MODEL"]),
            reasoning_effort="low",
        )
        student_verifier = OpenAIBackend(
            client=client,
            model=os.getenv(
                "EVAL_STUDENT_VERIFIER_MODEL",
                os.environ["TURN_VERIFIER_LLM_MODEL"],
            ),
            reasoning_effort="medium",
        )
        tutor_backend = self._tutor_backend(args.tutor, client)

        cases = BenchmarkCaseStore(args.cases).load()
        if args.limit is not None:
            cases = cases[: args.limit]
        if not cases:
            raise RuntimeError(f"No evaluation cases found in {args.cases}")

        output_path = args.output_dir / f"{args.tutor}.jsonl"
        store = EvaluationStore(output_path)
        completed = store.completed_session_ids()
        variants = self._variants(args.variants)
        sessions = self._pending_sessions(
            cases=cases,
            variants=variants,
            completed=completed,
            tutor_name=args.tutor,
            tutor_backend=tutor_backend,
            student_generator=student_generator,
            student_verifier=student_verifier,
            profile_store=profile_store,
            student_attempts=args.student_attempts,
            student_state_seed=args.student_state_seed,
            max_rounds=args.max_rounds,
        )

        if not sessions:
            print("All requested evaluation dialogues are already generated.")
            print(f"Saved: {output_path}")
            return

        print(
            f"Generating {len(sessions)} dialogue(s) with "
            f"{args.workers} worker(s)."
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[Future[EvaluationSessionResult], tuple[int, EvaluationSession]] = {
                executor.submit(session.run): (index, session)
                for index, session in sessions
            }

            for future in as_completed(futures):
                index, session = futures[future]
                result = future.result()

                # Keep persistent evaluation-result writes on the main thread.
                store.append(result)
                completed.add(result.session_id)
                print(
                    f"[{index}/{len(cases) * len(variants)}] "
                    f"{result.case_id} | {result.student_variant.value} | {args.tutor} | "
                    f"rounds={len(result.rounds)} solved={result.solved} "
                    f"termination={result.termination_reason}"
                )

        print(f"Saved: {output_path}")

    def _pending_sessions(
        self,
        *,
        cases: list[BenchmarkCase],
        variants: list[StudentVariant],
        completed: set[str],
        tutor_name: str,
        tutor_backend: StructuredModelBackend,
        student_generator: StructuredModelBackend,
        student_verifier: StructuredModelBackend,
        profile_store: StudentProfileStore,
        student_attempts: int,
        student_state_seed: int,
        max_rounds: int,
    ) -> list[tuple[int, EvaluationSession]]:
        sessions: list[tuple[int, EvaluationSession]] = []
        total = len(cases) * len(variants)
        index = 0

        for case in cases:
            for variant in variants:
                index += 1
                session = EvaluationSession(
                    case=case,
                    student=ControlledStudentSystem(
                        generator=student_generator,
                        verifier=student_verifier,
                        profile_store=profile_store,
                        code_runner=DockerCodeRunner(),
                        max_attempts=student_attempts,
                    ),
                    tutor=self._tutor(tutor_name, tutor_backend),
                    student_variant=variant,
                    student_state_seed=student_state_seed,
                    max_rounds=max_rounds,
                )

                if session.session_id in completed:
                    print(f"[{index}/{total}] skip {session.session_id}")
                    continue

                sessions.append((index, session))

        return sessions

    def _tutor_backend(
        self,
        name: str,
        client: OpenAI,
    ) -> StructuredModelBackend:
        if name == "socraticrepair":
            return ModalBackend()

        return OpenAIBackend(
            client=client,
            model=os.getenv("EVAL_GPT4O_MODEL", "gpt-4o-2024-11-20"),
            temperature=0.0,
        )

    def _tutor(
        self,
        name: str,
        backend: StructuredModelBackend,
    ) -> TutorSystem:
        if name == "socraticrepair":
            return SocraticRepairTutorSystem(
                backend=backend,
                project_root=self.root,
            )

        return GPT4oTutorSystem(backend=backend)

    def _arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--tutor", choices=["socraticrepair", "gpt4o"], required=True
        )
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
            default=self.root / "data" / "evaluation" / "test_dialogues",
        )
        parser.add_argument("--variants", default="receptive,uncertain,persistent")
        parser.add_argument("--student-state-seed", type=int, default=0)
        parser.add_argument("--student-attempts", type=int, default=3)
        parser.add_argument("--max-rounds", type=int, default=14)
        parser.add_argument("--workers", type=int, default=self.DEFAULT_WORKERS)
        parser.add_argument("--limit", type=int, default=None)
        args = parser.parse_args()
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be at least 1")
        if args.student_attempts < 1:
            parser.error("--student-attempts must be at least 1")
        if args.max_rounds < 1:
            parser.error("--max-rounds must be at least 1")
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        return args

    @staticmethod
    def _variants(value: str) -> list[StudentVariant]:
        variants: list[StudentVariant] = []
        seen: set[StudentVariant] = set()
        for name in value.split(","):
            stripped = name.strip()
            if not stripped:
                continue
            variant = StudentVariant(stripped.lower())
            if variant not in seen:
                variants.append(variant)
                seen.add(variant)
        if not variants:
            raise ValueError("At least one Student variant is required")
        return variants


if __name__ == "__main__":
    GenerateTestDialoguesCLI().run()
