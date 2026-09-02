from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from app.common.code_runner import TestRunResult
from app.common.models import BenchmarkCase, LearnerState

from ..backends import StructuredModelBackend
from ..conversation import ConversationRenderer
from ..models import (
    BaselineTutorOutput,
    EvaluationRound,
    EvaluationStudentTurn,
    EvaluationTutorTurn,
    MasteryState,
    RuntimePlan,
    SocraticRepairTutorOutput,
    StepStatus,
)

SOCRATIC_DISCLOSURE_POLICY = """
Socratic disclosure policy:

Your goal is to help the Student discover the repair through their own reasoning.

- Do not immediately reveal the bug location, exact repair, corrected expression,
  corrected line, or corrected code merely because it is available in the private
  context or pedagogical plan.
- Prefer questions, traces, counterexamples, and progressively stronger hints.
- Treat expected_answer and trusted_corrected_code as private guidance for evaluating
  and steering the Student, not as content that should automatically be revealed.
- When the Student is incorrect, address the misconception and scaffold toward the
  reasoning needed for the repair rather than simply supplying the repair.
- When the Student is confused, increase assistance gradually: simplify the question,
  provide a concrete trace, or give a partial hint.
- Reveal an exact repair only after the Student has demonstrated the relevant reasoning,
  or after repeated scaffolding has failed and stronger assistance is appropriate.
- Even when stronger assistance is needed, explain why the repair follows from the
  reasoning developed in the conversation.
""".strip()


class TutorSystem(ABC):
    @property
    @abstractmethod
    def system_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def start_case(self, case: BenchmarkCase) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_turn(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        student_turn: EvaluationStudentTurn,
        latest_code_execution: TestRunResult | None,
        current_working_code: str,
    ) -> EvaluationTutorTurn:
        raise NotImplementedError

    def mastery_state(self) -> MasteryState | None:
        return None

    def metadata(self) -> dict:
        return {}


class SocraticRepairTutorSystem(TutorSystem):
    NONPOSITIVE_STATES = {
        LearnerState.INCORRECT,
        LearnerState.QUESTION,
        LearnerState.CONFUSION,
        LearnerState.IRRELEVANT,
    }

    def __init__(
        self,
        *,
        backend: StructuredModelBackend,
        project_root: str | Path,
    ):
        self.backend = backend
        root = Path(project_root)
        self.plan_system_prompt = self._read_system_prompt(
            root / "data" / "prepared_training" / "planner_examples-train.jsonl"
        )
        original_tutor_prompt = self._read_system_prompt(
            root
            / "data"
            / "prepared_training"
            / "tutor_examples_step_assessments.jsonl"
        )

        self.tutor_system_prompt = (
            original_tutor_prompt + "\n\n" + SOCRATIC_DISCLOSURE_POLICY
        )
        self.case: BenchmarkCase | None = None
        self.plan: RuntimePlan | None = None
        self.plan_raw_output = ""
        self.demonstrated: set[str] = set()

    @property
    def system_id(self) -> str:
        return self.backend.backend_id

    @property
    def config_id(self) -> str:
        return f"{self.backend.config_id}:socraticrepair"

    def start_case(self, case: BenchmarkCase) -> None:
        planner_input = {
            "problem_statement": case.problem_statement,
            "buggy_code": case.buggy_code,
            "tests": case.tests,
            "observed_failure": case.observed_failure,
            "trusted_corrected_code": case.correct_code,
        }
        result = self.backend.generate(
            system_prompt=self.plan_system_prompt,
            user_prompt=json.dumps(planner_input, ensure_ascii=False, indent=2),
            output_type=RuntimePlan,
            max_output_tokens=2200,
        )

        self.case = case
        self.plan = result.parsed
        self.plan_raw_output = result.raw_text
        self.demonstrated = set()

    def mastery_state(self) -> MasteryState | None:
        if self.plan is None:
            return None

        step_ids = [step.step_id for step in self.plan.steps]
        demonstrated = [step_id for step_id in step_ids if step_id in self.demonstrated]
        undemonstrated = [
            step_id for step_id in step_ids if step_id not in self.demonstrated
        ]
        return MasteryState(
            demonstrated_step_ids=demonstrated,
            active_step_id=undemonstrated[0] if undemonstrated else None,
            undemonstrated_step_ids=undemonstrated,
        )

    def metadata(self) -> dict:
        if self.plan is None:
            return {}
        return {
            "plan": self.plan.model_dump(mode="json"),
            "raw_plan_output": self.plan_raw_output,
        }

    def generate_turn(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        student_turn: EvaluationStudentTurn,
        latest_code_execution: TestRunResult | None,
        current_working_code: str,
    ) -> EvaluationTutorTurn:
        if self.case is None or self.plan is None:
            raise RuntimeError("Tutor system has not been started")

        state = self.mastery_state()
        assert state is not None
        step_by_id = {step.step_id: step for step in self.plan.steps}
        active_step = (
            step_by_id[state.active_step_id].model_dump(mode="json")
            if state.active_step_id is not None
            else None
        )

        latest_student: dict = {
            "round": len(history) + 1,
            "reply": student_turn.reply,
        }
        if student_turn.proposed_code.strip():
            latest_student["proposed_code"] = student_turn.proposed_code
        if latest_code_execution is not None:
            latest_student["code_execution"] = self._execution_payload(
                latest_code_execution
            )

        user_payload = {
            "private_problem_context": {
                "problem_statement": self.case.problem_statement,
                "original_buggy_code": self.case.buggy_code,
                "tests": self.case.tests,
                "observed_failure": self.case.observed_failure,
                "trusted_corrected_code": self.case.correct_code,
            },
            "pedagogical_plan": self.plan.model_dump(mode="json"),
            "plan_state_before_latest_student_turn": {
                **state.model_dump(mode="json"),
                "active_step": active_step,
            },
            "current_working_code": current_working_code,
            "dialogue_history_before_latest_student_turn": ConversationRenderer.structured(
                history
            ),
            "latest_student_turn": latest_student,
        }
        result = self.backend.generate(
            system_prompt=self.tutor_system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
            output_type=SocraticRepairTutorOutput,
            max_output_tokens=1200,
        )
        violations = self._apply_assessments(result.parsed, state)

        return EvaluationTutorTurn(
            system_id=self.system_id,
            reply=result.parsed.reply,
            raw_output=result.raw_text,
            learner_state=result.parsed.learner_state,
            step_assessments=result.parsed.step_assessments,
            protocol_violations=violations,
        )

    def _apply_assessments(
        self,
        output: SocraticRepairTutorOutput,
        state: MasteryState,
    ) -> list[str]:
        expected = state.undemonstrated_step_ids
        actual = [assessment.step_id for assessment in output.step_assessments]
        violations: list[str] = []

        if actual != expected:
            violations.append(
                "step_assessment_ids_do_not_match_current_undemonstrated_steps"
            )
        if len(actual) != len(set(actual)):
            violations.append("duplicate_step_assessment")

        valid_ids = set(expected)
        newly_demonstrated = {
            assessment.step_id
            for assessment in output.step_assessments
            if assessment.step_id in valid_ids
            and assessment.status == StepStatus.DEMONSTRATED
        }
        if (
            output.learner_state in self.NONPOSITIVE_STATES
            and state.active_step_id in newly_demonstrated
        ):
            violations.append("nonpositive_state_demonstrated_active_step")

        self.demonstrated.update(newly_demonstrated)
        return violations

    @staticmethod
    def _read_system_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prepared training file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                for message in row["prompt_messages"]:
                    if message.get("role") == "system":
                        return message["content"]
                break
        raise RuntimeError(f"No system prompt found in {path}")

    @staticmethod
    def _execution_payload(result: TestRunResult) -> dict:
        return {
            "passed": result.passed,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "output": result.output,
        }


class GPT4oTutorSystem(TutorSystem):
    BASE_SYSTEM_PROMPT = """
    You are a Socratic Python debugging tutor. Help the Student reason toward a correct repair rather than solving the task for them.

    A private trusted corrected program may be supplied for technical grounding. Use it only to avoid giving incorrect guidance.

    Respond to the Student's latest reasoning, question, proposed code, and execution evidence. Keep the dialogue focused on the current debugging need. Prefer one focused question, trace, prediction, comparison, or small hint per turn. If the Student is wrong or confused, reduce the difficulty of the next step. If they go off topic, steer them back. Do not invent execution results.

    Return exactly one concise Tutor reply.
    """.strip()

    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + SOCRATIC_DISCLOSURE_POLICY

    def __init__(self, *, backend: StructuredModelBackend):
        self.backend = backend
        self.case: BenchmarkCase | None = None

    @property
    def system_id(self) -> str:
        return f"gpt4o_baseline__{self.backend.backend_id}"

    @property
    def config_id(self) -> str:
        return f"{self.backend.config_id}:gpt4o-baseline"

    def start_case(self, case: BenchmarkCase) -> None:
        self.case = case

    def generate_turn(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        student_turn: EvaluationStudentTurn,
        latest_code_execution: TestRunResult | None,
        current_working_code: str,
    ) -> EvaluationTutorTurn:
        if self.case is None:
            raise RuntimeError("Tutor system has not been started")

        student_text = ConversationRenderer.render_student_turn(
            student_turn, latest_code_execution
        )

        user_prompt = (
            "Runtime-visible programming case:\n"
            f"{self.case.visible_context()}\n"
            "Private trusted corrected program for technical grounding only:\n"
            f"```python\n{self.case.correct_code.rstrip()}\n```\n\n"
            "Current working code:\n"
            f"```python\n{current_working_code.rstrip()}\n```\n\n"
            "Conversation history:\n"
            f"{ConversationRenderer.render(history)}\n\n"
            "Latest Student turn:\n"
            f"{student_text}"
        )
        result = self.backend.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_type=BaselineTutorOutput,
            max_output_tokens=1000,
        )
        return EvaluationTutorTurn(
            system_id=self.system_id,
            reply=result.parsed.reply,
            raw_output=result.raw_text,
        )
