from types import SimpleNamespace

from app.agents.offline.offline_plan_verifier_agent import (
    OfflinePlanVerifierAgent,
    PlanVerification,
)
from app.agents.offline.offline_planner_agent import OfflinePlannerAgent
from app.common.models import (
    BenchmarkCase,
    BugAnnotation,
    PedagogicalPlan,
    PlannerOutput,
    PlanStep,
)
from app.execution.code_runner import TestRunResult
from app.training.plan_pipeline import PlanningPipeline


def make_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="range-case",
        problem_statement="Return all numbers including stop.",
        buggy_code="def numbers(start, stop):\n    return list(range(start, stop))\n",
        tests=(
            "from solution import numbers\n\n"
            "def test_endpoint():\n"
            "    assert numbers(1, 3) == [1, 2, 3]\n"
        ),
        observed_failure="assert [1, 2] == [1, 2, 3]",
        bugs=[
            BugAnnotation(
                bug_id="bug_1",
                description="range excludes its stop argument.",
                fix="Use a stop boundary one greater than the endpoint.",
            )
        ],
        correct_code=(
            "def numbers(start, stop):\n    return list(range(start, stop + 1))\n"
        ),
    )


def make_output() -> PlannerOutput:
    return PlannerOutput(
        diagnosis_summary="The program treats range's exclusive stop as inclusive.",
        corrected_code=(
            "def numbers(start, stop):\n    return list(range(start, stop + 1))\n"
        ),
        plan=PedagogicalPlan(
            plan_summary="Trace the boundary, explain it, then repair it.",
            steps=[
                PlanStep(
                    step_id="step_1",
                    target_concept="Observe range output",
                    guiding_question="What values does range(1, 3) produce?",
                    expected_answer="It produces 1 and 2, not 3.",
                    related_bug_ids=["bug_1"],
                ),
                PlanStep(
                    step_id="step_2",
                    target_concept="Repair the exclusive boundary",
                    guiding_question="How should the boundary change to include 3?",
                    expected_answer="The stop supplied to range must be one greater.",
                    related_bug_ids=["bug_1"],
                ),
            ],
        ),
    )


class FakePlannerResponses:
    def parse(self, *, model, instructions, input, text_format):
        assert text_format is PlannerOutput
        assert "TRAINING-ONLY ORACLE CONTEXT" in input
        return SimpleNamespace(output_parsed=make_output(), output_text="")


class FakePlannerClient:
    def __init__(self):
        self.responses = FakePlannerResponses()


class FakeVerifierResponses:
    def parse(self, *, model, instructions, input, text_format):
        assert text_format is PlanVerification
        assert "CANDIDATE PLANNER OUTPUT" in input
        return SimpleNamespace(
            output_parsed=PlanVerification(
                accepted=True,
                covered_bug_ids=["bug_1"],
                missing_bug_ids=[],
                invented_or_unsupported_claims=[],
                errors=[],
                regeneration_feedback="",
            ),
            output_text="",
        )


class FakeVerifierClient:
    def __init__(self):
        self.responses = FakeVerifierResponses()


class FakeRunner:
    def run(self, *, code, tests):
        return TestRunResult(
            passed="stop + 1" in code,
            exit_code=0 if "stop + 1" in code else 1,
            stdout="1 passed" if "stop + 1" in code else "1 failed",
            stderr="",
        )


def test_pipeline_returns_verified_planner_output():
    pipeline = PlanningPipeline(
        planner=OfflinePlannerAgent(
            llm=FakePlannerClient(),
            model="fake-planner",
        ),
        verifier=OfflinePlanVerifierAgent(
            llm=FakeVerifierClient(),
            model="fake-verifier",
        ),
        code_runner=FakeRunner(),
    )

    result = pipeline.generate(make_case())

    assert "stop + 1" in result.output.corrected_code
    assert result.output.plan.steps[0].step_id == "step_1"
    assert result.verification.accepted
