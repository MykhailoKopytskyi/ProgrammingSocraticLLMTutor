from types import SimpleNamespace

from app.agents.offline.offline_plan_verifier_agent import (
    OfflinePlanVerifierAgent,
    PlanVerification,
)
from app.agents.offline.offline_planner_agent import (
    OfflinePlannerAgent,
    OfflinePlannerOutput,
)
from app.common.models import (
    BenchmarkCase,
    BugAnnotation,
    PedagogicalPlan,
    PlanStep,
)
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


def make_offline_output() -> OfflinePlannerOutput:
    return OfflinePlannerOutput(
        diagnosis_summary="The program treats range's exclusive stop as inclusive.",
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
        assert text_format is OfflinePlannerOutput
        assert "TRAINING-ONLY ORACLE CONTEXT" in input
        assert "stop + 1" in input
        return SimpleNamespace(output_parsed=make_offline_output(), output_text="")


class FakePlannerClient:
    def __init__(self):
        self.responses = FakePlannerResponses()


class FakeVerifierResponses:
    def parse(self, *, model, instructions, input, text_format):
        assert text_format is PlanVerification
        assert "CANDIDATE OFFLINE PLANNER OUTPUT" in input
        assert '"corrected_code"' not in input.split("CANDIDATE OFFLINE PLANNER OUTPUT:", 1)[1]
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


def test_pipeline_uses_trusted_code_in_final_planner_output():
    case = make_case()
    pipeline = PlanningPipeline(
        planner=OfflinePlannerAgent(
            llm=FakePlannerClient(),
            model="fake-planner",
        ),
        verifier=OfflinePlanVerifierAgent(
            llm=FakeVerifierClient(),
            model="fake-verifier",
        ),
    )

    result = pipeline.generate(case)

    assert result.output.corrected_code == case.correct_code
    assert result.output.plan.steps[0].step_id == "step_1"
    assert result.verification.accepted
