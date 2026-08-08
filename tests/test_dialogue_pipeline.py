from app.agents.offline.offline_dialogue_verifier_agent import DialogueVerification
from app.agents.offline.offline_plan_verifier_agent import PlanVerification
from app.agents.offline.offline_turn_verifier_agent import TutorHardCheck
from app.agents.student_agent import LearnerState, StudentProfile, StudentTurn
from app.agents.tutor_agent import TutorAction, TutorTurn
from app.common.models import (
    BenchmarkCase,
    BugAnnotation,
    PedagogicalPlan,
    PlannerOutput,
    PlanStep,
)
from app.execution.code_runner import TestRunResult
from app.training.dialogue_pipeline import DialogueGenerationPipeline
from app.training.plan_pipeline import VerifiedPlan


class FakeStudentAgent:
    def __init__(self):
        self._turns = [
            StudentTurn(
                learner_state=LearnerState.START,
                reply="I do not understand why 3 is missing.",
            ),
            StudentTurn(
                learner_state=LearnerState.COMPREHENSION,
                reply="The stop is excluded, so I need the next boundary.",
                proposed_code=(
                    "def numbers(start, stop):\n"
                    "    return list(range(start, stop + 1))\n"
                ),
            ),
        ]

    def generate_turn(self, conversation):
        return self._turns.pop(0)


class FakeTutorAgent:
    def __init__(self):
        self.execution_results = []
        self._turns = [
            TutorTurn(
                analysis_and_decision="Student identified the observed boundary.",
                learner_state=LearnerState.CORRECT,
                active_step_id="step_1",
                step_completed=True,
                tutor_action=TutorAction.ADVANCE,
                reply="Good. What does that suggest about the stop boundary?",
            ),
            TutorTurn(
                analysis_and_decision="Student repaired and explained the boundary.",
                learner_state=LearnerState.COMPREHENSION,
                active_step_id="step_2",
                step_completed=True,
                tutor_action=TutorAction.SUMMARY,
                reply="Yes. Can you summarize why the revised boundary works?",
            ),
        ]

    def generate_turn(
        self,
        *,
        conversation,
        progress,
        latest_code_execution=None,
        regeneration_feedback="",
    ):
        self.execution_results.append(latest_code_execution)
        return self._turns.pop(0)


class FakeTurnVerifier:
    def __init__(self):
        self.execution_results = []

    def verify(self, **kwargs):
        self.execution_results.append(kwargs.get("latest_code_execution"))
        return TutorHardCheck(
            technical_error=False,
            learner_state_mismatch=False,
            wrong_active_step=False,
            unjustified_step_completion=False,
            latest_student_turn_not_addressed=False,
            solution_leakage=False,
            malformed_or_incoherent=False,
            serious_repetition=False,
            reasons=[],
            regeneration_feedback=None,
        )


class FakeDialogueVerifier:
    def verify(self, **kwargs):
        return DialogueVerification(
            accepted=True,
            main_issue="",
            errors=[],
            regeneration_feedback="",
        )


class FakeRunner:
    def run(self, *, code, tests):
        passed = "stop + 1" in code
        return TestRunResult(
            passed=passed,
            exit_code=0 if passed else 1,
            stdout="1 passed" if passed else "1 failed",
            stderr="",
        )


def test_dialogue_pipeline_generates_verified_complete_dialogue():
    case = BenchmarkCase(
        case_id="range-case",
        problem_statement="Return all numbers including stop.",
        buggy_code="def numbers(start, stop):\n    return list(range(start, stop))\n",
        tests="from solution import numbers\n",
        observed_failure="1 failed",
        bugs=[
            BugAnnotation(
                bug_id="bug_1",
                description="range excludes stop.",
                fix="Use the next boundary.",
            )
        ],
        correct_code=(
            "def numbers(start, stop):\n    return list(range(start, stop + 1))\n"
        ),
    )
    plan = PedagogicalPlan(
        plan_summary="Observe and repair the boundary.",
        steps=[
            PlanStep(
                step_id="step_1",
                target_concept="Observe the exclusive stop",
                guiding_question="What values are produced?",
                expected_answer="The stop value is excluded.",
                related_bug_ids=["bug_1"],
            ),
            PlanStep(
                step_id="step_2",
                target_concept="Formulate the repair",
                guiding_question="How should the boundary change?",
                expected_answer="Use the next boundary.",
                related_bug_ids=["bug_1"],
            ),
        ],
    )
    output = PlannerOutput(
        diagnosis_summary="The range stop is exclusive.",
        corrected_code=case.correct_code,
        plan=plan,
    )
    verified_plan = VerifiedPlan(
        output=output,
        verification=PlanVerification(
            accepted=True,
            covered_bug_ids=["bug_1"],
            missing_bug_ids=[],
            invented_or_unsupported_claims=[],
            errors=[],
            regeneration_feedback="",
        ),
        attempts=1,
    )

    tutor = FakeTutorAgent()
    turn_verifier = FakeTurnVerifier()
    pipeline = DialogueGenerationPipeline(
        turn_verifier=turn_verifier,
        dialogue_verifier=FakeDialogueVerifier(),
        code_runner=FakeRunner(),
    )
    dialogue = pipeline.generate(
        case=case,
        verified_plan=verified_plan,
        profile=StudentProfile(misconceptions=["range includes stop"]),
        student_agent=FakeStudentAgent(),
        tutor_agent=tutor,
    )

    assert tutor.execution_results[0] is None
    assert tutor.execution_results[1] is not None
    assert tutor.execution_results[1].passed
    assert turn_verifier.execution_results[0] is None
    assert turn_verifier.execution_results[1] is not None
    assert turn_verifier.execution_results[1].passed
    assert dialogue.dialogue_verification.accepted
    assert len(dialogue.turns) == 4
    assert dialogue.turns[2].code_execution is not None
    assert dialogue.turns[2].code_execution.passed
