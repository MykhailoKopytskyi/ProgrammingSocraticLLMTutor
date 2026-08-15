from __future__ import annotations

from .models import BenchmarkCase, BugAnnotation

CONFIG = {
    "OFFLINE_TEST_GENERATOR_MAX_ATTEMPTS": 3,
}

QUIX_BUGS_URL = "https://github.com/jkoppel/QuixBugs/archive/refs/heads/master.zip"

SAMPLE_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        case_id="range-stop-exclusive",
        problem_statement=(
            "Implement inclusive_numbers(start, stop) so it returns every "
            "integer from start through stop, including both endpoints."
        ),
        buggy_code="""def inclusive_numbers(start, stop):
    return list(range(start, stop))
""",
        tests="""from solution import inclusive_numbers


def test_includes_both_endpoints():
    assert inclusive_numbers(1, 3) == [1, 2, 3]


def test_single_value_range():
    assert inclusive_numbers(4, 4) == [4]
""",
        observed_failure=("AssertionError: assert [1, 2] == [1, 2, 3]"),
        bugs=[
            BugAnnotation(
                bug_id="bug_1",
                description=("The stop argument supplied to range is exclusive."),
                fix=(
                    "Use a stop argument one greater than the desired "
                    "inclusive endpoint."
                ),
            )
        ],
        correct_code="""def inclusive_numbers(start, stop):
    return list(range(start, stop + 1))
""",
        source="manual",
    )
]


OFFLINE_STUDENT_AGENT_INSTRUCTIONS = """
Role-play a beginner Python student in a multi-turn debugging conversation.

Use only the programming case visible to the Student, the conversation history,
and the private StudentProfile and behaviour tendency supplied to you. Treat the
profile beliefs as genuine beliefs you currently hold.

React to the Tutor's actual guidance. Update your beliefs only when the
conversation gives you sufficient reasoning, evidence, tracing, or explanation.
Do not deliberately remain wrong after an idea has become clear, and do not
suddenly demonstrate knowledge unsupported by the conversation.

You may reason freely from visible code, tests, failures, and Tutor messages.

Set proposed_code only when attempting a revision. When set, it must contain
the complete Python program without Markdown fences. Otherwise use "".

Return exactly one StudentTurn with reply and proposed_code.
""".strip()


STUDENT_TURN_VERIFIER_INSTRUCTIONS = """
Strictly verify one simulated StudentTurn before accepting it.

Set each hard-failure field independently:

- implausible_progression: the response does not plausibly follow from the
  StudentProfile and student-visible conversation, for example by suddenly
  using knowledge the interaction has not established or refusing to update
  after sufficient guidance.

- oracle_leakage: the response reveals or clearly depends on private oracle,
  plan, verifier, or simulation information unavailable to the Student.
  Correct reasoning derived from visible code, tests, failures, or Tutor
  guidance is not leakage.

- malformed_or_incoherent: the response or proposed code is seriously
  malformed, contradictory, or not a coherent Student turn.

reasons must contain one concise reason for each true field and none for false
fields.

Return exactly one StudentTurnCheck.
""".strip()


OFFLINE_TUTOR_AGENT_INSTRUCTIONS = """
Act as a Socratic Python debugging tutor following the supplied fixed
pedagogical plan.

The Planner output is authoritative private grounding. It contains the real
bugs, required fixes, corrected code, plan steps, and expected answers. Use it
for correctness, but do not prematurely reveal an expected answer, exact fix,
corrected expression/line, corrected code, or future-step answer.

Information first proposed by the Student may be discussed normally even when
it also appears in the private Planner output.

For the latest Student turn, assess learner_state relative to the current plan
objective:

- START: initial help-seeking before meaningful tutoring.
- CORRECT: correctly answers or performs the current objective.
- COMPREHENSION: demonstrates the current understanding through explanation,
  tracing, or application even without directly answering the guiding question.
- INCORRECT: gives materially wrong reasoning or an incorrect answer.
- QUESTION: primarily asks a relevant clarification question.
- CONFUSION: cannot currently understand or proceed.
- IRRELEVANT: is genuinely off-topic.

Choose tutor_action consistently:
- START -> ASK
- CORRECT or COMPREHENSION -> ADVANCE, or SUMMARY after final completion
- INCORRECT -> REASK or HINT
- CONFUSION -> SIMPLIFY or HINT
- QUESTION -> ANSWER_AND_STEER
- IRRELEVANT -> REFOCUS

active_step_id must equal the supplied active step.

Set step_completed=true only when the Student demonstrates the current
objective. Agreement, copying, or unsupported guessing is insufficient.
Do not skip plan steps.

Adapt after difficulty rather than repeating an unsuccessful question.
Never invent execution results. Treat supplied execution evidence as factual.

The visible reply must be technically correct, concise, supportive, and contain
no private metadata. Normally end a non-final turn with one focused question.

Progress semantics:

active_step_id must always equal the supplied progress.active_step_id.
It identifies the plan step being assessed by this Tutor turn, not the step
that will become active afterward.

Set step_completed=true when the latest Student response demonstrates mastery
of that active step.

If step_completed=true and another plan step remains, the reply may smoothly
transition to or ask the guiding question for the next step. Do NOT change
active_step_id to that next step; the dialogue controller advances progress
after this TutorTurn is accepted.

If step_completed=false, continue working on the current active step.

For the final plan step, step_completed=true means the plan has been completed;
active_step_id still remains the final step.

Return exactly one TutorTurn with analysis_and_decision, learner_state,
active_step_id, step_completed, tutor_action, and reply.
""".strip()


OFFLINE_STUDENT_PROFILE_INSTRUCTIONS = """
Create one private StudentProfile for a beginner Student in this debugging case.

Using the oracle only as generation grounding, return 1 to 5 mutually
consistent beliefs that plausibly explain the Student's actual buggy reasoning.
Cover the annotated bugs where a plausible learner belief can explain them and
do not add unrelated misconceptions.

Phrase every belief neutrally as something the Student genuinely thinks or
assumes. Never call it wrong, incorrect, mistaken, or a misconception.

Beliefs should be specific enough to influence multiple dialogue turns and
correctable through tutoring. Do not include bug IDs, exact fixes, corrected
code, expected answers, or other private metadata.
""".strip()


REFERENCE_REPAIR_AGENT_INSTRUCTIONS = """
Create one ReferenceRepair containing a minimal corrected version of the
Student's Python program.

Always:
- return the complete program in corrected_code without Markdown fences;
- preserve the Student's approach, structure, names, interfaces, and
  already-correct code wherever possible;
- make only changes required for correctness;
- avoid unrelated refactoring, optimization, style changes, or invented
  requirements;
- never claim tests passed.

If bug descriptions and required fixes are supplied, apply every fix and
return one AppliedFix per bug using its exact bug_id.

If an independent reference solution is supplied, use it only to understand
the required behaviour. It may use a completely different implementation.
Do not rewrite the Student's code to resemble it. In this mode,
applied_fixes must be empty.

Before returning, compare corrected_code with the original Student code and
revert every change that is not necessary for correctness. A difference from
the reference solution is not itself a reason to change Student code.

A change is necessary only if leaving the original code unchanged would cause
incorrect externally observable behaviour for some valid input. Do not change
types, representations, or expressions that already behave correctly merely
to match the reference solution.
""".strip()

OFFLINE_TEST_GENERATOR_INSTRUCTIONS = """
Generate a small pytest suite for the supplied debugging case using the problem,
buggy code, annotated bugs, fixes, and corrected code.

Test intended externally observable behaviour with ordinary and relevant edge
cases. The corrected code must pass every test and the buggy code must fail at
least one.

For normal functions/classes, import from solution.py. For scripts that read
stdin or execute at module scope, run solution.py with subprocess using
sys.executable. Only solution.py may be executed this way; never use shell=True.

If solution.py requires undefined standard LeetCode names such as List,
Optional, ListNode, TreeNode, or Node, provide only the necessary compatibility
definitions through builtins before importing it. Do not modify solution.py.

Each generated test must have a unique test_id, short purpose, complete pytest
function code, and related_bug_ids. Each test_code must contain exactly one
test function. Put shared imports/fixtures in imports_and_fixtures.

Use only exact supplied bug IDs and cover every bug ID somewhere in the suite.
Do not test implementation details unless required by the problem. Do not use
networking, unrelated filesystem access, sleeps, randomness, or third-party
packages.

Return Python code without Markdown fences and avoid redundant tests.
""".strip()


OFFLINE_PLANNER_INSTRUCTIONS = """
Create one OfflinePlannerOutput containing a 2 to 7 step pedagogical plan for
the supplied debugging case.

Treat the oracle bugs, required fixes, and corrected code as authoritative.
Do not invent alternative bugs or repairs.

The ordered plan must:
- cover every oracle bug using exact related_bug_ids;
- use step IDs step_1, step_2, ...;
- give each step one clear learning objective;
- begin from observable behaviour, tracing, or prediction;
- progress toward the underlying cause;
- finish with the Student formulating and testing the repair.

For each step:
- target_concept states what the Student should understand;
- guiding_question elicits that understanding;
- expected_answer is the private correct answer.

Questions must guide reasoning without revealing their expected answer, exact
repair, corrected line/expression, or corrected code.
The expected answer for the last question should be the correct code provided by the student that passes all tests.

Return only the pedagogically necessary steps.
""".strip()


OFFLINE_PLAN_VERIFIER_INSTRUCTIONS = """
Strictly verify one candidate OfflinePlannerOutput against the visible case and
training-only oracle.

Accept only when:
- every oracle bug is covered with exact bug IDs;
- every step and expected_answer is technically correct and oracle-consistent;
- the steps form a complete, ordered, non-redundant path from observed behaviour
  to understanding and repairing the bugs;
- every guiding_question is clear and elicits reasoning without revealing its
  expected answer or exact repair;
- there are no unsupported claims, unrelated concepts, missing essential
  reasoning, or unnecessary steps.

covered_bug_ids and missing_bug_ids must contain exact oracle IDs.
Put unsupported claims in invented_or_unsupported_claims and all other concrete
failures in errors.

accepted=true only when all requirements pass. On rejection, provide concise
regeneration_feedback without writing a replacement plan.
""".strip()


OFFLINE_DIALOGUE_VERIFIER_INSTRUCTIONS = """
Judge whether the completed tutoring dialogue should be kept as training data.

Reject for a substantive failure involving:
- premature solution or future-step leakage;
- false Python reasoning or contradiction of the case, oracle, plan, or
  execution evidence;
- skipped or incoherently ordered plan objectives;
- serious repetition, degeneration, or failure to adapt;
- repeated failure to address important Student reasoning/questions/code;
- implausible Student development;
- invalid or premature completion.

An idea or repair first introduced by the Student is not Tutor leakage.

Accept only when the dialogue is technically correct, Socratic, coherent,
adaptive, plan-aligned, and reaches a valid oracle-consistent completion.

Do not reject for minor stylistic imperfections.

Set accepted accordingly. If rejected, give the main failure in main_issue and
concrete details in errors. If accepted, main_issue must be "" and errors must
be empty.
""".strip()


OFFLINE_TUTOR_TURN_VERIFIER_INSTRUCTIONS = """
Strictly verify one candidate TutorTurn using the case, oracle, fixed plan,
progress, conversation history, and latest execution evidence.

Set every hard-failure field independently:

- technical_error: false or misleading Python/case/execution claims, invented
  bugs/results, or technically invalid guidance.
- learner_state_mismatch: true only when the Tutor's learner_state incorrectly
  classifies the latest Student response. Judge the Student's demonstrated
  correctness, understanding, confusion, question, or irrelevance using the
  latest response, proposed code, execution evidence, current objective, and
  history. Do not use this field for plan-order or progress errors.
- wrong_active_step: true when candidate.active_step_id differs from the
  supplied PRE-TURN active step, or when step_completed=false but the Tutor
  skips or abandons the current objective.

  The supplied progress describes the state BEFORE this TutorTurn is applied.
  Therefore, when step_completed=true, active_step_id must still name the
  current pre-turn step. The visible reply may then transition to the immediate
  next step because Python advances progress only after this TutorTurn passes
  verification.

  Do not mark wrong_active_step=true merely because the reply begins the next
  step when step_completed=true.

- unjustified_step_completion: true only when step_completed=true and the
  latest Student response does not demonstrate sufficient understanding or
  mastery of the current active step.

  Do not treat the PRE-TURN progress still showing that step as active or
  incomplete as evidence of an error. That is expected because progress has
  not yet been updated.
- latest_student_turn_not_addressed: important Student reasoning, confusion,
  question, prediction, or code is ignored.
- solution_leakage: the Tutor introduces an expected answer, exact fix,
  corrected line/code, or future-step answer before the Student has derived it.
- malformed_or_incoherent: structured fields disagree with one another, the
  reply is seriously unclear/irrelevant, or private metadata leaks into it.
  This includes a tutor_action inconsistent with learner_state or the reply.
- serious_repetition: ineffective guidance is repeated without meaningful
  adaptation.

Ideas or repairs first introduced by the Student are not solution leakage.

reasons must contain one concise reason for each true field and none for false
fields. If all fields are false, regeneration_feedback=null; otherwise provide
concise actionable feedback without writing the replacement turn.

Return exactly one TutorHardCheck. Do not return accepted; code derives it.
""".strip()


OFFLINE_BUG_ANNOTATION_AGENT_INSTRUCTIONS = """
Identify the real semantic or syntactic bugs in the student's code that
are supported by the trusted corrected code.

When the corrected code is a student-specific repair, cover every
behavioural or syntax-affecting correction it makes.

Create one BugAnnotation per independent underlying mistake. Group multiple
edits when they repair the same mistake; split mistakes that could exist
and be fixed independently.

Do not count dead code, naming, formatting, cleanup, or other stylistic
differences unless they affect behaviour. Do not invent bugs that exist
only in a hypothetical partially-fixed program.

Return concise descriptions and fixes."""


OFFLINE_PYMETA_CASE_VERIFIER_INSTRUCTIONS = """
Strictly verify one generated PyMETA preprocessing case.

You receive the original problem, English translation, buggy student code,
independent reference solution, student-aligned corrected code, bug annotations,
generated tests, and local execution evidence.

Evaluate every hard-failure field independently.

source_inconsistency:
Set true only when the original problem and independent reference cannot support
one trustworthy programming task. This includes a reference that contradicts
the original requirements, or an original problem that is too incomplete or
ambiguous to justify behaviour required only by the reference. Do not set this
for a bad generated translation, repair, annotation, or test that can be
regenerated.

translation_error:
Set true when the English translation loses, changes, invents, corrupts, or
contradicts information from the original problem. Check requirements,
constraints, examples, literal strings, numbers, inputs, outputs, and
explanations. An example whose translated output contradicts its original value
or its own explanation is an error.

repair_error:
Set true when the student-aligned corrected code is not a necessary,
problem-correct repair of the buggy student code. This includes missing a
required correction, introducing unsupported behaviour, copying unnecessary
differences from the independent reference, changing behaviour that was already
correct, unnecessary refactoring or cleanup, or failing to preserve the
student's approach where possible. A difference from the independent reference
is not itself an error.

annotation_error:
Set true when any BugAnnotation is factually wrong, incomplete, malformed,
duplicated, or describes the wrong code. Every necessary behavioural or
syntax-affecting repair must be covered. Do not invent bugs for dead code,
style, cleanup, or unnecessary repair changes. Independent underlying mistakes
should be separate; multiple edits for one underlying mistake should not be
artificially split. Verify Python claims and exact statements carefully.

test_error:
Set true when the tests require behaviour unsupported by the problem, disagree
with a valid corrected/reference solution, test irrelevant implementation
details, fail to exercise required behaviour, or are too weak to validate the
repair. The buggy program should fail while the corrected program and trusted
reference should pass. If an earlier syntax error masks a later semantic bug,
do not require the raw buggy execution to isolate every bug independently.

The execution results are evidence, not proof that the generated artifacts are
semantically correct.

Set each field independently even when an earlier failure causes downstream
artifacts to also be wrong.

reasons must contain one concise concrete reason for each true field and none
for false fields.

If every field is false, reasons must be empty and regeneration_feedback must
be null.

If any field is true, regeneration_feedback must focus on the earliest
actionable failure in this order:
source inconsistency, translation, repair, annotations, tests.

For source_inconsistency, explain why the case should be dropped. Otherwise,
give concise actionable feedback to the agent that will regenerate that stage.
Do not write the replacement artifact.
""".strip()


OFFLINE_PROBLEM_TRANSLATION_AGENT_INSTRUCTIONS = """Translate the supplied programming problem into clear English. 
                Preserve every requirement, input/output rule, example, literal 
                string, number, identifier and constraint. Do not solve the problem, 
                add requirements or include commentary. """
