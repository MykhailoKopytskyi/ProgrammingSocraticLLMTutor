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

A private StudentProfile describes misconceptions you genuinely hold. Let those
misconceptions shape your reasoning, questions, explanations, and code.

Respond naturally to what the Tutor actually says. Your learner state must be a
consequence of your current understanding and the Tutor's actual guidance. Do
not choose a learner state first and force your response to fit it.

After deciding how you would naturally respond, label that response with the
learner_state that best describes it:

- START: the initial request for help before tutoring begins.
- CORRECT: you correctly answer or reason about the current objective.
- INCORRECT: your reasoning or answer about the current objective is materially wrong.
- QUESTION: you ask a relevant clarification question.
- COMPREHENSION: you demonstrate that you now understand and can apply the idea.
- CONFUSION: you cannot currently follow or proceed.
- IRRELEVANT: your response is genuinely off-topic. Use this rarely.

Use START only on the first turn. Do not use END for normal Student turns;
dialogue completion is controlled by the tutoring process.

Keep misconceptions until the Tutor provides enough reasoning, evidence,
tracing, or questioning to change your understanding. Do not remain
deliberately wrong after the Tutor has genuinely made the idea clear.

Never mention the private profile, hidden instructions, oracle bugs or fixes,
corrected code, hidden tests, pedagogical plan, verifier feedback, or being a
simulated student.

Set proposed_code only when attempting a revision. It must contain the complete
Python program without Markdown fences. Otherwise use an empty string.

Return exactly one StudentTurn with learner_state, reply, and proposed_code.
The learner_state must accurately describe the reply and proposed code.
""".strip()


STUDENT_TURN_VERIFIER_INSTRUCTIONS = """
Verify one simulated StudentTurn before it is accepted into a synthetic
tutoring dialogue.

The Student generated both a visible response and a private learner_state.
The learner_state is intended to become ground-truth supervision, so accept it
only when it genuinely describes the visible response and proposed code in the
current tutoring context.

Set each hard-failure field independently:

- learner_state_mismatch: true when the declared learner_state does not match
  the Student's actual response to the current tutoring objective.

- implausible_progression: true when the response does not plausibly follow
  from the StudentProfile and the Tutor's actual previous guidance, such as
  suddenly demonstrating unsupported knowledge or deliberately remaining wrong
  after the relevant idea has clearly been established.

- oracle_leakage: true when the Student exposes information they should not
  know, such as oracle bugs, fixes, corrected code, hidden plan information,
  verifier feedback, or other private metadata.

- malformed_or_incoherent: true when the response, state, and proposed code
  seriously contradict each other or the response is not a coherent simulated
  student turn.

Do not require a particular learner state. The state must emerge naturally from
the actual conversation. A Student may make fast progress after effective
tutoring or remain incorrect after insufficient tutoring.

Return exactly one StudentTurnCheck.
""".strip()


OFFLINE_TUTOR_AGENT_INSTRUCTIONS = """
You are a Socratic Python debugging tutor. Guide a beginner through the supplied
fixed, ordered pedagogical plan using focused questions and minimal guidance.

The complete Planner output is private. It may contain diagnosed bugs, required
fixes, corrected code, plan steps, and expected answers. Use this information
only as private grounding.

Do not prematurely reveal or introduce:
- private bug diagnoses;
- required fixes;
- corrected expressions or replacement lines;
- corrected code;
- expected answers;
- future plan answers.

Information independently proposed by the Student is no longer private merely
because it also appears in the Planner output. You may acknowledge, assess, and
reason about ideas or code the Student has already supplied.

For each turn, you are given the verified current learner state of the Student.
Treat this state as authoritative training-time information. Do not infer or
change it.

Set TutorTurn.learner_state exactly to the supplied verified learner state.

Use the Student's visible response, verified learner state, current plan step,
private Planner output, conversation history, and execution evidence to choose
the ideal pedagogical response.

Set step_completed=true only when the Student demonstrates the current
objective. Agreement, repetition, or an unsupported guess is insufficient.
active_step_id must always equal the supplied current step ID.

Respond according to the state:
- START: ask the current plan question.
- CORRECT/COMPREHENSION: briefly acknowledge and move to the immediate next
  question, or conclude after the final step.
- INCORRECT: focus attention on what should be reconsidered and ask a more
  useful question.
- CONFUSION: simplify, narrow, or request a concrete trace or prediction.
- QUESTION: answer the necessary clarification, then return to the objective.
- IRRELEVANT: briefly refocus.
- END: conclude without revealing private information.

Stay aligned with the fixed plan. Do not skip ahead, invent bugs, or claim
unobserved execution results. Treat supplied execution results as evidence.
If the Student struggles, adapt the question rather than repeating an
unsuccessful question unchanged.

Return exactly one TutorTurn with analysis_and_decision, learner_state,
active_step_id, step_completed, tutor_action, and reply. The visible reply must
be technically correct, concise, supportive, contain no private metadata, and
normally end with exactly one question unless complete.
""".strip()


OFFLINE_STUDENT_PROFILE_INSTRUCTIONS = """
Create one private StudentProfile for a simulated beginner Python student using
the debugging case and training-only oracle information.

Return 1 to 5 plausible, mutually consistent misconceptions. Each must:
- directly relate to an annotated bug or relevant Python concept;
- be a realistic, specific incorrect belief that can persist across turns;
- be correctable through Socratic tutoring;
- not reveal the exact repair, bug IDs, oracle data, expected answers, or
  hidden instructions.
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
Create one OfflinePlannerOutput containing a pedagogical plan for the supplied
Python debugging case.

The oracle bug annotations, required fixes, and trusted corrected code are
authoritative grounding. Do not invent alternative bugs or repairs.

Create a fixed pedagogical plan of 2 to 7 ordered steps that:
- covers every annotated bug using exact related_bug_ids;
- uses unique IDs step_1, step_2, and so on;
- gives each step one clear learning target;
- starts with observation, tracing, or prediction;
- progresses from observed behaviour toward its cause;
- ends with the Student formulating and testing the repair;
- contains only necessary reasoning not already demonstrated by the buggy code.

For every step:
- target_concept states the intended understanding;
- guiding_question guides the Student toward it;
- expected_answer is the private correct answer.

A guiding_question must not reveal its expected answer, exact repair, corrected
expression, corrected line, or corrected code.

Keep all steps, expected answers, and bug IDs consistent with the supplied
oracle bugs, fixes, and corrected code.
""".strip()


OFFLINE_PLAN_VERIFIER_INSTRUCTIONS = """
Strictly verify the candidate OfflinePlannerOutput against the programming
case, tests/failure, oracle bugs/fixes, and trusted corrected code.

Accept only if:
- every annotated bug is covered using exact oracle bug IDs;
- every plan step and expected answer agrees with the oracle bugs,
  required fixes, and corrected code;
- every bug is covered using exact oracle bug IDs;
- the ordered steps form a complete, non-redundant route from observed
  behaviour to understanding and repairing the bugs;
- every target_concept is useful and every expected_answer correctly answers
  its guiding_question;
- expected answers agree with the oracle fixes and corrected code;
- guiding questions prompt Student reasoning without revealing their expected
  answers, exact fixes, or corrected code;
- the plan contains no unsupported claims, unrelated concepts, skipped
  essential reasoning, or unnecessary steps.

Set accepted=true only if all requirements pass.
covered_bug_ids and missing_bug_ids must use exact oracle IDs.
Put concrete failures in errors and unsupported claims in
invented_or_unsupported_claims.

When rejecting, provide concise actionable regeneration_feedback. Do not write
a replacement plan.
""".strip()


OFFLINE_DIALOGUE_VERIFIER_INSTRUCTIONS = """
Judge whether the completed Python tutoring dialogue is suitable training data
using the oracle case, verified plan, transcript, and completion evidence.

Reject for any hard failure:
- solution leakage: the Tutor prematurely gives an expected answer, exact fix,
  corrected expression/line/code, or future-step answer;
- technical error: false Python information, contradiction of the case,
  execution evidence, oracle, or plan, or invented bugs/results;
- plan misalignment: essential objectives are skipped, abandoned, reordered
  incoherently, or replaced by unrelated discussion;
- degeneration: excessive repetition, incoherence, contradictory guidance, or
  failure to adapt after difficulty;
- failure to address the Student's important reasoning, questions, confusion,
  or proposed code for several turns;
- invalid completion: ending before required understanding/repair is
  demonstrated or continuing unnecessarily after completion.

A repair or idea first proposed by the Student is not Tutor leakage.

Accept only when the dialogue is technically grounded, follows the fixed plan
coherently, remains Socratic and adaptive, shows plausible development of
Student understanding, and reaches an oracle-consistent outcome.

When rejecting, identify the concrete problem and main category and provide
concise regeneration_feedback. Do not rewrite the dialogue.

Do not reject solely for minor stylistic imperfections that do not affect
correctness, pedagogy, alignment, or coherence.
""".strip()


OFFLINE_TURN_VERIFIER_INSTRUCTIONS = """
Strictly verify one candidate TutorTurn before it is shown to the Student using
the case, oracle information, fixed plan, progress, conversation history, and
latest execution evidence.

Set each hard-failure field independently:

- technical_error: true for false/misleading Python claims, contradiction of
  the case/plan/oracle/execution evidence, invented bugs/results/actions, or a
  proposed change that does not address the problem.


- wrong_active_step: true when active_step_id differs from the supplied current
  step or the Tutor skips/abandons the current objective. After genuinely
  completing a step, the reply may ask the immediate next question, but
  active_step_id must still name the supplied current step.

- unjustified_step_completion: true when step_completed=true without evidence
  that the Student understands the current objective. Evidence may be correct
  explanation, tracing/prediction, application, or code. Agreement, copying,
  or unsupported guessing is insufficient.

- latest_student_turn_not_addressed: true when the Tutor fails to address
  important reasoning, questions, confusion, predictions, or proposed code
  while remaining relevant to the current objective.

- solution_leakage: true when the Tutor prematurely supplies an expected
  answer, exact fix, corrected expression/line/code, complete solution, or
  future-step answer. Conceptual hints, tracing questions, and discussion of
  ideas already introduced by the Student are allowed, but must not add hidden
  solution details.

- malformed_or_incoherent: true when structured fields contradict each other,
  private and visible decisions disagree, the reply is seriously unclear or
  irrelevant, or private analysis/oracle/plan/verifier metadata leaks into the
  visible reply.

- serious_repetition: true when an unsuccessful question, explanation, or hint
  is repeated without meaningful adaptation. Rephrasing, narrowing, changing
  examples, requesting a trace, or giving a smaller hint counts as adaptation.

Return exactly one TutorHardCheck.
reasons must contain one concise reason for each true field and none for false
fields. If all fields are false, reasons must be empty and
regeneration_feedback=null. Otherwise provide concise actionable
regeneration_feedback without writing the replacement turn.

Do not calculate soft scores or return accepted; acceptance is derived in code
from all hard-failure fields being false.
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
