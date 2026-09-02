from __future__ import annotations

import json
from pathlib import Path

from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.models import BenchmarkCase

from ..backends import StructuredModelBackend
from ..conversation import ConversationRenderer
from ..models import EvaluationRound, EvaluationSessionResult
from ..storage import JsonlStore
from .results import ScoredTurn, ScoreSummaryBuilder, TurnQualityScore


class DialogueEvaluator:
    JUDGE_SYSTEM_PROMPT = """
Evaluate one Tutor response from a Python debugging dialogue. The Tutor identity is hidden. Use the benchmark oracle only as reference grounding; do not require textual agreement with the reference repair when another repair is technically valid.

Score each dimension independently on 0, 1, or 2.

1. technical_correctness
0 = materially wrong, misleading, unsupported, or invents evidence.
1 = broadly correct but contains a meaningful imprecision or weakly supported claim.
2 = technically correct and appropriately grounded in the available code/evidence.

2. target_alignment
0 = works on the wrong or unrelated debugging objective.
1 = related but too broad, premature, or only partly aligned with the learner's current need.
2 = directly addresses the correct current debugging/learning objective.

3. adaptive_feedback
0 = substantially ignores the latest Student reasoning, question, confusion, code, or result.
1 = adapts partially but misses an important Student-specific signal.
2 = clearly responds to and builds on the Student's specific latest state/content.

4. scaffolding
0 = gives away the diagnosis/repair when the Student could reasonably have been guided to derive it.
1 = gives useful guidance but is somewhat too revealing or insufficiently incremental.
2 = gives the minimum useful support while leaving meaningful reasoning/application to the Student.

5. reasoning_elicitation
0 = mainly tells or directs without eliciting meaningful reasoning where reasoning is appropriate.
1 = elicits some reasoning, but it is shallow or most of the reasoning is already supplied.
2 = meaningfully asks the Student to trace, predict, compare, explain, justify, test, or derive an important part of the repair.
If a direct answer to a Student clarification is pedagogically appropriate, do not penalise merely because the response is not phrased as a question.

6. actionability
0 = leaves the Student without a clear useful next step.
1 = somewhat useful but vague or incomplete.
2 = gives a clear next thing to inspect, reason about, test, explain, or implement.

solution_leakage is true only when the Tutor prematurely reveals essentially the diagnosis, exact repair, corrected expression/code, or complete solution before the Student has introduced/derived it. Information already introduced by the Student may be discussed normally.

Keep notes short and evidence-based. Return exactly one TurnQualityScore.
""".strip()

    def __init__(
        self,
        *,
        backend: StructuredModelBackend,
        cases_path: str | Path,
    ):
        self.backend = backend
        self.cases = {
            case.case_id: case for case in BenchmarkCaseStore(cases_path).load()
        }
        self.summary_builder = ScoreSummaryBuilder()

    def score_file(
        self,
        *,
        input_path: str | Path,
        scores_path: str | Path,
        summary_path: str | Path,
    ) -> dict:
        session_store = JsonlStore(input_path)
        score_store = JsonlStore(scores_path)
        sessions = session_store.read_models(EvaluationSessionResult)
        if not sessions:
            raise RuntimeError(f"No evaluation sessions found in {Path(input_path)}")
        self._validate_sessions(sessions)

        scored = score_store.read_models(ScoredTurn)
        self._validate_existing_scores(scored)
        completed = self._completed_turns(scored)

        for session in sessions:
            case = self.cases[session.case_id]
            for round_index, record in enumerate(session.rounds):
                if record.tutor is None:
                    continue

                key = self._turn_key(session.session_id, record.round_index)
                if key in completed:
                    continue

                previous = tuple(session.rounds[:round_index])
                score = self._score_turn(
                    case=case,
                    session=session,
                    previous=previous,
                    round_index=round_index,
                )
                score_store.append_model(score)
                scored.append(score)
                completed.add(key)

        summary = self.summary_builder.build(
            sessions=sessions,
            scored=scored,
            judge_backend=self.backend.backend_id,
            judge_config_id=self.backend.config_id,
        )
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _score_turn(
        self,
        *,
        case: BenchmarkCase,
        session: EvaluationSessionResult,
        previous: tuple[EvaluationRound, ...],
        round_index: int,
    ) -> ScoredTurn:
        record = session.rounds[round_index]
        assert record.tutor is not None

        prompt = self._turn_prompt(
            case=case,
            session=session,
            previous=previous,
            round_index=round_index,
        )
        result = self.backend.generate(
            system_prompt=self.JUDGE_SYSTEM_PROMPT,
            user_prompt=prompt,
            output_type=TurnQualityScore,
            max_output_tokens=1800,
        )

        learner_state_match = None
        if record.tutor.learner_state is not None:
            learner_state_match = (
                record.tutor.learner_state == record.student.target_learner_state
            )

        return ScoredTurn(
            session_id=session.session_id,
            case_id=session.case_id,
            round_index=record.round_index,
            tutor_system=session.tutor_system,
            student_variant=session.student_variant.value,
            judge_backend=self.backend.backend_id,
            judge_config_id=self.backend.config_id,
            learner_state_match=learner_state_match,
            scores=result.parsed,
        )

    def _turn_prompt(
        self,
        *,
        case: BenchmarkCase,
        session: EvaluationSessionResult,
        previous: tuple[EvaluationRound, ...],
        round_index: int,
    ) -> str:
        record = session.rounds[round_index]
        assert record.tutor is not None
        return (
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "EVALUATION ORACLE:\n"
            f"{case.oracle_context()}\n\n"
            "VISIBLE CONVERSATION BEFORE THIS TURN:\n"
            f"{ConversationRenderer.render(previous)}\n\n"
            "CONTROLLED STUDENT TARGET STATE:\n"
            f"{record.student.target_learner_state.value}\n\n"
            "LATEST STUDENT TURN:\n"
            f"{ConversationRenderer.render_student_turn(record.student, record.code_execution)}\n\n"
            "TUTOR RESPONSE TO SCORE:\n"
            f"{record.tutor.reply.strip()}"
        )

    def _validate_sessions(self, sessions: list[EvaluationSessionResult]) -> None:
        seen: set[str] = set()
        for session in sessions:
            if session.session_id in seen:
                raise RuntimeError(f"Duplicate evaluation session: {session.session_id}")
            seen.add(session.session_id)
            if session.case_id not in self.cases:
                raise RuntimeError(
                    f"Evaluation session references unknown case_id {session.case_id}."
                )

    def _validate_existing_scores(
        self,
        rows: list[ScoredTurn],
    ) -> None:
        seen: set[str] = set()
        for row in rows:
            key = self._turn_key(row.session_id, row.round_index)
            if key in seen:
                raise RuntimeError(f"Duplicate score record: {key}")
            seen.add(key)

            if row.judge_backend != self.backend.backend_id:
                raise RuntimeError(
                    f"Existing score file uses judge {row.judge_backend}, "
                    f"but this run uses {self.backend.backend_id}. Use a new output path."
                )
            if row.judge_config_id != self.backend.config_id:
                raise RuntimeError(
                    "Existing score file uses a different judge backend configuration. "
                    "Use a new output path rather than reusing stale scores."
                )

    @staticmethod
    def _completed_turns(rows: list[ScoredTurn]) -> set[str]:
        return {
            DialogueEvaluator._turn_key(row.session_id, row.round_index)
            for row in rows
        }

    @staticmethod
    def _turn_key(session_id: str, round_index: int) -> str:
        return f"{session_id}::round-{round_index}"
