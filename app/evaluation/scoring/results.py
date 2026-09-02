from __future__ import annotations

from collections import defaultdict
from statistics import median

from pydantic import Field

from app.common.models import StrictModel

from ..models import EvaluationSessionResult


class TurnQualityScore(StrictModel):
    technical_correctness: int = Field(ge=0, le=2)
    target_alignment: int = Field(ge=0, le=2)
    adaptive_feedback: int = Field(ge=0, le=2)
    scaffolding: int = Field(ge=0, le=2)
    reasoning_elicitation: int = Field(ge=0, le=2)
    actionability: int = Field(ge=0, le=2)
    solution_leakage: bool
    notes: str = ""


class ScoredTurn(StrictModel):
    session_id: str
    case_id: str
    round_index: int
    tutor_system: str
    student_variant: str
    judge_backend: str
    judge_config_id: str = ""
    learner_state_match: bool | None = None
    scores: TurnQualityScore


class ScoreSummaryBuilder:
    METRICS = (
        "technical_correctness",
        "target_alignment",
        "adaptive_feedback",
        "scaffolding",
        "reasoning_elicitation",
        "actionability",
    )

    def build(
        self,
        *,
        sessions: list[EvaluationSessionResult],
        scored: list[ScoredTurn],
        judge_backend: str,
        judge_config_id: str,
    ) -> dict:
        variants = sorted({session.student_variant.value for session in sessions})
        return {
            "judge_backend": judge_backend,
            "judge_config_id": judge_config_id,
            "overall": self._group_summary(sessions, scored),
            "by_student_variant": {
                variant: self._group_summary(
                    [
                        session
                        for session in sessions
                        if session.student_variant.value == variant
                    ],
                    [row for row in scored if row.student_variant == variant],
                )
                for variant in variants
            },
        }

    def _group_summary(
        self,
        sessions: list[EvaluationSessionResult],
        scored: list[ScoredTurn],
    ) -> dict:
        session_ids = {session.session_id for session in sessions}
        relevant_scores = [row for row in scored if row.session_id in session_ids]

        by_session: dict[str, list[ScoredTurn]] = defaultdict(list)
        for row in relevant_scores:
            by_session[row.session_id].append(row)

        dialogue_means: dict[str, dict[str, float]] = {}
        for session_id, turns in by_session.items():
            dialogue_means[session_id] = {
                metric: sum(getattr(turn.scores, metric) for turn in turns) / len(turns)
                for metric in self.METRICS
            }

        metric_means: dict[str, float] = {}
        for metric in self.METRICS:
            values = [row[metric] for row in dialogue_means.values()]
            metric_means[metric] = sum(values) / len(values) if values else 0.0

        leakage_dialogues = {
            row.session_id for row in relevant_scores if row.scores.solution_leakage
        }
        leakage_turns = sum(row.scores.solution_leakage for row in relevant_scores)
        state_rows = [row for row in relevant_scores if row.learner_state_match is not None]
        round_counts = [len(session.rounds) for session in sessions]
        tutor_turns = [
            record.tutor
            for session in sessions
            for record in session.rounds
            if record.tutor is not None
        ]
        violation_turns = sum(bool(turn.protocol_violations) for turn in tutor_turns)

        return {
            "sessions": len(sessions),
            "scored_turns": len(relevant_scores),
            "solve_rate": (
                sum(session.solved for session in sessions) / len(sessions)
                if sessions
                else 0.0
            ),
            "mean_rounds": (
                sum(round_counts) / len(round_counts) if round_counts else 0.0
            ),
            "median_rounds": median(round_counts) if round_counts else 0.0,
            "pedagogical_metrics_macro_by_dialogue": metric_means,
            "turn_leakage_rate": (
                leakage_turns / len(relevant_scores) if relevant_scores else 0.0
            ),
            "dialogue_leakage_rate": (
                len(leakage_dialogues) / len(sessions) if sessions else 0.0
            ),
            "learner_state_accuracy": (
                sum(bool(row.learner_state_match) for row in state_rows) / len(state_rows)
                if state_rows
                else None
            ),
            "protocol_violation_turn_rate": (
                violation_turns / len(tutor_turns) if tutor_turns else None
            ),
            "termination_reasons": self._termination_counts(sessions),
        }

    @staticmethod
    def _termination_counts(sessions: list[EvaluationSessionResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for session in sessions:
            counts[session.termination_reason] = counts.get(session.termination_reason, 0) + 1
        return counts
