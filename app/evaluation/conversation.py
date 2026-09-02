from __future__ import annotations

from app.common.code_runner import TestRunResult

from .models import EvaluationCodeExecution, EvaluationRound, EvaluationStudentTurn


class ConversationRenderer:
    @classmethod
    def render(cls, history: tuple[EvaluationRound, ...]) -> str:
        if not history:
            return "[empty]"

        parts: list[str] = []
        for record in history:
            parts.append(
                "STUDENT:\n"
                + cls.render_student_turn(record.student, record.code_execution)
            )
            if record.tutor is not None:
                parts.append(f"TUTOR:\n{record.tutor.reply.strip()}")
        return "\n\n".join(parts)

    @staticmethod
    def render_student_turn(
        student: EvaluationStudentTurn,
        execution: EvaluationCodeExecution | TestRunResult | None = None,
    ) -> str:
        text = student.reply.strip()
        if student.proposed_code.strip():
            text += (
                "\n\nProposed code:\n```python\n"
                f"{student.proposed_code.rstrip()}\n```"
            )
        if execution is not None:
            text += (
                f"\n\nCode execution:\npassed={execution.passed}\n"
                f"{execution.output}"
            )
        return text

    @staticmethod
    def structured(history: tuple[EvaluationRound, ...]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for record in history:
            student: dict[str, object] = {
                "role": "student",
                "round": record.round_index,
                "reply": record.student.reply,
            }
            if record.student.proposed_code.strip():
                student["proposed_code"] = record.student.proposed_code
            if record.code_execution is not None:
                student["code_execution"] = record.code_execution.model_dump(mode="json")
            messages.append(student)

            if record.tutor is not None:
                messages.append(
                    {
                        "role": "tutor",
                        "round": record.round_index,
                        "reply": record.tutor.reply,
                    }
                )
        return messages
