from __future__ import annotations

from .message import Message
from .models import StudentTurn


def history_to_text(
    history: list[Message],
) -> str:
    """
    Convert accepted visible conversation messages into prompt text.

    Metadata is deliberately excluded because it may contain hidden labels
    or other offline-only information.
    """

    if not history:
        return "[No previous messages]"

    formatted_messages: list[str] = []

    for index, message in enumerate(history, start=1):
        role = message["role"].upper()
        content = message["content"].strip()

        formatted_messages.append(f"[{index}] {role}\n{content}")

    return "\n\n".join(formatted_messages)


def student_turn_to_message(
    turn: StudentTurn,
) -> Message:
    """
    Convert a structured StudentTurn into the visible message stored in
    conversation history.

    learner_state is deliberately excluded because it is hidden from the Tutor.
    """

    content = turn.reply.strip()

    if turn.proposed_code.strip():
        content += f"\n\nProposed code:\n```python\n{turn.proposed_code.rstrip()}\n```"

    return Message(
        role="student",
        content=content,
    )
