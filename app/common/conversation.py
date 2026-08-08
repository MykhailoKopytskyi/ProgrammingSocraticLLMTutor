from __future__ import annotations

from collections.abc import Iterable

from .message import Message


class Conversation:
    """Owns the visible ordered message history for one tutoring dialogue."""

    def __init__(
        self,
        messages: Iterable[Message] | None = None,
    ):
        self._messages: list[Message] = list(messages or [])

    @property
    def messages(self) -> list[Message]:
        """Return a copy so callers cannot mutate history without add()."""

        return list(self._messages)

    @property
    def last_message(self) -> Message | None:
        if not self._messages:
            return None

        return self._messages[-1]

    @property
    def is_empty(self) -> bool:
        return not self._messages

    def add(self, message: Message) -> None:
        role = message["role"].strip()
        content = message["content"].strip()

        if not role:
            raise ValueError("message role must not be empty")

        if not content:
            raise ValueError("message content must not be empty")

        self._messages.append(message)

    def to_text(self) -> str:
        if not self._messages:
            return "[empty]"

        return "\n\n".join(
            f"[{index}] {message['role'].upper()}\n{message['content'].strip()}"
            for index, message in enumerate(self._messages, start=1)
        )

    def __len__(self) -> int:
        return len(self._messages)
