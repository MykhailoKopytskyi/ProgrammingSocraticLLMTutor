from typing import Any, NotRequired, TypedDict


class Message(TypedDict):
    role: str
    content: str
    metadata: NotRequired[
        dict[str, Any]
    ]  # Need to change later to define stricter format !!!
