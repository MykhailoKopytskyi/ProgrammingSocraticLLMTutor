from app.common.conversation import Conversation
from app.common.message import Message


def test_conversation_owns_and_formats_message_history():
    conversation = Conversation()
    conversation.add(Message(role="student", content="Why does this fail?"))
    conversation.add(Message(role="tutor", content="What does the loop produce?"))

    assert len(conversation) == 2
    assert conversation.last_message == {
        "role": "tutor",
        "content": "What does the loop produce?",
    }
    assert "[1] STUDENT" in conversation.to_text()
    assert "[2] TUTOR" in conversation.to_text()
