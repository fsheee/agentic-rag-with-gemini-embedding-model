"""Conversation history endpoints.

POST /ask writes turns; these endpoints read them back so a client can
restore the user's chat history after a reload or on another device.
Ownership rules are identical to /ask: identity comes from the JWT, a
conversation is filtered on id AND owner together, and another user's
conversation id resolves to 404 — not 403 — so the response never
reveals whether the row exists.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from app.auth_routes import get_current_user
from app.crud import get_conversation_for_user, get_messages, list_conversations
from app.db import get_session
from app.schema import Conversation, Message, User

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummary(BaseModel):
    id: int
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime | None


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


def _summary(conversation: Conversation) -> ConversationSummary:
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _detail(
    conversation: Conversation, messages: list[Message]
) -> ConversationDetail:
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at)
            for m in messages
        ],
    )


@router.get("", response_model=list[ConversationSummary])
def conversations(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[ConversationSummary]:
    """The caller's conversations, most recently active first."""
    return [_summary(c) for c in list_conversations(session, user.id)]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def conversation_messages(
    conversation_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ConversationDetail:
    """One conversation with its full transcript, oldest message first."""
    conversation = get_conversation_for_user(session, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")

    return _detail(conversation, get_messages(session, conversation_id))
