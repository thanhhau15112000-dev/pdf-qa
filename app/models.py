from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1, le=80)
    quote: str = Field(min_length=8, max_length=450)


class AnswerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=300)
    status: Literal["supported", "inference", "insufficient", "conflict"]
    answer: str = Field(min_length=1, max_length=6000)
    citations: list[Citation] = Field(max_length=8)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[AnswerItem] = Field(min_length=1, max_length=12)
    notice: str = Field(max_length=1500)


class AskRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=80)
    question: str = Field(default="", max_length=2000)
    mode: Literal["ask", "solve"] = "ask"


class UnlockRequest(BaseModel):
    code: str = Field(min_length=1, max_length=200)
