from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    page: int = Field(ge=1, le=80)
    quote: str = Field(min_length=8, max_length=450)


class AnswerItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=300)
    status: Literal["supported", "inference", "insufficient", "conflict"]
    answer: str = Field(max_length=6000)
    citations: list[Citation] = Field(max_length=30)  # PDF uploads are limited to 30 pages.
    missing_information: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=8
    )
    clarifying_question: str = Field(default="", max_length=500)
    assumptions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(default_factory=list, max_length=8)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    items: list[AnswerItem] = Field(min_length=1, max_length=12)
    notice: str = Field(max_length=1500)


class AnswerResult(Answer):
    provider: Literal["groq", "gemini"]
    model: str


class AskRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=80)
    question: str = Field(default="", max_length=2000)
    mode: Literal["ask", "solve"] = "ask"
    context_question: str = Field(default="", max_length=2000)

