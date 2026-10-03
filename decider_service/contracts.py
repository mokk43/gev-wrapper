from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

StructuredValue = str | dict[str, JsonValue] | list[JsonValue]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(strict=True)

    type: Literal["choice"]
    instructions: StructuredValue | None = None
    criteria: dict[str, StructuredValue | None] = Field(min_length=2)


class NoulCriteria(BaseModel):
    model_config = ConfigDict(strict=True)

    true: StructuredValue | None = None
    false: StructuredValue | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(strict=True)

    type: Literal["noul"]
    instructions: StructuredValue | None = None
    criteria: NoulCriteria | None = None


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(strict=True)

    type: Literal["score"]
    instructions: StructuredValue | None = None
    criteria: list[StructuredValue] = Field(min_length=1)


Question = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    state: StructuredValue
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)


Answer = Annotated[
    ChoiceAnswer | NoulAnswer,
    Field(discriminator="type"),
]


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    answers: dict[str, Answer] = Field(min_length=1)
    usage: Usage
