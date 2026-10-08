"""Lazy API request schemas; result viewing does not import Pydantic."""


def v3_api_schemas():
    from pydantic import BaseModel, Field, ConfigDict, StrictBool, StrictInt

    class Search(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1, max_length=2000)
        k: int = Field(default=5, ge=1, le=10)

    class Context(Search):
        max_characters: int = Field(default=4000, ge=256, le=12000)

    class Command(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tool: str = Field(min_length=1, max_length=64)
        arguments: dict = Field(default_factory=dict)

    class Prediction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        equation: str
        arm: str
        seed: StrictInt = Field(ge=0)
        points: list[list[float]] = Field(min_length=1, max_length=1024)

    class ActivePrediction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        equation: str
        points: list[list[float]] = Field(min_length=1, max_length=1024)

    class Question(BaseModel):
        model_config = ConfigDict(extra="forbid")
        question: str = Field(min_length=1, max_length=2000)

    class Registration(BaseModel):
        model_config = ConfigDict(extra="forbid")
        equation: str
        arm: str
        seed: StrictInt = Field(ge=0)
        approved: StrictBool = False

    class Rollback(BaseModel):
        model_config = ConfigDict(extra="forbid")
        equation: str
        approved: StrictBool = False

    return locals()
