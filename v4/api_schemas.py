"""Public API request contracts; imported only when serving or testing the API."""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Prediction(Request):
    trial_id: str = Field(pattern="^[0-9a-f]{16}$")
    coordinates: list[list[float]] = Field(min_length=1, max_length=1024)


class Question(Request):
    question: str = Field(min_length=1, max_length=2000)
    use_llm: bool = False
    dense: bool = False
    backend: Literal["numpy", "faiss"] = "numpy"


class Registration(Request):
    equation: str
    trial_id: str = Field(pattern="^[0-9a-f]{16}$")
    approved: bool = False


class Tool(Request):
    name: Literal[
        "comparison",
        "models",
        "model_info",
        "predict",
        "search",
        "generate_report",
        "generate_feature_catalog",
    ]
    arguments: dict = Field(default_factory=dict)
    approved: bool = False
