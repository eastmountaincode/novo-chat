from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GatewayOperation(str, Enum):
    ASK = "ask"
    INDEX_REBUILD = "index_rebuild"
    MODEL_START = "model_start"
    MODEL_STOP = "model_stop"


class GatewayJobRequest(BaseModel):
    """Intentionally closed browser-to-gateway command vocabulary."""

    model_config = ConfigDict(extra="forbid")

    operation: GatewayOperation
    corpus: str | None = Field(
        default=None,
        max_length=197,
        pattern=r"^novo:[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$",
    )
    question: str | None = Field(default=None, min_length=1, max_length=20_000)
    retrieval_question: str | None = Field(default=None, max_length=30_000)
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
    strategy: str = Field(default="hybrid", pattern=r"^(auto|hybrid|fts)$")
    max_sources: int = Field(default=6, ge=1, le=100)
    retrieval_top_k: int = Field(default=16, ge=1, le=32)
    force: bool = False

    @model_validator(mode="after")
    def required_fields_for_operation(self) -> "GatewayJobRequest":
        if self.operation is GatewayOperation.ASK:
            if not self.corpus or not self.question or not self.model:
                raise ValueError("ask requires corpus, question, and model")
        elif self.operation is GatewayOperation.INDEX_REBUILD:
            if not self.corpus:
                raise ValueError("index_rebuild requires corpus")
        elif not self.model:
            raise ValueError(f"{self.operation.value} requires model")
        return self


class WorkerAvailability(BaseModel):
    available: bool
    detail: str = ""
    capabilities: dict[str, Any] = Field(default_factory=dict)
    modelStatus: dict[str, Any] = Field(default_factory=dict)


class GatewayIndexStatusItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notebookId: str = Field(min_length=1, max_length=192)
    notebookName: str = Field(default="", max_length=4096)
    contentRevision: str = Field(min_length=1, max_length=256)
    exactReady: bool
    activatedAt: str | None = Field(default=None, max_length=64)
    chunkCount: int | None = Field(default=None, ge=0)


class GatewayIndexStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: str = Field(min_length=1, max_length=197)
    exactReady: bool
    indexes: list[GatewayIndexStatusItem] = Field(min_length=1, max_length=256)


class JobAccepted(BaseModel):
    jobId: str
    state: str
    operation: str


class JobPublicError(BaseModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    message: str = Field(max_length=2000)
    retryable: bool = False


class JobPublicStatus(BaseModel):
    jobId: str
    state: str
    operation: str
    progress: Any | None = None
    error: JobPublicError | None = None
    createdAt: str | None = None
    updatedAt: str | None = None
