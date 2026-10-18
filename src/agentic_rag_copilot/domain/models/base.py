from pydantic import BaseModel, ConfigDict


class ImmutableModel(BaseModel):
    """Strict, frozen base model for entities and API payloads."""

    model_config = ConfigDict(strict=True, frozen=True)


class MutableModel(BaseModel):
    """Strict, mutable base model for agent state tracking."""

    model_config = ConfigDict(strict=True, frozen=False)
