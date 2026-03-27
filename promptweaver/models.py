"""Data models for Prompt Weaver."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Component(BaseModel):
    """A single prompt component with optional semantic opposite."""

    word: str
    opposite: Optional[str] = None


class Template(BaseModel):
    """A prompt template with slot placeholders."""

    id: str
    structure: str
    required_components: list[str]
    notes: str = ""


class SlotSpec(BaseModel):
    """Parsed slot specification from a template structure string."""

    category: str
    count: int = 1
    separator: str = " "


class GeneratedPrompt(BaseModel):
    """A fully generated prompt with metadata."""

    hash: str
    template_id: str
    positive: str
    negative: str
    components: dict[str, list[str]]
    created_at: datetime = Field(default_factory=datetime.now)
    favorited: bool = False
