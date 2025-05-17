"""Core type definitions for the Agentic RAG system."""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class ContentType(str, Enum):
    """Types of content that can be processed by the system."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions and outputs."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class DocumentSegment(BaseModel):
    """A segment of a document that can be processed by agents."""
    id: str = Field(..., description="Unique identifier for the segment")
    content: Union[str, bytes] = Field(..., description="The actual content")
    content_type: ContentType = Field(..., description="Type of the content")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    position: int = Field(..., description="Position in the original document")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    requires_human_review: bool = Field(default=False)
    parent_id: Optional[str] = Field(default=None, description="ID of parent segment")
    children: List[str] = Field(default_factory=list, description="IDs of child segments")


class AgentResponse(BaseModel):
    """Response from an agent after processing."""
    agent_id: str = Field(..., description="ID of the agent")
    success: bool = Field(..., description="Whether the processing was successful")
    confidence: ConfidenceLevel = Field(..., description="Confidence in the result")
    result: Optional[Dict] = Field(default=None, description="Processing result")
    error: Optional[str] = Field(default=None, description="Error message if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)


class ProcessingContext(BaseModel):
    """Context maintained throughout the processing pipeline."""
    document_id: str = Field(..., description="ID of the document being processed")
    segments: List[DocumentSegment] = Field(default_factory=list)
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    requires_human_review: bool = Field(default=False)
    status: str = Field(default="initialized") 