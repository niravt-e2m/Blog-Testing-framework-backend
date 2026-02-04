"""SQLAlchemy database models for evaluation storage"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    Enum as SQLEnum,
    JSON,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class EvaluationStatus(str, enum.Enum):
    """Status enum for evaluation jobs"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Evaluation(Base):
    """
    Database model for storing evaluation history.
    
    Stores both the input data and complete evaluation results.
    """
    __tablename__ = "evaluations"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Job identifier (for async processing)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        unique=True,
        index=True,
        default=uuid.uuid4,
    )
    
    # Input data
    blog_title: Mapped[str] = mapped_column(Text, nullable=True)
    blog_content: Mapped[str] = mapped_column(Text, nullable=False)
    tone_guidelines: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_links: Mapped[dict] = mapped_column(JSON, default=list)
    target_audience: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Evaluation results
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metric_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    ai_likelihood: Mapped[dict] = mapped_column(JSON, default=dict)
    strengths: Mapped[dict] = mapped_column(JSON, default=list)
    improvements: Mapped[dict] = mapped_column(JSON, default=list)
    
    # Full response (for caching)
    full_response: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Metadata
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Status tracking
    status: Mapped[EvaluationStatus] = mapped_column(
        SQLEnum(EvaluationStatus),
        default=EvaluationStatus.QUEUED,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )
    
    # User tracking (optional, for multi-tenant setups)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_evaluations_status_created", "status", "created_at"),
        Index("ix_evaluations_user_created", "user_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Evaluation(id={self.id}, status={self.status}, score={self.overall_score})>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "blog_title": self.blog_title,
            "overall_score": self.overall_score,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_request(cls, request_data: dict, job_id: Optional[uuid.UUID] = None) -> "Evaluation":
        """Create evaluation model from request data"""
        return cls(
            job_id=job_id or uuid.uuid4(),
            blog_title=request_data.get("blog_title", ""),
            blog_content=request_data.get("blog_content", ""),
            tone_guidelines=request_data.get("tone_guidelines", ""),
            reference_links=request_data.get("reference_links", []),
            target_audience=request_data.get("target_audience", {}),
            status=EvaluationStatus.QUEUED,
        )
    
    def update_with_result(self, result: dict):
        """Update model with evaluation result"""
        self.overall_score = result.get("overallScore")
        self.metric_scores = {
            "completeness": result.get("completeness"),
            "relevance": result.get("relevance"),
            "clarity": result.get("clarity"),
            "factual_accuracy": result.get("factualAccuracy"),
            "instruction_following": result.get("instructionFollowing"),
            "style_tone": result.get("writingStyleTone"),
            "context_awareness": result.get("contextAwareness"),
            "safety": result.get("safety"),
        }
        self.ai_likelihood = result.get("aiDetectionDetails", {})
        self.strengths = result.get("strengths", [])
        self.improvements = result.get("improvements", [])
        self.full_response = result
        self.metadata = result.get("metadata", {})
        self.status = EvaluationStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str):
        """Mark evaluation as failed"""
        self.status = EvaluationStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()


class CachedReference(Base):
    """
    Database model for caching reference URL validation results.
    """
    __tablename__ = "cached_references"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    
    accessible: Mapped[bool] = mapped_column(Boolean, default=False)
    response_time_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_claims: Mapped[dict] = mapped_column(JSON, default=list)
    credibility_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        index=True,
    )
    
    def __repr__(self) -> str:
        return f"<CachedReference(url={self.url[:50]}..., accessible={self.accessible})>"
