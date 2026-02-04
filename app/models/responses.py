"""Pydantic response models for API endpoints"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class Severity(str, Enum):
    """Severity levels for flags and suggestions"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MetricCategory(str, Enum):
    """Categories for the 8 evaluation metrics"""
    INSTRUCTION_FOLLOWING = "instructionFollowing"
    FACTUAL_ACCURACY = "factualAccuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    WRITING_STYLE_TONE = "writingStyleTone"
    CLARITY = "clarity"
    CONTEXT_AWARENESS = "contextAwareness"
    SAFETY = "safety"


class SubScore(BaseModel):
    """Sub-score within a metric"""
    label: str
    score: float = Field(..., ge=0, le=100)


class Flag(BaseModel):
    """Issue flag identified during evaluation"""
    type: str = Field(..., description="Type of issue (e.g., 'passive_voice', 'bias')")
    message: str = Field(..., description="Description of the issue")
    severity: Severity = Field(..., description="Severity level")
    start_index: Optional[int] = Field(None, alias="startIndex")
    end_index: Optional[int] = Field(None, alias="endIndex")
    
    class Config:
        populate_by_name = True


class Suggestion(BaseModel):
    """Improvement suggestion with actionable items"""
    id: str = Field(..., description="Unique identifier")
    category: MetricCategory = Field(..., description="Related metric category")
    priority: Severity = Field(..., description="Priority level")
    title: str = Field(..., description="Brief title of the suggestion")
    description: str = Field(..., description="Detailed description")
    action_items: Optional[List[str]] = Field(
        None, 
        alias="actionItems",
        description="Specific actions to take"
    )
    
    class Config:
        populate_by_name = True


class DetailedMetric(BaseModel):
    """Detailed metric result with analysis"""
    label: str = Field(..., description="Display name of the metric")
    score: float = Field(..., ge=0, le=100, description="Score from 0-100")
    explanation: str = Field(..., description="Analysis explanation")
    category: Optional[MetricCategory] = Field(None, description="Metric category")
    sub_scores: Optional[List[SubScore]] = Field(
        None,
        alias="subScores",
        description="Sub-component scores"
    )
    suggestions: Optional[List[str]] = Field(
        None,
        description="Specific suggestions for this metric"
    )
    evidence: Optional[List[str]] = Field(
        None,
        description="Specific examples from the content"
    )
    
    class Config:
        populate_by_name = True


class MetricResult(BaseModel):
    """Individual metric evaluation result"""
    score: float = Field(..., ge=0, le=100)
    analysis: str = Field(..., description="Detailed analysis")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    sub_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        alias="subMetrics",
        description="Additional metric-specific data"
    )
    
    class Config:
        populate_by_name = True


class AILikelihood(BaseModel):
    """AI detection result"""
    percentage: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Likelihood percentage (0=AI, 100=human)"
    )
    classification: Literal[
        "Definitely AI",
        "Likely AI", 
        "Likely human-written",
        "Definitely human"
    ] = Field(..., description="Classification label")
    reasoning: str = Field(..., description="Explanation of the assessment")
    linguistic_markers: Optional[List[str]] = Field(
        None,
        alias="linguisticMarkers",
        description="Identified linguistic patterns"
    )
    statistical_analysis: Optional[Dict[str, Any]] = Field(
        None,
        alias="statisticalAnalysis",
        description="Statistical metrics"
    )
    
    class Config:
        populate_by_name = True


class Improvement(BaseModel):
    """Prioritized improvement recommendation"""
    priority: Severity = Field(..., description="Priority level")
    category: str = Field(..., description="Metric category name")
    suggestion: str = Field(..., description="Actionable recommendation")
    impact: str = Field(..., description="Expected improvement description")


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation"""
    content_length: int = Field(..., alias="contentLength", description="Character count")
    word_count: int = Field(..., alias="wordCount")
    paragraph_count: int = Field(..., alias="paragraphCount")
    estimated_reading_time: str = Field(..., alias="estimatedReadingTime")
    processing_time: float = Field(..., alias="processingTime", description="Seconds")
    evaluation_timestamp: datetime = Field(..., alias="evaluationTimestamp")
    
    class Config:
        populate_by_name = True


class MetricsCollection(BaseModel):
    """Collection of all 8 metric results"""
    completeness: MetricResult
    relevance: MetricResult
    clarity: MetricResult
    factual_accuracy: MetricResult = Field(..., alias="factualAccuracy")
    instruction_following: MetricResult = Field(..., alias="instructionFollowing")
    style_tone: MetricResult = Field(..., alias="styleTone")
    context_awareness: MetricResult = Field(..., alias="contextAwareness")
    safety: MetricResult
    
    class Config:
        populate_by_name = True


class EvaluationResponse(BaseModel):
    """Complete evaluation response matching frontend EvaluationResult"""
    
    # Overall assessment
    overall_score: float = Field(
        ...,
        alias="overallScore",
        ge=0,
        le=100,
        description="Weighted overall score"
    )
    summary: str = Field(..., description="2-3 sentence overview")
    
    # AI Detection
    ai_likelihood: float = Field(
        ...,
        alias="aiLikelihood",
        ge=0,
        le=100,
        description="AI likelihood percentage (0=AI, 100=human)"
    )
    
    # Detailed metrics for frontend compatibility
    metrics: List[DetailedMetric] = Field(
        ...,
        description="List of detailed metric results"
    )
    
    # Issues and suggestions
    flags: List[Flag] = Field(default_factory=list, description="Identified issues")
    suggested_changes: List[str] = Field(
        default_factory=list,
        alias="suggestedChanges",
        description="List of suggested changes"
    )
    suggestions: List[Suggestion] = Field(
        default_factory=list,
        description="Structured improvement suggestions"
    )
    
    # Individual metric scores (0-100) for frontend compatibility
    instruction_following: float = Field(..., alias="instructionFollowing", ge=0, le=100)
    factual_accuracy: float = Field(..., alias="factualAccuracy", ge=0, le=100)
    relevance: float = Field(..., ge=0, le=100)
    completeness: float = Field(..., ge=0, le=100)
    writing_style_tone: float = Field(..., alias="writingStyleTone", ge=0, le=100)
    clarity: float = Field(..., ge=0, le=100)
    context_awareness: float = Field(..., alias="contextAwareness", ge=0, le=100)
    safety: float = Field(..., ge=0, le=100)
    
    # Extended response fields (optional)
    strengths: Optional[List[str]] = Field(
        None,
        description="Top 5 performing areas"
    )
    improvements: Optional[List[Improvement]] = Field(
        None,
        description="Prioritized improvement recommendations"
    )
    ai_detection_details: Optional[AILikelihood] = Field(
        None,
        alias="aiDetectionDetails",
        description="Detailed AI detection analysis"
    )
    metadata: Optional[EvaluationMetadata] = Field(
        None,
        description="Evaluation metadata"
    )
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "overallScore": 75.2,
                "summary": "Your content shows strong instruction following and clarity with room for improvement in completeness.",
                "aiLikelihood": 29,
                "metrics": [
                    {
                        "label": "Instruction Following",
                        "score": 93,
                        "explanation": "Content adheres well to the provided tone guidelines.",
                        "category": "instructionFollowing"
                    }
                ],
                "flags": [
                    {
                        "type": "completeness",
                        "message": "Section on 'Advanced Techniques' is underdeveloped",
                        "severity": "MEDIUM"
                    }
                ],
                "suggestedChanges": [
                    "Expand the section on Advanced Techniques",
                    "Add more code examples"
                ],
                "suggestions": [
                    {
                        "id": "sug_1",
                        "category": "completeness",
                        "priority": "HIGH",
                        "title": "Expand Advanced Techniques Section",
                        "description": "The section needs more depth",
                        "actionItems": ["Add 2-3 more examples", "Include best practices"]
                    }
                ],
                "instructionFollowing": 93,
                "factualAccuracy": 87,
                "relevance": 81,
                "completeness": 63,
                "writingStyleTone": 90,
                "clarity": 89,
                "contextAwareness": 83,
                "safety": 95
            }
        }


class AsyncEvaluationResponse(BaseModel):
    """Response for async evaluation submission"""
    job_id: str = Field(..., alias="jobId", description="Unique job identifier")
    status: Literal["processing", "queued"] = Field(..., description="Current status")
    message: Optional[str] = Field(None, description="Additional information")
    
    class Config:
        populate_by_name = True


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str = Field(..., alias="jobId")
    status: Literal["queued", "processing", "completed", "failed"] = Field(...)
    progress: Optional[int] = Field(None, ge=0, le=100, description="Completion percentage")
    result: Optional[EvaluationResponse] = Field(None, description="Result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        populate_by_name = True


class QuickEvaluationResponse(BaseModel):
    """Simplified response for quick evaluation"""
    completeness_score: float = Field(..., alias="completenessScore", ge=0, le=100)
    clarity_score: float = Field(..., alias="clarityScore", ge=0, le=100)
    safety_score: float = Field(..., alias="safetyScore", ge=0, le=100)
    ai_likelihood: float = Field(..., alias="aiLikelihood", ge=0, le=100)
    top_improvements: List[str] = Field(..., alias="topImprovements", max_length=3)
    
    class Config:
        populate_by_name = True


class ReferenceValidationResult(BaseModel):
    """Result for a single reference validation"""
    url: str
    accessible: bool
    response_time_ms: Optional[int] = Field(None, alias="responseTimeMs")
    content_type: Optional[str] = Field(None, alias="contentType")
    credibility_score: Optional[float] = Field(None, alias="credibilityScore", ge=0, le=100)
    key_claims: Optional[List[str]] = Field(None, alias="keyClaims")
    error: Optional[str] = Field(None, description="Error message if not accessible")
    
    class Config:
        populate_by_name = True


class ReferenceValidationResponse(BaseModel):
    """Response for reference validation endpoint"""
    results: List[ReferenceValidationResult]


class MetricDefinition(BaseModel):
    """Definition of a metric for documentation"""
    name: str
    description: str
    target_score: int = Field(..., alias="targetScore")
    weight: float
    
    class Config:
        populate_by_name = True


class MetricsDefinitionResponse(BaseModel):
    """Response for metrics definition endpoint"""
    metrics: List[MetricDefinition]
    scoring_bands: Dict[str, str] = Field(..., alias="scoringBands")
    
    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, str]
    version: str
