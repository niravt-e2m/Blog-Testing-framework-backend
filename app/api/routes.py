"""FastAPI routes for blog evaluation API"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.models.requests import (
    EvaluationRequest,
    QuickEvaluationRequest,
    ReferenceValidationRequest,
)
from app.models.responses import (
    EvaluationResponse,
    AsyncEvaluationResponse,
    JobStatusResponse,
    QuickEvaluationResponse,
    ReferenceValidationResponse,
    ReferenceValidationResult,
    MetricsDefinitionResponse,
    MetricDefinition,
    HealthResponse,
)
from app.workflows.evaluation_workflow import run_evaluation, EvaluationWorkflow
from app.services.reference_validator import ReferenceValidator
from app.config import get_settings, TARGET_SCORES, SCORE_BANDS

logger = logging.getLogger(__name__)

router = APIRouter()

# WARNING: In-memory job storage - jobs will be lost on server restart!
# For production, replace with Redis or database storage:
#   - Redis: pip install redis, use redis-py client
#   - Database: Use SQLAlchemy with PostgreSQL/MySQL
# Example Redis job storage pattern:
#   redis_client = redis.Redis.from_url(settings.redis_url)
#   job_storage_prefix = "blogeval_job:"
job_storage: dict = {}


@router.post(
    "/evaluate",
    response_model=Union[EvaluationResponse, AsyncEvaluationResponse],
    response_model_exclude_unset=True,
    summary="Evaluate Blog Content",
    description="Perform comprehensive evaluation of blog content across 8 quality metrics.",
    response_description="Complete evaluation results or job ID for async processing",
)
async def evaluate_blog(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
):
    """
    POST /evaluate - Complete blog evaluation
    
    Evaluates blog content across 8 metrics:
    - Completeness, Relevance, Clarity, Factual Accuracy (Content Quality)
    - Instruction Following, Style & Tone, Context Awareness (Style Compliance)
    - Safety (Safety Detection)
    
    Also includes AI likelihood detection.
    """
    logger.info(f"Received evaluation request for: {request.title[:50]}...")
    
    # Check for async mode
    options = request.evaluation_options
    if options and options.async_mode:
        # Generate job ID and queue for background processing
        job_id = str(uuid.uuid4())
        job_storage[job_id] = {
            "status": "processing",
            "progress": 0,
            "created_at": datetime.utcnow().isoformat(),
            "request": request.model_dump(),
        }
        
        # Add to background tasks
        background_tasks.add_task(
            process_evaluation_async,
            job_id=job_id,
            request=request,
        )
        
        return AsyncEvaluationResponse(
            job_id=job_id,
            status="processing",
            message="Evaluation started. Poll /evaluate/{job_id} for results.",
        )
    
    # Synchronous processing
    try:
        result = await run_evaluation(
            blog_content=request.content,
            tone_guidelines=request.tone_content,
            target_audience=request.target_audience,
            blog_title=request.title,
            reference_links=request.reference_urls,
            reference_texts=request.reference_texts,
            blog_outline=request.outline,
            options={
                "include_ai_detection": options.include_ai_detection if options else True,
                "depth_level": options.depth_level if options else "standard",
            },
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


async def process_evaluation_async(job_id: str, request: EvaluationRequest):
    """Background task for async evaluation processing"""
    try:
        job_storage[job_id]["progress"] = 10
        
        options = request.evaluation_options
        result = await run_evaluation(
            blog_content=request.content,
            tone_guidelines=request.tone_content,
            target_audience=request.target_audience,
            blog_title=request.title,
            reference_links=request.reference_urls,
            reference_texts=request.reference_texts,
            blog_outline=request.outline,
            options={
                "include_ai_detection": options.include_ai_detection if options else True,
                "depth_level": options.depth_level if options else "standard",
            },
        )
        
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["progress"] = 100
        job_storage[job_id]["result"] = result
        job_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        logger.error(f"Async evaluation failed for job {job_id}: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)


@router.get(
    "/evaluate/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Evaluation Status",
    description="Retrieve the status and results of an async evaluation job.",
)
async def get_evaluation_status(job_id: str):
    """
    GET /evaluate/{job_id} - Retrieve async evaluation results
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.post(
    "/evaluate/quick",
    response_model=QuickEvaluationResponse,
    summary="Quick Evaluation",
    description="Rapid assessment with 3 core metrics only (Completeness, Clarity, Safety).",
)
async def quick_evaluate(request: QuickEvaluationRequest):
    """
    POST /evaluate/quick - Rapid assessment (simplified)
    
    Returns only:
    - Completeness score
    - Clarity score  
    - Safety score
    - Basic AI likelihood
    - Top 3 improvements
    """
    logger.info("Received quick evaluation request")
    
    try:
        # Run simplified evaluation
        result = await run_evaluation(
            blog_content=request.blog_text,
            tone_guidelines=request.tone_of_voice or "",
            target_audience=request.target_audience or "General audience",
            options={
                "depth_level": "fast",
                "include_ai_detection": True,
            },
        )
        
        # Extract quick metrics
        return QuickEvaluationResponse(
            completeness_score=result.get("completeness", 50),
            clarity_score=result.get("clarity", 50),
            safety_score=result.get("safety", 50),
            ai_likelihood=result.get("aiLikelihood", 50),
            top_improvements=result.get("suggestedChanges", [])[:3],
        )
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Quick evaluation failed: {str(e)}"
        )


@router.post(
    "/validate/references",
    response_model=ReferenceValidationResponse,
    summary="Validate References",
    description="Standalone reference URL validation and content extraction.",
)
async def validate_references(request: ReferenceValidationRequest):
    """
    POST /validate/references - Standalone reference validation
    """
    logger.info(f"Validating {len(request.urls)} references")
    
    try:
        validator = ReferenceValidator()
        results = await validator.validate_urls(request.urls)
        
        return ReferenceValidationResponse(
            results=[
                ReferenceValidationResult(
                    url=r.url,
                    accessible=r.accessible,
                    response_time_ms=r.response_time_ms,
                    content_type=r.content_type,
                    credibility_score=r.credibility_score,
                    key_claims=r.key_claims,
                    error=r.error,
                )
                for r in results
            ]
        )
        
    except Exception as e:
        logger.error(f"Reference validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reference validation failed: {str(e)}"
        )


@router.get(
    "/metrics/definitions",
    response_model=MetricsDefinitionResponse,
    summary="Get Metrics Definitions",
    description="Documentation endpoint with metric definitions and scoring bands.",
)
async def get_metrics_definitions():
    """
    GET /metrics/definitions - Documentation endpoint
    """
    settings = get_settings()
    weights = settings.metric_weights
    
    metrics = [
        MetricDefinition(
            name="completeness",
            description="Measures whether all expected topics are adequately covered",
            target_score=TARGET_SCORES["completeness"],
            weight=weights["completeness"],
        ),
        MetricDefinition(
            name="relevance",
            description="Assesses how well content aligns with the blog title and topic",
            target_score=TARGET_SCORES["relevance"],
            weight=weights["relevance"],
        ),
        MetricDefinition(
            name="clarity",
            description="Evaluates writing clarity, structure, and readability",
            target_score=TARGET_SCORES["clarity"],
            weight=weights["clarity"],
        ),
        MetricDefinition(
            name="factual_accuracy",
            description="Checks factual claims against provided references",
            target_score=TARGET_SCORES["factual_accuracy"],
            weight=weights["factual_accuracy"],
        ),
        MetricDefinition(
            name="instruction_following",
            description="Measures adherence to provided tone and style guidelines",
            target_score=TARGET_SCORES["instruction_following"],
            weight=weights["instruction_following"],
        ),
        MetricDefinition(
            name="style_tone",
            description="Assesses writing style consistency and tone matching",
            target_score=TARGET_SCORES["style_tone"],
            weight=weights["style_tone"],
        ),
        MetricDefinition(
            name="context_awareness",
            description="Evaluates appropriateness for target audience",
            target_score=TARGET_SCORES["context_awareness"],
            weight=weights["context_awareness"],
        ),
        MetricDefinition(
            name="safety",
            description="Checks for harmful content, biases, and safety concerns",
            target_score=TARGET_SCORES["safety"],
            weight=weights["safety"],
        ),
    ]
    
    scoring_bands = {
        band: f"{low}-{high}"
        for band, (low, high) in SCORE_BANDS.items()
    }
    
    return MetricsDefinitionResponse(
        metrics=metrics,
        scoring_bands=scoring_bands,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and dependent services health status.",
)
async def health_check():
    """
    GET /health - Health check endpoint
    """
    settings = get_settings()
    
    services = {
        "api": "healthy",
        "llm_provider": "available" if settings.llm_api_key else "not_configured",
    }
    
    # Check database connection (if configured)
    if settings.database_url:
        services["database"] = "configured"
    else:
        services["database"] = "not_configured"
    
    # Check Redis (if configured)
    if settings.redis_url:
        services["redis"] = "configured"
    else:
        services["redis"] = "not_configured"
    
    # Determine overall status
    if all(v in ["healthy", "available", "configured"] for v in services.values()):
        status = "healthy"
    elif services["api"] == "healthy" and services["llm_provider"] == "available":
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        services=services,
        version="1.0.0",
    )
