"""FastAPI application entry point for Blog Evaluation System"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events"""
    # Startup
    logger.info("Starting Blog Evaluation API...")
    settings = get_settings()
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Model: {settings.llm_model}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Blog Evaluation API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI Blog Evaluation System",
        description="""
## AI-Powered Blog Quality Evaluation

Evaluate AI-generated blogs across 8 quality metrics using specialized AI agents:

### Metrics Evaluated
- **Completeness** (20% weight) - Topic coverage and depth
- **Relevance** - Content alignment with title/topic
- **Clarity** - Readability and structure
- **Factual Accuracy** - Claim verification
- **Instruction Following** - Guideline adherence
- **Style & Tone** - Voice consistency
- **Context Awareness** - Audience appropriateness
- **Safety** - Harmful content detection

### Features
- 🤖 AI likelihood detection
- 📊 Detailed scoring with evidence
- 💡 Actionable improvement suggestions
- 🔗 Reference URL validation
- ⚡ Async processing support

### Architecture
- **3 Specialized AI Agents** orchestrated via LangGraph
- **7-Node Workflow** with parallel execution
- **FastAPI** backend with async support
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1", tags=["Evaluation"])
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "AI Blog Evaluation System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred",
                "error": str(exc) if settings.debug else "Internal server error",
            },
        )
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
