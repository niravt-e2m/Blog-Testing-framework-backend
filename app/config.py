"""Configuration management for the Blog Evaluation System"""

from functools import lru_cache
from typing import List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM Provider Configuration (OpenAI or OpenRouter)
    llm_provider: Literal["openai", "openrouter"] = "openrouter"
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Database Configuration
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/blog_evaluator"
    database_sync_url: str = "postgresql://user:password@localhost:5432/blog_evaluator"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Application Configuration
    app_env: Literal["development", "production", "testing"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = '["http://localhost:3000", "http://localhost:5173"]'
    
    # Rate Limiting
    rate_limit_per_hour: int = 100
    
    # Processing Limits
    max_content_length: int = 50000
    evaluation_timeout: int = 180
    reference_validation_timeout: int = 10
    
    # Metric Weights
    weight_completeness: float = 0.20
    weight_relevance: float = 0.1143
    weight_clarity: float = 0.1143
    weight_factual_accuracy: float = 0.1143
    weight_instruction_following: float = 0.1143
    weight_style_tone: float = 0.1143
    weight_context_awareness: float = 0.1143
    weight_safety: float = 0.1143
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string or comma-separated list"""
        if not self.cors_origins:
            return ["*"]

        try:
            # Try parsing as JSON first
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            # Fallback to comma-separated string
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @property
    def metric_weights(self) -> dict:
        """Return all metric weights as a dictionary"""
        return {
            "completeness": self.weight_completeness,
            "relevance": self.weight_relevance,
            "clarity": self.weight_clarity,
            "factual_accuracy": self.weight_factual_accuracy,
            "instruction_following": self.weight_instruction_following,
            "style_tone": self.weight_style_tone,
            "context_awareness": self.weight_context_awareness,
            "safety": self.weight_safety,
        }
    
    @property
    def llm_api_key(self) -> str:
        """Get the appropriate API key based on provider"""
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key
    
    @property
    def llm_model(self) -> str:
        """Get the appropriate model based on provider"""
        if self.llm_provider == "openrouter":
            return self.openrouter_model
        return self.openai_model
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Target scores for each metric (for reference)
TARGET_SCORES = {
    "completeness": 63,
    "relevance": 81,
    "clarity": 89,
    "factual_accuracy": 87,
    "instruction_following": 93,
    "style_tone": 90,
    "context_awareness": 83,
    "safety": 95,
}

# Score classification bands
SCORE_BANDS = {
    "excellent": (90, 100),
    "good": (80, 89),
    "adequate": (70, 79),
    "below_standard": (60, 69),
    "poor": (0, 59),
}


def classify_score(score: float) -> str:
    """Classify a score into a performance band"""
    for band_name, (low, high) in SCORE_BANDS.items():
        if low <= score <= high:
            return band_name
    return "poor"
