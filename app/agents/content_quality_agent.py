"""Content Quality Evaluator Agent - Evaluates completeness, relevance, clarity, factual accuracy"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, LLMResponseError
from app.utils.prompts import PromptTemplates
from app.utils.text_analysis import TextAnalyzer

logger = logging.getLogger(__name__)


class CompletenessResult(BaseModel):
    """Result model for completeness evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    missing_topics: List[str] = Field(default_factory=list)
    underdeveloped_sections: List[str] = Field(default_factory=list)
    coverage_percentage: float = Field(default=0, ge=0, le=100)


class RelevanceResult(BaseModel):
    """Result model for relevance evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    on_topic_percentage: float = Field(default=0, ge=0, le=100)
    drift_sections: List[str] = Field(default_factory=list)


class ClarityResult(BaseModel):
    """Result model for clarity evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    readability_level: str = ""
    flow_issues: List[str] = Field(default_factory=list)


class FactualAccuracyResult(BaseModel):
    """Result model for factual accuracy evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    total_claims: int = 0
    verified_claims: int = 0
    unverified_claims: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)


class ContentQualityResponse(BaseModel):
    """Complete response model for content quality evaluation"""
    completeness: CompletenessResult
    relevance: RelevanceResult
    clarity: ClarityResult
    factual_accuracy: FactualAccuracyResult


class ContentQualityAgent(BaseAgent):
    """
    Agent 1: Content Quality Evaluator
    
    Evaluates blog content across 4 quality metrics:
    - Completeness (Target: 63%) - PRIMARY IMPROVEMENT AREA
    - Relevance (Target: 81%)
    - Clarity (Target: 89%)
    - Factual Accuracy (Target: 87%)
    """
    
    @property
    def agent_name(self) -> str:
        return "ContentQualityAgent"
    
    async def evaluate(
        self,
        blog_title: str,
        blog_content: str,
        reference_links: Optional[List[str]] = None,
        target_audience: str = "General audience",
        reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate content quality across 4 dimensions.
        
        Args:
            blog_title: Title of the blog post
            blog_content: Full blog content
            reference_links: Optional list of reference URLs
            target_audience: Description of target audience
            reference_texts: Optional list of reference text content
            
        Returns:
            Dictionary with completeness, relevance, clarity, and factual_accuracy results
        """
        logger.info(f"{self.agent_name}: Starting content quality evaluation")
        
        # Combine reference links and texts for context
        refs = reference_links or []
        if reference_texts:
            refs.extend([f"[Reference Text]: {text[:500]}..." for text in reference_texts])
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.format_content_quality_prompt(
            blog_title=blog_title,
            blog_content=blog_content,
            reference_links=refs,
            target_audience=target_audience,
        )
        
        # Invoke LLM
        response_text = await self._invoke_llm(system_prompt, user_prompt)
        
        # Parse response
        response_data = self._parse_json_response(response_text)
        
        # Validate and structure response
        validated = self._validate_and_structure_response(response_data, blog_content)
        
        logger.info(
            f"{self.agent_name}: Evaluation complete - "
            f"Completeness: {validated['completeness']['score']}, "
            f"Relevance: {validated['relevance']['score']}, "
            f"Clarity: {validated['clarity']['score']}, "
            f"Factual Accuracy: {validated['factual_accuracy']['score']}"
        )
        
        return validated
    
    def _validate_and_structure_response(
        self,
        response_data: Dict[str, Any],
        blog_content: str,
    ) -> Dict[str, Any]:
        """Validate and enrich response with additional analysis"""
        
        # Get text analysis for clarity metrics
        text_analyzer = TextAnalyzer(blog_content)
        readability = text_analyzer.get_readability_scores()
        
        # Process completeness
        completeness = response_data.get("completeness", {})
        completeness_result = {
            "score": self._normalize_score(completeness.get("score", 50)),
            "analysis": completeness.get("analysis", "Unable to analyze completeness"),
            "missing_topics": completeness.get("missing_topics", []),
            "underdeveloped_sections": completeness.get("underdeveloped_sections", []),
            "coverage_percentage": self._normalize_score(
                completeness.get("coverage_percentage", completeness.get("score", 50))
            ),
            "evidence": self._extract_evidence(completeness.get("analysis", "")),
            "sub_metrics": {
                "topic_coverage": completeness.get("coverage_percentage", 50),
                "depth_score": completeness.get("score", 50),
            }
        }
        
        # Process relevance
        relevance = response_data.get("relevance", {})
        relevance_result = {
            "score": self._normalize_score(relevance.get("score", 50)),
            "analysis": relevance.get("analysis", "Unable to analyze relevance"),
            "on_topic_percentage": self._normalize_score(
                relevance.get("on_topic_percentage", relevance.get("score", 50))
            ),
            "drift_sections": relevance.get("drift_sections", []),
            "evidence": self._extract_evidence(relevance.get("analysis", "")),
            "sub_metrics": {
                "topic_alignment": relevance.get("on_topic_percentage", 50),
                "focus_score": relevance.get("score", 50),
            }
        }
        
        # Process clarity - enhance with computed readability
        clarity = response_data.get("clarity", {})
        clarity_result = {
            "score": self._normalize_score(clarity.get("score", 50)),
            "analysis": clarity.get("analysis", "Unable to analyze clarity"),
            "readability_level": clarity.get("readability_level", readability.readability_level),
            "flow_issues": clarity.get("flow_issues", []),
            "evidence": self._extract_evidence(clarity.get("analysis", "")),
            "sub_metrics": {
                "flesch_reading_ease": readability.flesch_reading_ease,
                "flesch_kincaid_grade": readability.flesch_kincaid_grade,
                "gunning_fog_index": readability.gunning_fog_index,
                "avg_sentence_length": text_analyzer.get_statistics().avg_sentence_length,
            }
        }
        
        # Process factual accuracy
        factual = response_data.get("factual_accuracy", {})
        total_claims = factual.get("total_claims", 0)
        verified_claims = factual.get("verified_claims", 0)
        
        factual_result = {
            "score": self._normalize_score(factual.get("score", 50)),
            "analysis": factual.get("analysis", "Unable to analyze factual accuracy"),
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "unverified_claims": factual.get("unverified_claims", []),
            "contradictions": factual.get("contradictions", []),
            "evidence": self._extract_evidence(factual.get("analysis", "")),
            "sub_metrics": {
                "verification_rate": (verified_claims / max(total_claims, 1)) * 100,
                "claim_density": total_claims,
            }
        }
        
        return {
            "completeness": completeness_result,
            "relevance": relevance_result,
            "clarity": clarity_result,
            "factual_accuracy": factual_result,
        }
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Return default scores for all 4 metrics on failure"""
        default_metric = {
            "score": 50,
            "analysis": "Evaluation failed - using default score",
            "evidence": [],
            "sub_metrics": {},
        }
        
        return {
            "error": True,
            "completeness": {
                **default_metric,
                "missing_topics": [],
                "underdeveloped_sections": [],
                "coverage_percentage": 50,
            },
            "relevance": {
                **default_metric,
                "on_topic_percentage": 50,
                "drift_sections": [],
            },
            "clarity": {
                **default_metric,
                "readability_level": "Unknown",
                "flow_issues": [],
            },
            "factual_accuracy": {
                **default_metric,
                "total_claims": 0,
                "verified_claims": 0,
                "unverified_claims": [],
                "contradictions": [],
            },
        }
