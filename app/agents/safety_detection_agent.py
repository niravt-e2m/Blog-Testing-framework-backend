"""Safety & AI Detection Analyzer Agent - Evaluates safety and AI likelihood"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from app.utils.prompts import PromptTemplates
from app.utils.text_analysis import TextAnalyzer

logger = logging.getLogger(__name__)


class SafetyFlag(BaseModel):
    """Model for safety flag"""
    category: str
    severity: str  # HIGH, MEDIUM, LOW
    description: str


class StatisticalAnalysis(BaseModel):
    """Model for statistical AI detection analysis"""
    sentence_length_variance: str
    vocabulary_diversity: str
    pattern_score: str


class SafetyResult(BaseModel):
    """Result model for safety evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    is_safe: bool = True
    flags: List[SafetyFlag] = Field(default_factory=list)
    biases_detected: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AILikelihoodResult(BaseModel):
    """Result model for AI likelihood detection"""
    percentage: float = Field(..., ge=0, le=100)
    classification: str  # "Definitely AI" | "Likely AI" | "Likely human-written" | "Definitely human"
    reasoning: str
    linguistic_markers: List[str] = Field(default_factory=list)
    statistical_analysis: Optional[StatisticalAnalysis] = None


class SafetyDetectionResponse(BaseModel):
    """Complete response model for safety and AI detection"""
    safety: SafetyResult
    ai_likelihood: AILikelihoodResult


class SafetyDetectionAgent(BaseAgent):
    """
    Agent 3: Safety & AI Detection Analyzer
    
    Evaluates blog content for:
    - Safety (Target: 95%) - harmful content, biases, misinformation
    - AI Likelihood Detection - probability content is AI-generated vs human-written
    """
    
    @property
    def agent_name(self) -> str:
        return "SafetyDetectionAgent"
    
    async def evaluate(
        self,
        blog_content: str,
        target_audience: str = "General audience",
        include_ai_detection: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate content safety and detect AI likelihood.
        
        Args:
            blog_content: Full blog content
            target_audience: Description of target audience
            include_ai_detection: Whether to include AI detection analysis
            
        Returns:
            Dictionary with safety and ai_likelihood results
        """
        logger.info(f"{self.agent_name}: Starting safety and AI detection evaluation")
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.format_safety_detection_prompt(
            blog_content=blog_content,
            target_audience=target_audience,
        )
        
        # Invoke LLM
        response_text = await self._invoke_llm(system_prompt, user_prompt)
        
        # Parse response
        response_data = self._parse_json_response(response_text)
        
        # Get text analysis for statistical AI detection metrics
        text_analyzer = TextAnalyzer(blog_content)
        ai_patterns = text_analyzer.detect_ai_patterns()
        
        # Validate and structure response
        validated = self._validate_and_structure_response(
            response_data, 
            ai_patterns,
            include_ai_detection
        )
        
        logger.info(
            f"{self.agent_name}: Evaluation complete - "
            f"Safety: {validated['safety']['score']}, "
            f"AI Likelihood: {validated['ai_likelihood']['percentage']}%"
        )
        
        return validated
    
    def _validate_and_structure_response(
        self,
        response_data: Dict[str, Any],
        ai_patterns: Dict[str, Any],
        include_ai_detection: bool,
    ) -> Dict[str, Any]:
        """Validate and structure the response data"""
        
        # Process safety
        safety = response_data.get("safety", {})
        flags = safety.get("flags", [])
        
        # Ensure flags are properly structured
        structured_flags = []
        for f in flags:
            if isinstance(f, dict):
                structured_flags.append({
                    "category": f.get("category", "General"),
                    "severity": f.get("severity", "MEDIUM").upper(),
                    "description": f.get("description", "Unspecified issue"),
                })
            elif isinstance(f, str):
                structured_flags.append({
                    "category": "General",
                    "severity": "MEDIUM",
                    "description": f,
                })
        
        # Determine if content is safe based on flags
        high_severity_flags = [f for f in structured_flags if f["severity"] == "HIGH"]
        is_safe = len(high_severity_flags) == 0
        
        safety_result = {
            "score": self._normalize_score(safety.get("score", 95)),
            "analysis": safety.get("analysis", "No safety issues identified"),
            "is_safe": safety.get("is_safe", is_safe),
            "flags": structured_flags,
            "biases_detected": safety.get("biases_detected", []),
            "recommendations": safety.get("recommendations", []),
            "evidence": self._extract_evidence(safety.get("analysis", "")),
            "sub_metrics": {
                "flag_count": len(structured_flags),
                "high_severity_count": len(high_severity_flags),
                "bias_count": len(safety.get("biases_detected", [])),
            }
        }
        
        # Process AI likelihood
        ai = response_data.get("ai_likelihood", {})
        
        # Get percentage (note: 0 = AI, 100 = human in our scale)
        percentage = self._normalize_score(ai.get("percentage", 50))
        
        # Classify based on percentage
        classification = ai.get("classification", self._classify_ai_likelihood(percentage))
        
        # Get statistical analysis from LLM or compute
        stat_analysis = ai.get("statistical_analysis", {})
        if not stat_analysis or not isinstance(stat_analysis, dict):
            stat_analysis = {
                "sentence_length_variance": self._describe_variance(
                    ai_patterns.get("sentence_length_uniformity", 0.5)
                ),
                "vocabulary_diversity": self._describe_diversity(
                    ai_patterns.get("vocabulary_diversity", 0.5)
                ),
                "pattern_score": self._describe_pattern_score(
                    ai_patterns.get("transition_phrase_ratio", 0)
                ),
            }
        
        ai_result = {
            "percentage": percentage,
            "classification": classification,
            "reasoning": ai.get("reasoning", "Unable to determine AI likelihood"),
            "linguistic_markers": ai.get("linguistic_markers", []),
            "statistical_analysis": stat_analysis,
            "computed_metrics": {
                "transition_phrase_ratio": ai_patterns.get("transition_phrase_ratio", 0),
                "sentence_uniformity": ai_patterns.get("sentence_length_uniformity", 0),
                "vocabulary_diversity": ai_patterns.get("vocabulary_diversity", 0),
                "high_freq_word_count": ai_patterns.get("high_frequency_word_count", 0),
            }
        }
        
        return {
            "safety": safety_result,
            "ai_likelihood": ai_result,
        }
    
    def _classify_ai_likelihood(self, percentage: float) -> str:
        """Classify AI likelihood based on percentage"""
        if percentage <= 20:
            return "Definitely AI"
        elif percentage <= 40:
            return "Likely AI"
        elif percentage <= 70:
            return "Likely human-written"
        else:
            return "Definitely human"
    
    def _describe_variance(self, uniformity: float) -> str:
        """Describe sentence length variance"""
        if uniformity > 0.8:
            return "Very uniform (AI indicator)"
        elif uniformity > 0.6:
            return "Somewhat uniform"
        elif uniformity > 0.4:
            return "Moderate variation"
        else:
            return "High variation (human indicator)"
    
    def _describe_diversity(self, diversity: float) -> str:
        """Describe vocabulary diversity"""
        if diversity < 0.3:
            return "Low diversity (AI indicator)"
        elif diversity < 0.5:
            return "Moderate diversity"
        elif diversity < 0.7:
            return "Good diversity"
        else:
            return "High diversity (human indicator)"
    
    def _describe_pattern_score(self, ratio: float) -> str:
        """Describe pattern detection score"""
        if ratio > 2:
            return "High AI pattern density"
        elif ratio > 1:
            return "Moderate pattern presence"
        elif ratio > 0.5:
            return "Low pattern presence"
        else:
            return "Minimal patterns detected"
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Return default scores on failure"""
        return {
            "error": True,
            "safety": {
                "score": 50,
                "analysis": "Evaluation failed - using default score",
                "is_safe": True,
                "flags": [],
                "biases_detected": [],
                "recommendations": [],
                "evidence": [],
                "sub_metrics": {},
            },
            "ai_likelihood": {
                "percentage": 50,
                "classification": "Likely human-written",
                "reasoning": "Evaluation failed - unable to determine",
                "linguistic_markers": [],
                "statistical_analysis": {
                    "sentence_length_variance": "Unknown",
                    "vocabulary_diversity": "Unknown",
                    "pattern_score": "Unknown",
                },
                "computed_metrics": {},
            },
        }
