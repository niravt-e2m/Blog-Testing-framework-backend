"""Style & Compliance Evaluator Agent - Evaluates instruction following, style/tone, context awareness"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from app.utils.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class ViolationItem(BaseModel):
    """Model for guideline violation"""
    guideline: str
    violation: str


class InconsistencyItem(BaseModel):
    """Model for style inconsistency"""
    section: str
    issue: str


class InstructionFollowingResult(BaseModel):
    """Result model for instruction following evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    guideline_adherence: float = Field(default=0, ge=0, le=100)
    violations: List[ViolationItem] = Field(default_factory=list)
    structural_compliance: bool = True


class StyleToneResult(BaseModel):
    """Result model for style and tone evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    tone_consistency: float = Field(default=0, ge=0, le=100)
    voice_match: str = "Good"
    inconsistencies: List[InconsistencyItem] = Field(default_factory=list)


class ContextAwarenessResult(BaseModel):
    """Result model for context awareness evaluation"""
    score: float = Field(..., ge=0, le=100)
    analysis: str
    audience_alignment: float = Field(default=0, ge=0, le=100)
    terminology_appropriateness: str = ""
    knowledge_level_match: bool = True


class StyleComplianceResponse(BaseModel):
    """Complete response model for style compliance evaluation"""
    instruction_following: InstructionFollowingResult
    style_tone: StyleToneResult
    context_awareness: ContextAwarenessResult


class StyleComplianceAgent(BaseAgent):
    """
    Agent 2: Style & Compliance Evaluator
    
    Evaluates blog content across 3 style metrics:
    - Instruction Following (Target: 93%)
    - Writing Style & Tone (Target: 90%)
    - Context Awareness (Target: 83%)
    """
    
    @property
    def agent_name(self) -> str:
        return "StyleComplianceAgent"
    
    async def evaluate(
        self,
        blog_content: str,
        tone_guidelines: str,
        target_audience: str = "General audience",
        blog_outline: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate style compliance across 3 dimensions.
        
        Args:
            blog_content: Full blog content
            tone_guidelines: Tone of voice guidelines to check against
            target_audience: Description of target audience
            blog_outline: Blog outline/structure for the blog
            
        Returns:
            Dictionary with instruction_following, style_tone, and context_awareness results
        """
        logger.info(f"{self.agent_name}: Starting style compliance evaluation")
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.format_style_compliance_prompt(
            blog_content=blog_content,
            tone_guidelines=tone_guidelines,
            target_audience=target_audience,
            blog_outline=blog_outline,
        )
        
        # Invoke LLM
        response_text = await self._invoke_llm(system_prompt, user_prompt)
        
        # Parse response
        response_data = self._parse_json_response(response_text)
        
        # Validate and structure response
        validated = self._validate_and_structure_response(response_data)
        
        logger.info(
            f"{self.agent_name}: Evaluation complete - "
            f"Instruction Following: {validated['instruction_following']['score']}, "
            f"Style/Tone: {validated['style_tone']['score']}, "
            f"Context Awareness: {validated['context_awareness']['score']}"
        )
        
        return validated
    
    def _validate_and_structure_response(
        self,
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and structure the response data"""
        
        # Process instruction following
        instruction = response_data.get("instruction_following", {})
        violations = instruction.get("violations", [])
        
        # Ensure violations are properly structured
        structured_violations = []
        for v in violations:
            if isinstance(v, dict):
                structured_violations.append({
                    "guideline": v.get("guideline", "Unknown guideline"),
                    "violation": v.get("violation", v.get("description", "Unspecified violation")),
                })
            elif isinstance(v, str):
                structured_violations.append({
                    "guideline": "General",
                    "violation": v,
                })
        
        instruction_result = {
            "score": self._normalize_score(instruction.get("score", 50)),
            "analysis": instruction.get("analysis", "Unable to analyze instruction following"),
            "guideline_adherence": self._normalize_score(
                instruction.get("guideline_adherence", instruction.get("score", 50))
            ),
            "violations": structured_violations,
            "structural_compliance": instruction.get("structural_compliance", True),
            "evidence": self._extract_evidence(instruction.get("analysis", "")),
            "sub_metrics": {
                "adherence_percentage": instruction.get("guideline_adherence", 50),
                "violation_count": len(structured_violations),
            }
        }
        
        # Process style and tone
        style = response_data.get("style_tone", {})
        inconsistencies = style.get("inconsistencies", [])
        
        # Ensure inconsistencies are properly structured
        structured_inconsistencies = []
        for i in inconsistencies:
            if isinstance(i, dict):
                structured_inconsistencies.append({
                    "section": i.get("section", "Unknown section"),
                    "issue": i.get("issue", "Unspecified issue"),
                })
            elif isinstance(i, str):
                structured_inconsistencies.append({
                    "section": "General",
                    "issue": i,
                })
        
        style_result = {
            "score": self._normalize_score(style.get("score", 50)),
            "analysis": style.get("analysis", "Unable to analyze style and tone"),
            "tone_consistency": self._normalize_score(
                style.get("tone_consistency", style.get("score", 50))
            ),
            "voice_match": style.get("voice_match", "Good"),
            "inconsistencies": structured_inconsistencies,
            "evidence": self._extract_evidence(style.get("analysis", "")),
            "sub_metrics": {
                "consistency_score": style.get("tone_consistency", 50),
                "inconsistency_count": len(structured_inconsistencies),
            }
        }
        
        # Process context awareness
        context = response_data.get("context_awareness", {})
        context_result = {
            "score": self._normalize_score(context.get("score", 50)),
            "analysis": context.get("analysis", "Unable to analyze context awareness"),
            "audience_alignment": self._normalize_score(
                context.get("audience_alignment", context.get("score", 50))
            ),
            "terminology_appropriateness": context.get(
                "terminology_appropriateness", "Appropriate"
            ),
            "knowledge_level_match": context.get("knowledge_level_match", True),
            "evidence": self._extract_evidence(context.get("analysis", "")),
            "sub_metrics": {
                "audience_score": context.get("audience_alignment", 50),
                "terminology_score": 80 if context.get("knowledge_level_match", True) else 60,
            }
        }
        
        return {
            "instruction_following": instruction_result,
            "style_tone": style_result,
            "context_awareness": context_result,
        }
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Return default scores for all 3 metrics on failure"""
        default_metric = {
            "score": 50,
            "analysis": "Evaluation failed - using default score",
            "evidence": [],
            "sub_metrics": {},
        }
        
        return {
            "error": True,
            "instruction_following": {
                **default_metric,
                "guideline_adherence": 50,
                "violations": [],
                "structural_compliance": True,
            },
            "style_tone": {
                **default_metric,
                "tone_consistency": 50,
                "voice_match": "Unknown",
                "inconsistencies": [],
            },
            "context_awareness": {
                **default_metric,
                "audience_alignment": 50,
                "terminology_appropriateness": "Unknown",
                "knowledge_level_match": True,
            },
        }
