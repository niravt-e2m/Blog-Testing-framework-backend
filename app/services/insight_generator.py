"""Insight generation service for improvement recommendations"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from app.services.aggregator import MetricScore, ScoreAggregator

logger = logging.getLogger(__name__)


@dataclass
class Improvement:
    """Structured improvement recommendation"""
    priority: str  # HIGH, MEDIUM, LOW
    category: str
    suggestion: str
    impact: str
    action_items: Optional[List[str]] = None


@dataclass
class Flag:
    """Issue flag identified in content"""
    type: str
    message: str
    severity: str  # HIGH, MEDIUM, LOW
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class InsightGenerator:
    """
    Service for generating actionable improvement suggestions.
    
    Responsibilities:
    - Generate prioritized improvement recommendations
    - Create specific action items for each metric
    - Estimate improvement impact
    - Convert agent results to user-friendly flags and suggestions
    """
    
    # Suggestion templates for each metric
    SUGGESTION_TEMPLATES = {
        "completeness": {
            "high": [
                "Add sections covering: {missing_topics}",
                "Expand the depth of coverage in key areas",
                "Include more examples and supporting details",
            ],
            "medium": [
                "Consider adding more context to: {underdeveloped}",
                "Strengthen supporting arguments with additional evidence",
            ],
            "low": [
                "Fine-tune coverage of minor topics for thoroughness",
            ],
        },
        "relevance": {
            "high": [
                "Remove or revise off-topic sections: {drift_sections}",
                "Refocus content to align with the main topic",
                "Ensure all examples directly support the main thesis",
            ],
            "medium": [
                "Tighten connections between supporting points and main topic",
                "Review tangential content for necessity",
            ],
            "low": [
                "Minor refinements to maintain strict topic focus",
            ],
        },
        "clarity": {
            "high": [
                "Simplify complex sentences for better readability",
                "Add transition phrases between paragraphs",
                "Break up long paragraphs into digestible chunks",
            ],
            "medium": [
                "Review sentence structure variety",
                "Clarify technical terms for your audience",
            ],
            "low": [
                "Minor polish to sentence flow",
            ],
        },
        "factual_accuracy": {
            "high": [
                "Add citations for claims: {unverified}",
                "Verify and correct any factual inconsistencies",
                "Include source references for statistics",
            ],
            "medium": [
                "Strengthen evidence for key assertions",
                "Cross-reference claims with authoritative sources",
            ],
            "low": [
                "Add supplementary references for completeness",
            ],
        },
        "instruction_following": {
            "high": [
                "Review and address guideline violations: {violations}",
                "Realign content structure with requirements",
            ],
            "medium": [
                "Fine-tune adherence to tone guidelines",
                "Verify all structural requirements are met",
            ],
            "low": [
                "Minor adjustments to perfectly match guidelines",
            ],
        },
        "style_tone": {
            "high": [
                "Revise sections with tone inconsistencies: {inconsistencies}",
                "Establish and maintain a consistent voice throughout",
            ],
            "medium": [
                "Smooth out occasional tone shifts",
                "Align vocabulary with target style",
            ],
            "low": [
                "Polish for stylistic perfection",
            ],
        },
        "context_awareness": {
            "high": [
                "Simplify terminology for your target audience",
                "Adjust assumed knowledge level",
                "Add explanations for technical concepts",
            ],
            "medium": [
                "Review jargon usage for audience appropriateness",
                "Consider adding context for industry-specific terms",
            ],
            "low": [
                "Fine-tune language complexity for optimal engagement",
            ],
        },
        "safety": {
            "high": [
                "Address identified safety concerns immediately",
                "Remove or revise potentially harmful content",
                "Review for unintended biases: {biases}",
            ],
            "medium": [
                "Review flagged content for appropriateness",
                "Ensure balanced perspective on sensitive topics",
            ],
            "low": [
                "Minor sensitivity review for best practices",
            ],
        },
    }
    
    IMPACT_DESCRIPTIONS = {
        "completeness": "Improving completeness will provide readers with comprehensive coverage and establish authority on the topic.",
        "relevance": "Better relevance will keep readers engaged and improve content focus.",
        "clarity": "Enhanced clarity will make your content more accessible and improve reader comprehension.",
        "factual_accuracy": "Stronger factual accuracy builds trust and credibility with your audience.",
        "instruction_following": "Better guideline adherence ensures content meets expectations and requirements.",
        "style_tone": "Consistent style creates a professional impression and better reading experience.",
        "context_awareness": "Improved context awareness ensures content resonates with your target audience.",
        "safety": "Addressing safety concerns protects both your audience and your reputation.",
    }
    
    def __init__(self, aggregator: Optional[ScoreAggregator] = None):
        """Initialize with optional aggregator instance"""
        self.aggregator = aggregator or ScoreAggregator()
    
    def generate_improvements(
        self,
        metrics: Dict[str, MetricScore],
        agent_results: Dict[str, Any],
        max_improvements: int = 10,
    ) -> List[Improvement]:
        """
        Generate prioritized improvement recommendations.
        
        Args:
            metrics: Dictionary of MetricScore objects
            agent_results: Raw results from all agents
            max_improvements: Maximum number of improvements to return
            
        Returns:
            List of Improvement objects sorted by priority
        """
        improvements = []
        
        for metric_name, metric in metrics.items():
            priority = self.aggregator.determine_priority(metric.score)
            
            # Get specific issues from agent results
            issues = self._extract_issues(metric_name, agent_results)
            
            # Generate suggestions based on priority
            suggestions = self._create_suggestions(
                metric_name,
                priority,
                issues,
            )
            
            for suggestion in suggestions:
                improvements.append(Improvement(
                    priority=priority,
                    category=metric_name,
                    suggestion=suggestion,
                    impact=self._estimate_impact(metric_name, metric.score),
                    action_items=self._create_action_items(metric_name, issues),
                ))
        
        # Sort by priority and limit
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        improvements.sort(key=lambda x: (priority_order.get(x.priority, 2), -len(x.suggestion)))
        
        return improvements[:max_improvements]
    
    def _extract_issues(
        self,
        metric_name: str,
        agent_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract specific issues from agent results for a metric"""
        issues = {}
        
        # Content Quality Agent results
        cq = agent_results.get("content_quality", {})
        if metric_name == "completeness":
            issues["missing_topics"] = cq.get("completeness", {}).get("missing_topics", [])
            issues["underdeveloped"] = cq.get("completeness", {}).get("underdeveloped_sections", [])
        elif metric_name == "relevance":
            issues["drift_sections"] = cq.get("relevance", {}).get("drift_sections", [])
        elif metric_name == "clarity":
            issues["flow_issues"] = cq.get("clarity", {}).get("flow_issues", [])
        elif metric_name == "factual_accuracy":
            issues["unverified"] = cq.get("factual_accuracy", {}).get("unverified_claims", [])
            issues["contradictions"] = cq.get("factual_accuracy", {}).get("contradictions", [])
        
        # Style Compliance Agent results
        sc = agent_results.get("style_compliance", {})
        if metric_name == "instruction_following":
            issues["violations"] = sc.get("instruction_following", {}).get("violations", [])
        elif metric_name == "style_tone":
            issues["inconsistencies"] = sc.get("style_tone", {}).get("inconsistencies", [])
        elif metric_name == "context_awareness":
            issues["terminology"] = sc.get("context_awareness", {}).get("terminology_appropriateness", "")
        
        # Safety Detection Agent results
        sd = agent_results.get("safety_detection", {})
        if metric_name == "safety":
            issues["flags"] = sd.get("safety", {}).get("flags", [])
            issues["biases"] = sd.get("safety", {}).get("biases_detected", [])
        
        return issues
    
    def _create_suggestions(
        self,
        metric_name: str,
        priority: str,
        issues: Dict[str, Any],
    ) -> List[str]:
        """Create specific suggestions based on metric and issues"""
        templates = self.SUGGESTION_TEMPLATES.get(metric_name, {})
        priority_key = priority.lower()
        template_list = templates.get(priority_key, ["Improve this area for better results"])
        
        suggestions = []
        for template in template_list:
            # Format template with specific issues
            suggestion = template
            
            # Replace placeholders with actual issues
            for key, value in issues.items():
                placeholder = "{" + key + "}"
                if placeholder in suggestion:
                    if isinstance(value, list) and value:
                        formatted = ", ".join(str(v)[:50] for v in value[:3])
                        suggestion = suggestion.replace(placeholder, formatted)
                    elif isinstance(value, str) and value:
                        suggestion = suggestion.replace(placeholder, value[:100])
                    else:
                        # Remove placeholder if no data
                        suggestion = suggestion.replace(f": {placeholder}", "")
                        suggestion = suggestion.replace(placeholder, "identified areas")
            
            suggestions.append(suggestion)
        
        return suggestions[:2]  # Limit suggestions per metric
    
    def _estimate_impact(self, metric_name: str, current_score: float) -> str:
        """Estimate the impact of improving this metric"""
        base_impact = self.IMPACT_DESCRIPTIONS.get(
            metric_name,
            "Improving this area will enhance overall content quality."
        )
        
        # Calculate potential score improvement
        target = self.aggregator.targets.get(metric_name, 80)
        gap = target - current_score
        
        if gap > 20:
            impact_level = f"Could improve overall score by 3-5 points. "
        elif gap > 10:
            impact_level = f"Could improve overall score by 1-3 points. "
        else:
            impact_level = "Minor improvement to overall score. "
        
        return impact_level + base_impact
    
    def _create_action_items(
        self,
        metric_name: str,
        issues: Dict[str, Any],
    ) -> List[str]:
        """Create specific action items based on issues"""
        action_items = []
        
        if metric_name == "completeness":
            for topic in issues.get("missing_topics", [])[:3]:
                action_items.append(f"Add section covering: {topic}")
            for section in issues.get("underdeveloped", [])[:2]:
                action_items.append(f"Expand content in: {section}")
        
        elif metric_name == "relevance":
            for section in issues.get("drift_sections", [])[:3]:
                action_items.append(f"Review and potentially remove: {section}")
        
        elif metric_name == "clarity":
            for issue in issues.get("flow_issues", [])[:3]:
                action_items.append(f"Address flow issue: {issue}")
        
        elif metric_name == "factual_accuracy":
            for claim in issues.get("unverified", [])[:3]:
                action_items.append(f"Add citation for: {claim[:50]}...")
        
        elif metric_name == "instruction_following":
            for violation in issues.get("violations", [])[:3]:
                if isinstance(violation, dict):
                    action_items.append(f"Fix: {violation.get('violation', 'violation')}")
                else:
                    action_items.append(f"Fix: {violation}")
        
        elif metric_name == "style_tone":
            for inconsistency in issues.get("inconsistencies", [])[:3]:
                if isinstance(inconsistency, dict):
                    action_items.append(f"Revise {inconsistency.get('section', 'section')}: {inconsistency.get('issue', '')}")
        
        elif metric_name == "safety":
            for bias in issues.get("biases", [])[:3]:
                action_items.append(f"Review for bias: {bias}")
        
        return action_items if action_items else ["Review this section for improvement opportunities"]
    
    def extract_flags(
        self,
        agent_results: Dict[str, Any],
    ) -> List[Flag]:
        """
        Extract all flags/issues from agent results.
        
        Returns list matching frontend Flag type.
        """
        flags = []
        
        # Safety flags
        safety_result = agent_results.get("safety_detection", {}).get("safety", {})
        for flag in safety_result.get("flags", []):
            if isinstance(flag, dict):
                flags.append(Flag(
                    type=flag.get("category", "safety"),
                    message=flag.get("description", "Safety concern identified"),
                    severity=flag.get("severity", "MEDIUM"),
                ))
        
        # Clarity issues as flags
        clarity_result = agent_results.get("content_quality", {}).get("clarity", {})
        for issue in clarity_result.get("flow_issues", [])[:3]:
            flags.append(Flag(
                type="clarity",
                message=issue,
                severity="LOW",
            ))
        
        # Style inconsistencies as flags
        style_result = agent_results.get("style_compliance", {}).get("style_tone", {})
        for inconsistency in style_result.get("inconsistencies", [])[:3]:
            if isinstance(inconsistency, dict):
                flags.append(Flag(
                    type="style",
                    message=f"{inconsistency.get('section', 'Section')}: {inconsistency.get('issue', 'Style inconsistency')}",
                    severity="MEDIUM",
                ))
        
        # Instruction violations as flags
        instruction_result = agent_results.get("style_compliance", {}).get("instruction_following", {})
        for violation in instruction_result.get("violations", [])[:3]:
            if isinstance(violation, dict):
                flags.append(Flag(
                    type="compliance",
                    message=f"Guideline violation: {violation.get('violation', 'Deviation from guidelines')}",
                    severity="MEDIUM",
                ))
        
        # Unverified claims as flags
        accuracy_result = agent_results.get("content_quality", {}).get("factual_accuracy", {})
        for claim in accuracy_result.get("unverified_claims", [])[:3]:
            flags.append(Flag(
                type="accuracy",
                message=f"Unverified claim: {claim[:100]}",
                severity="MEDIUM",
            ))
        
        return flags
    
    def generate_suggested_changes(
        self,
        improvements: List[Improvement],
        max_changes: int = 10,
    ) -> List[str]:
        """
        Generate simple list of suggested changes for frontend.
        
        Returns list of string suggestions.
        """
        changes = []
        for improvement in improvements[:max_changes]:
            changes.append(improvement.suggestion)
        return changes
    
    def create_structured_suggestions(
        self,
        improvements: List[Improvement],
    ) -> List[Dict[str, Any]]:
        """
        Create structured suggestion objects for frontend.
        
        Returns list matching frontend Suggestion type.
        """
        suggestions = []
        
        for i, improvement in enumerate(improvements):
            suggestions.append({
                "id": f"sug_{uuid.uuid4().hex[:8]}",
                "category": self.aggregator.METRIC_CATEGORIES.get(
                    improvement.category, improvement.category
                ),
                "priority": improvement.priority,
                "title": self._create_suggestion_title(improvement),
                "description": improvement.suggestion,
                "actionItems": improvement.action_items or [],
            })
        
        return suggestions
    
    def _create_suggestion_title(self, improvement: Improvement) -> str:
        """Create a brief title for a suggestion"""
        category_name = self.aggregator.METRIC_DISPLAY_NAMES.get(
            improvement.category, improvement.category
        )
        
        if improvement.priority == "HIGH":
            return f"Critical: Improve {category_name}"
        elif improvement.priority == "MEDIUM":
            return f"Recommended: Enhance {category_name}"
        else:
            return f"Optional: Refine {category_name}"
