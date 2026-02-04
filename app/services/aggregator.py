"""Score aggregation and analysis service"""

import logging
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from app.config import get_settings, classify_score, TARGET_SCORES

logger = logging.getLogger(__name__)


@dataclass
class MetricScore:
    """Individual metric score with metadata"""
    name: str
    score: float
    weight: float
    target: float
    analysis: str
    evidence: List[str]
    sub_metrics: Dict[str, Any]


class ScoreAggregator:
    """
    Service for aggregating and analyzing evaluation scores.
    
    Responsibilities:
    - Combine scores from 3 agents into unified metrics
    - Calculate weighted overall score
    - Identify strengths and weaknesses
    - Classify performance levels
    """
    
    # Metric name mappings for consistency
    METRIC_DISPLAY_NAMES = {
        "completeness": "Completeness",
        "relevance": "Relevance",
        "clarity": "Clarity",
        "factual_accuracy": "Factual Accuracy",
        "instruction_following": "Instruction Following",
        "style_tone": "Writing Style & Tone",
        "context_awareness": "Context Awareness",
        "safety": "Safety",
    }
    
    METRIC_CATEGORIES = {
        "completeness": "completeness",
        "relevance": "relevance",
        "clarity": "clarity",
        "factual_accuracy": "factualAccuracy",
        "instruction_following": "instructionFollowing",
        "style_tone": "writingStyleTone",
        "context_awareness": "contextAwareness",
        "safety": "safety",
    }
    
    def __init__(self, settings=None):
        """Initialize with optional custom settings"""
        self.settings = settings or get_settings()
        self.weights = self.settings.metric_weights
        self.targets = TARGET_SCORES
    
    def extract_all_scores(
        self,
        content_quality_result: Dict[str, Any],
        style_compliance_result: Dict[str, Any],
        safety_detection_result: Dict[str, Any],
    ) -> Dict[str, MetricScore]:
        """
        Extract all 8 metric scores from agent results.
        
        Args:
            content_quality_result: Results from Agent 1
            style_compliance_result: Results from Agent 2
            safety_detection_result: Results from Agent 3
            
        Returns:
            Dictionary mapping metric names to MetricScore objects
        """
        metrics = {}
        
        # Agent 1: Content Quality (4 metrics)
        for metric_name in ["completeness", "relevance", "clarity", "factual_accuracy"]:
            data = content_quality_result.get(metric_name, {})
            metrics[metric_name] = MetricScore(
                name=metric_name,
                score=self._normalize_score(data.get("score", 50)),
                weight=self.weights.get(metric_name, 0.1143),
                target=self.targets.get(metric_name, 80),
                analysis=data.get("analysis", ""),
                evidence=data.get("evidence", []),
                sub_metrics=data.get("sub_metrics", {}),
            )
        
        # Agent 2: Style Compliance (3 metrics)
        for metric_name in ["instruction_following", "style_tone", "context_awareness"]:
            data = style_compliance_result.get(metric_name, {})
            metrics[metric_name] = MetricScore(
                name=metric_name,
                score=self._normalize_score(data.get("score", 50)),
                weight=self.weights.get(metric_name, 0.1143),
                target=self.targets.get(metric_name, 80),
                analysis=data.get("analysis", ""),
                evidence=data.get("evidence", []),
                sub_metrics=data.get("sub_metrics", {}),
            )
        
        # Agent 3: Safety Detection (1 metric)
        safety_data = safety_detection_result.get("safety", {})
        metrics["safety"] = MetricScore(
            name="safety",
            score=self._normalize_score(safety_data.get("score", 50)),
            weight=self.weights.get("safety", 0.1143),
            target=self.targets.get("safety", 95),
            analysis=safety_data.get("analysis", ""),
            evidence=safety_data.get("evidence", []),
            sub_metrics=safety_data.get("sub_metrics", {}),
        )
        
        return metrics
    
    def _normalize_score(self, score: Any) -> float:
        """Normalize score to valid range"""
        try:
            score = float(score)
            return max(0, min(100, score))
        except (TypeError, ValueError):
            return 50.0
    
    def calculate_overall_score(self, metrics: Dict[str, MetricScore]) -> float:
        """
        Calculate weighted overall score.
        
        Formula:
        overall = (completeness × 0.20) + (other_metrics × 0.1143 each)
        
        Args:
            metrics: Dictionary of MetricScore objects
            
        Returns:
            Weighted overall score (0-100)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric in metrics.items():
            weighted_score = metric.score * metric.weight
            total_score += weighted_score
            total_weight += metric.weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            overall = total_score / total_weight
        else:
            overall = total_score
        
        return round(overall, 2)
    
    def identify_strengths(
        self,
        metrics: Dict[str, MetricScore],
        top_n: int = 5,
    ) -> List[str]:
        """
        Identify top performing areas (strengths).
        
        Args:
            metrics: Dictionary of MetricScore objects
            top_n: Number of strengths to return
            
        Returns:
            List of strength descriptions
        """
        # Sort metrics by score (descending)
        sorted_metrics = sorted(
            metrics.values(),
            key=lambda m: m.score,
            reverse=True
        )
        
        strengths = []
        for metric in sorted_metrics[:top_n]:
            display_name = self.METRIC_DISPLAY_NAMES.get(metric.name, metric.name)
            classification = classify_score(metric.score)
            
            # Generate strength statement
            if metric.score >= 90:
                strength = f"Excellent {display_name.lower()} with a score of {metric.score:.0f}/100"
            elif metric.score >= 80:
                strength = f"Strong {display_name.lower()} performance at {metric.score:.0f}/100"
            elif metric.score >= 70:
                strength = f"Good {display_name.lower()} at {metric.score:.0f}/100"
            else:
                strength = f"Adequate {display_name.lower()} at {metric.score:.0f}/100"
            
            # Add specific evidence if available
            if metric.evidence and metric.evidence[0]:
                strength += f" - {metric.evidence[0][:100]}"
            
            strengths.append(strength)
        
        return strengths
    
    def identify_weaknesses(
        self,
        metrics: Dict[str, MetricScore],
        bottom_n: int = 3,
    ) -> List[Tuple[str, float, str]]:
        """
        Identify areas needing improvement (weaknesses).
        
        Args:
            metrics: Dictionary of MetricScore objects
            bottom_n: Number of weaknesses to return
            
        Returns:
            List of tuples (metric_name, score, analysis)
        """
        # Sort metrics by score (ascending)
        sorted_metrics = sorted(
            metrics.values(),
            key=lambda m: m.score
        )
        
        weaknesses = []
        for metric in sorted_metrics[:bottom_n]:
            weaknesses.append((
                metric.name,
                metric.score,
                metric.analysis,
            ))
        
        return weaknesses
    
    def determine_priority(self, score: float) -> str:
        """Determine improvement priority based on score"""
        if score < 65:
            return "HIGH"
        elif score <= 80:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_gap_from_target(self, metrics: Dict[str, MetricScore]) -> Dict[str, float]:
        """Calculate the gap between current scores and targets"""
        gaps = {}
        for metric_name, metric in metrics.items():
            gaps[metric_name] = metric.target - metric.score
        return gaps
    
    def get_scores_dict(self, metrics: Dict[str, MetricScore]) -> Dict[str, float]:
        """Convert metrics to simple score dictionary"""
        return {name: metric.score for name, metric in metrics.items()}
    
    def generate_summary(
        self,
        metrics: Dict[str, MetricScore],
        overall_score: float,
        ai_likelihood: float,
    ) -> str:
        """
        Generate a 2-3 sentence summary of the evaluation.
        
        Args:
            metrics: Dictionary of MetricScore objects
            overall_score: Calculated overall score
            ai_likelihood: AI likelihood percentage
            
        Returns:
            Summary string
        """
        # Identify best and worst metrics
        sorted_metrics = sorted(metrics.values(), key=lambda m: m.score)
        worst = sorted_metrics[0]
        best = sorted_metrics[-1]
        
        # Get overall classification
        classification = classify_score(overall_score)
        
        # Build summary
        summary_parts = []
        
        # Overall assessment
        if classification == "excellent":
            summary_parts.append(f"Your content achieves an excellent overall score of {overall_score:.0f}/100.")
        elif classification == "good":
            summary_parts.append(f"Your content demonstrates good quality with an overall score of {overall_score:.0f}/100.")
        elif classification == "adequate":
            summary_parts.append(f"Your content is adequate with an overall score of {overall_score:.0f}/100.")
        else:
            summary_parts.append(f"Your content scores {overall_score:.0f}/100 and has room for improvement.")
        
        # Highlight strength
        best_name = self.METRIC_DISPLAY_NAMES.get(best.name, best.name)
        summary_parts.append(f"{best_name} is your strongest area at {best.score:.0f}/100.")
        
        # Note improvement area
        worst_name = self.METRIC_DISPLAY_NAMES.get(worst.name, worst.name)
        if worst.score < 70:
            summary_parts.append(f"Focus on improving {worst_name.lower()} (currently {worst.score:.0f}/100) for the biggest impact.")
        
        return " ".join(summary_parts)
    
    def get_detailed_metrics(
        self,
        metrics: Dict[str, MetricScore],
    ) -> List[Dict[str, Any]]:
        """
        Convert metrics to detailed format for API response.
        
        Returns list of metric objects matching frontend DetailedMetric type.
        """
        detailed = []
        
        for metric_name, metric in metrics.items():
            detailed.append({
                "label": self.METRIC_DISPLAY_NAMES.get(metric_name, metric_name),
                "score": metric.score,
                "explanation": metric.analysis,
                "category": self.METRIC_CATEGORIES.get(metric_name, metric_name),
                "subScores": [
                    {"label": k, "score": v}
                    for k, v in metric.sub_metrics.items()
                    if isinstance(v, (int, float))
                ],
                "suggestions": [],  # Will be filled by InsightGenerator
                "evidence": metric.evidence,
            })
        
        return detailed
