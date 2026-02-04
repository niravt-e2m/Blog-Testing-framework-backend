"""LangGraph workflow for blog evaluation orchestration"""

import asyncio
import logging
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END

from app.config import get_settings
from app.agents import ContentQualityAgent, StyleComplianceAgent, SafetyDetectionAgent
from app.services import ReferenceValidator, ScoreAggregator, InsightGenerator
from app.utils.text_analysis import TextAnalyzer

logger = logging.getLogger(__name__)


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that merges two dictionaries, with right taking precedence."""
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    return {**left, **right}


def merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    """Reducer that concatenates two lists."""
    if left is None:
        return right or []
    if right is None:
        return left or []
    return left + right


class WorkflowState(TypedDict, total=False):
    """
    Shared state passed between workflow nodes.
    
    Channels with Annotated types use custom reducers to merge parallel updates.
    """
    # Input data (set once by validate_inputs, not updated by parallel nodes)
    inputs: Dict[str, Any]
    
    # Metadata calculated during validation
    metadata: Dict[str, Any]
    
    # Results from each agent - USES MERGE REDUCER for parallel updates
    agent_results: Annotated[Dict[str, Any], merge_dicts]
    
    # Reference validation results
    reference_validation: Dict[str, Any]
    
    # Aggregated scores
    scores: Dict[str, float]
    
    # AI detection result - USES MERGE REDUCER for parallel updates  
    ai_likelihood: Annotated[Dict[str, Any], merge_dicts]
    
    # Generated insights
    strengths: List[str]
    improvements: List[Dict[str, Any]]
    
    # Final compiled report
    final_report: Dict[str, Any]
    
    # Processing status - USES MERGE REDUCER for parallel error collection
    errors: Annotated[List[str], merge_lists]
    start_time: float
    processing_time: float


@dataclass
class EvaluationWorkflow:
    """
    LangGraph workflow for comprehensive blog evaluation.
    
    Graph Structure (7 Nodes):
    1. Input Validator - Sanitize and validate inputs
    2. Content Quality Agent - Evaluate completeness, relevance, clarity, accuracy
    3. Style Compliance Agent - Evaluate instruction following, style, context
    4. Safety Detection Agent - Safety check and AI likelihood
    5. Reference Validator - Validate and analyze reference URLs (parallel)
    6. Score Aggregator - Combine scores, generate insights
    7. Report Compiler - Format final response
    
    Execution Flow:
    - Node 1 → Node 5 → Node 2 → Node 6 → Node 7
    - Node 1 → Nodes 3 & 4 (parallel with Node 5) → Node 6
    """
    
    settings: Any = field(default_factory=get_settings)
    
    # Agents
    content_quality_agent: Optional[ContentQualityAgent] = None
    style_compliance_agent: Optional[StyleComplianceAgent] = None
    safety_detection_agent: Optional[SafetyDetectionAgent] = None
    
    # Services
    reference_validator: Optional[ReferenceValidator] = None
    score_aggregator: Optional[ScoreAggregator] = None
    insight_generator: Optional[InsightGenerator] = None
    
    def __post_init__(self):
        """Initialize agents and services"""
        # Initialize agents
        self.content_quality_agent = ContentQualityAgent(settings=self.settings)
        self.style_compliance_agent = StyleComplianceAgent(settings=self.settings)
        self.safety_detection_agent = SafetyDetectionAgent(settings=self.settings)
        
        # Initialize services
        self.reference_validator = ReferenceValidator(
            timeout=self.settings.reference_validation_timeout
        )
        self.score_aggregator = ScoreAggregator(settings=self.settings)
        self.insight_generator = InsightGenerator(aggregator=self.score_aggregator)
    
    # =========================================================================
    # Node 1: Input Validator
    # =========================================================================
    
    async def validate_inputs(self, state: WorkflowState) -> WorkflowState:
        """
        Node 1: Validate and sanitize incoming data.
        
        Tasks:
        - Validate all required fields
        - Calculate basic metadata
        - Initialize state structure
        """
        logger.info("Node 1: Validating inputs")
        
        inputs = state.get("inputs", {})
        errors = []
        
        # Validate required fields
        blog_content = inputs.get("blog_content", "")
        if not blog_content or len(blog_content) < 100:
            errors.append("Blog content must be at least 100 characters")
        
        tone_guidelines = inputs.get("tone_guidelines", "")
        target_audience = inputs.get("target_audience", "General audience")
        
        # Calculate metadata
        text_analyzer = TextAnalyzer(blog_content)
        stats = text_analyzer.get_statistics()
        
        metadata = {
            "character_count": stats.character_count,
            "word_count": stats.word_count,
            "paragraph_count": stats.paragraph_count,
            "sentence_count": stats.sentence_count,
            "estimated_reading_time": text_analyzer.estimate_reading_time(),
            "start_time": datetime.utcnow().isoformat(),
        }
        
        # Initialize state structure
        return {
            **state,
            "inputs": {
                "blog_title": inputs.get("blog_title", "Untitled"),
                "blog_content": blog_content,
                "tone_guidelines": tone_guidelines,
                "target_audience": target_audience,
                "reference_links": inputs.get("reference_links", []),
                "reference_texts": inputs.get("reference_texts", []),
                "blog_outline": inputs.get("blog_outline", ""),
                "options": inputs.get("options", {}),
            },
            "metadata": metadata,
            "agent_results": {},
            "reference_validation": {},
            "scores": {},
            "errors": errors,
            "start_time": asyncio.get_event_loop().time(),
        }
    
    # =========================================================================
    # Node 2: Content Quality Agent
    # =========================================================================
    
    async def run_content_quality_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Node 2: Evaluate completeness, relevance, clarity, factual accuracy.
        
        This node runs AFTER reference_validation_node, so it can access
        extracted reference texts from state["reference_validation"]["extracted_texts"].
        """
        logger.info("Node 2: Running Content Quality Agent")
        
        inputs = state.get("inputs", {})
        errors = list(state.get("errors", []))  # Copy to avoid mutating shared state
        
        # Combine original reference_texts with extracted texts from reference validation
        reference_validation = state.get("reference_validation", {})
        extracted_texts = reference_validation.get("extracted_texts", [])
        all_reference_texts = inputs.get("reference_texts", []) + extracted_texts
        
        try:
            result = await self.content_quality_agent.safe_evaluate(
                blog_title=inputs.get("blog_title", ""),
                blog_content=inputs.get("blog_content", ""),
                reference_links=inputs.get("reference_links", []),
                target_audience=inputs.get("target_audience", "General audience"),
                reference_texts=all_reference_texts,
            )
            
            agent_results = dict(state.get("agent_results", {}))
            agent_results["content_quality"] = result
            
            return {"agent_results": agent_results}
            
        except Exception as e:
            logger.error(f"Content Quality Agent failed: {e}")
            errors.append(f"Content quality evaluation failed: {str(e)}")
            
            # Use default scores
            agent_results = dict(state.get("agent_results", {}))
            agent_results["content_quality"] = self.content_quality_agent._get_default_scores()
            
            return {"agent_results": agent_results, "errors": errors}
    
    # =========================================================================
    # Node 3: Style Compliance Agent
    # =========================================================================
    
    async def run_style_compliance_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Node 3: Evaluate instruction following, style/tone, context awareness.
        
        IMPORTANT: This node runs in parallel with safety_detection and reference_validation.
        To avoid LangGraph channel conflicts, it returns only the channels it updates.
        """
        logger.info("Node 3: Running Style Compliance Agent")
        
        inputs = state.get("inputs", {})
        errors = list(state.get("errors", []))  # Copy to avoid modifying shared state
        
        try:
            result = await self.style_compliance_agent.safe_evaluate(
                blog_content=inputs.get("blog_content", ""),
                tone_guidelines=inputs.get("tone_guidelines", ""),
                target_audience=inputs.get("target_audience", "General audience"),
                blog_outline=inputs.get("blog_outline", ""),
            )
            
            agent_results = dict(state.get("agent_results", {}))  # Copy to avoid modifying shared state
            agent_results["style_compliance"] = result
            
            # Return ONLY the channels we update
            return {"agent_results": agent_results}
            
        except Exception as e:
            logger.error(f"Style Compliance Agent failed: {e}")
            errors.append(f"Style compliance evaluation failed: {str(e)}")
            
            agent_results = dict(state.get("agent_results", {}))
            agent_results["style_compliance"] = self.style_compliance_agent._get_default_scores()
            
            # Return ONLY the channels we update
            return {"agent_results": agent_results, "errors": errors}
    
    # =========================================================================
    # Node 4: Safety Detection Agent
    # =========================================================================
    
    async def run_safety_detection_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Node 4: Safety check and AI likelihood detection.
        
        IMPORTANT: This node runs in parallel with style_compliance and reference_validation.
        To avoid LangGraph channel conflicts, it returns only the channels it updates.
        """
        logger.info("Node 4: Running Safety Detection Agent")
        
        inputs = state.get("inputs", {})
        options = inputs.get("options", {})
        errors = list(state.get("errors", []))  # Copy to avoid modifying shared state
        
        try:
            result = await self.safety_detection_agent.safe_evaluate(
                blog_content=inputs.get("blog_content", ""),
                target_audience=inputs.get("target_audience", "General audience"),
                include_ai_detection=options.get("include_ai_detection", True),
            )
            
            agent_results = dict(state.get("agent_results", {}))  # Copy to avoid modifying shared state
            agent_results["safety_detection"] = result
            
            # Extract AI likelihood for top-level access
            ai_likelihood = result.get("ai_likelihood", {})
            
            # Return ONLY the channels we update
            return {
                "agent_results": agent_results,
                "ai_likelihood": ai_likelihood,
            }
            
        except Exception as e:
            logger.error(f"Safety Detection Agent failed: {e}")
            errors.append(f"Safety detection failed: {str(e)}")
            
            agent_results = dict(state.get("agent_results", {}))
            defaults = self.safety_detection_agent._get_default_scores()
            agent_results["safety_detection"] = defaults
            
            # Return ONLY the channels we update
            return {
                "agent_results": agent_results,
                "ai_likelihood": defaults.get("ai_likelihood", {}),
                "errors": errors,
            }
    
    # =========================================================================
    # Node 5: Reference Validator
    # =========================================================================
    
    async def validate_references(self, state: WorkflowState) -> WorkflowState:
        """
        Node 5: Validate and analyze reference URLs.
        
        IMPORTANT: This node runs in parallel with style_compliance and safety_detection.
        To avoid LangGraph channel conflicts, it returns only the channels it updates.
        Extracted texts are stored in reference_validation["extracted_texts"]
        and read by content_quality_agent after this node completes.
        """
        logger.info("Node 5: Validating references")
        
        inputs = state.get("inputs", {})
        options = inputs.get("options", {})
        reference_links = inputs.get("reference_links", [])
        
        # Skip if fast mode or no references
        if options.get("depth_level") == "fast" or not reference_links:
            # Return ONLY the channels we update
            return {
                "reference_validation": {
                    "validated_urls": [],
                    "extracted_claims": [],
                    "extracted_texts": [],
                    "failed_urls": [],
                    "skipped": True,
                }
            }
        
        try:
            results = await self.reference_validator.validate_urls(reference_links)
            
            validated_urls = [r for r in results if r.accessible]
            failed_urls = [r for r in results if not r.accessible]
            all_claims = self.reference_validator.get_all_claims(validated_urls)

            extracted_texts = []
            for r in validated_urls:
                if r.main_content:
                    title = r.title or "Untitled reference"
                    snippet = r.main_content[:1500]
                    extracted_texts.append(
                        f"[Reference URL]: {r.url}\nTitle: {title}\nContent: {snippet}"
                    )
            
            # Return ONLY the channels we update
            return {
                "reference_validation": {
                    "validated_urls": [
                        {
                            "url": r.url,
                            "title": r.title,
                            "credibility_score": r.credibility_score,
                            "response_time_ms": r.response_time_ms,
                        }
                        for r in validated_urls
                    ],
                    "extracted_claims": all_claims,
                    "extracted_texts": extracted_texts,
                    "failed_urls": [
                        {"url": r.url, "error": r.error}
                        for r in failed_urls
                    ],
                    "skipped": False,
                }
            }
            
        except Exception as e:
            logger.warning(f"Reference validation failed: {e}")
            # Return ONLY the channels we update
            return {
                "reference_validation": {
                    "validated_urls": [],
                    "extracted_claims": [],
                    "extracted_texts": [],
                    "failed_urls": [],
                    "error": str(e),
                }
            }
    
    # =========================================================================
    # Node 6: Score Aggregator & Insight Generator
    # =========================================================================
    
    async def aggregate_and_generate_insights(self, state: WorkflowState) -> WorkflowState:
        """
        Node 6: Combine all scores and create recommendations.
        """
        logger.info("Node 6: Aggregating scores and generating insights")
        
        agent_results = state.get("agent_results", {})
        ai_likelihood = state.get("ai_likelihood", {})
        
        # Extract all metrics
        metrics = self.score_aggregator.extract_all_scores(
            content_quality_result=agent_results.get("content_quality", {}),
            style_compliance_result=agent_results.get("style_compliance", {}),
            safety_detection_result=agent_results.get("safety_detection", {}),
        )
        
        # Calculate overall score
        overall_score = self.score_aggregator.calculate_overall_score(metrics)
        
        # Get individual scores as dict
        scores = self.score_aggregator.get_scores_dict(metrics)
        scores["overall"] = overall_score
        
        # Identify strengths
        strengths = self.score_aggregator.identify_strengths(metrics)
        
        # Generate improvements
        improvements = self.insight_generator.generate_improvements(
            metrics=metrics,
            agent_results=agent_results,
        )
        
        # Convert improvements to dicts
        improvements_list = [
            {
                "priority": imp.priority,
                "category": imp.category,
                "suggestion": imp.suggestion,
                "impact": imp.impact,
                "action_items": imp.action_items,
            }
            for imp in improvements
        ]
        
        # Generate summary
        summary = self.score_aggregator.generate_summary(
            metrics=metrics,
            overall_score=overall_score,
            ai_likelihood=ai_likelihood.get("percentage", 50),
        )
        
        # Get detailed metrics for response
        detailed_metrics = self.score_aggregator.get_detailed_metrics(metrics)
        
        # Extract flags
        flags = self.insight_generator.extract_flags(agent_results)
        flags_list = [
            {
                "type": f.type,
                "message": f.message,
                "severity": f.severity,
                "startIndex": f.start_index,
                "endIndex": f.end_index,
            }
            for f in flags
        ]
        
        # Generate suggested changes
        suggested_changes = self.insight_generator.generate_suggested_changes(improvements)
        
        # Create structured suggestions
        suggestions = self.insight_generator.create_structured_suggestions(improvements)
        
        return {
            "scores": scores,
            "strengths": strengths,
            "improvements": improvements_list,
            "summary": summary,
            "detailed_metrics": detailed_metrics,
            "flags": flags_list,
            "suggested_changes": suggested_changes,
            "suggestions": suggestions,
        }
    
    # =========================================================================
    # Node 7: Report Compiler
    # =========================================================================
    
    async def compile_final_report(self, state: WorkflowState) -> WorkflowState:
        """
        Node 7: Format final JSON response.
        """
        logger.info("Node 7: Compiling final report")
        
        scores = state.get("scores", {})
        ai_likelihood = state.get("ai_likelihood", {})
        metadata = state.get("metadata", {})
        start_time = state.get("start_time", 0)
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Build final report matching frontend EvaluationResult type
        final_report = {
            # Overall assessment
            "overallScore": scores.get("overall", 50),
            "summary": state.get("summary", "Evaluation complete."),
            
            # AI Detection
            "aiLikelihood": ai_likelihood.get("percentage", 50),
            "aiDetectionDetails": {
                "percentage": ai_likelihood.get("percentage", 50),
                "classification": ai_likelihood.get("classification", "Unknown"),
                "reasoning": ai_likelihood.get("reasoning", ""),
                "linguisticMarkers": ai_likelihood.get("linguistic_markers", []),
                "statisticalAnalysis": ai_likelihood.get("statistical_analysis", {}),
            },
            
            # Detailed metrics
            "metrics": state.get("detailed_metrics", []),
            
            # Individual metric scores (for frontend compatibility)
            "instructionFollowing": scores.get("instruction_following", 50),
            "factualAccuracy": scores.get("factual_accuracy", 50),
            "relevance": scores.get("relevance", 50),
            "completeness": scores.get("completeness", 50),
            "writingStyleTone": scores.get("style_tone", 50),
            "clarity": scores.get("clarity", 50),
            "contextAwareness": scores.get("context_awareness", 50),
            "safety": scores.get("safety", 50),
            
            # Issues and suggestions
            "flags": state.get("flags", []),
            "suggestedChanges": state.get("suggested_changes", []),
            "suggestions": state.get("suggestions", []),
            
            # Extended fields
            "strengths": state.get("strengths", []),
            "improvements": state.get("improvements", []),
            
            # Metadata
            "metadata": {
                "contentLength": metadata.get("character_count", 0),
                "wordCount": metadata.get("word_count", 0),
                "paragraphCount": metadata.get("paragraph_count", 0),
                "estimatedReadingTime": metadata.get("estimated_reading_time", "Unknown"),
                "processingTime": round(processing_time, 2),
                "evaluationTimestamp": datetime.utcnow().isoformat(),
            },
            
            # Error information (if any)
            "errors": state.get("errors", []),
        }
        
        return {
            "final_report": final_report,
            "processing_time": processing_time,
        }


def create_evaluation_graph(workflow: Optional[EvaluationWorkflow] = None) -> StateGraph:
    """
    Create the LangGraph evaluation workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    if workflow is None:
        workflow = EvaluationWorkflow()
    
    # Create graph
    graph = StateGraph(WorkflowState)
    
    # Add nodes
    graph.add_node("validate_inputs", workflow.validate_inputs)
    graph.add_node("content_quality", workflow.run_content_quality_agent)
    graph.add_node("style_compliance", workflow.run_style_compliance_agent)
    graph.add_node("safety_detection", workflow.run_safety_detection_agent)
    graph.add_node("reference_validation_node", workflow.validate_references)
    graph.add_node("aggregate_insights", workflow.aggregate_and_generate_insights)
    graph.add_node("compile_report", workflow.compile_final_report)
    
    # Define edges
    # Start with input validation
    graph.set_entry_point("validate_inputs")
    
    # After validation, fan out to parallel nodes via a conditional edge
    def _fanout_after_validation(_state: WorkflowState):
        return ["style_compliance", "safety_detection", "reference_validation_node"]

    graph.add_conditional_edges("validate_inputs", _fanout_after_validation)
    graph.add_edge("reference_validation_node", "content_quality")
    
    # Wait for all evaluation nodes before aggregation
    graph.add_edge(
        ["content_quality", "style_compliance", "safety_detection"],
        "aggregate_insights",
    )
    
    # Aggregation to report compilation
    graph.add_edge("aggregate_insights", "compile_report")
    
    # Final node to end
    graph.add_edge("compile_report", END)
    
    return graph.compile()


async def run_evaluation(
    blog_content: str,
    tone_guidelines: str,
    target_audience: str,
    blog_title: str = "",
    reference_links: Optional[List[str]] = None,
    reference_texts: Optional[List[str]] = None,
    blog_outline: str = "",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the evaluation workflow.
    
    Args:
        blog_content: The blog content to evaluate
        tone_guidelines: Tone of voice guidelines
        target_audience: Target audience description
        blog_title: Optional blog title
        reference_links: Optional list of reference URLs
        reference_texts: Optional list of reference text content
        blog_outline: Blog outline/structure
        options: Evaluation options
        
    Returns:
        Final evaluation report
    """
    # Create workflow and graph
    workflow = EvaluationWorkflow()
    graph = create_evaluation_graph(workflow)
    
    # Prepare initial state
    initial_state: WorkflowState = {
        "inputs": {
            "blog_title": blog_title,
            "blog_content": blog_content,
            "tone_guidelines": tone_guidelines,
            "target_audience": target_audience,
            "reference_links": reference_links or [],
            "reference_texts": reference_texts or [],
            "blog_outline": blog_outline,
            "options": options or {},
        },
        "metadata": {},
        "agent_results": {},
        "reference_validation": {},
        "scores": {},
        "ai_likelihood": {},
        "strengths": [],
        "improvements": [],
        "final_report": {},
        "errors": [],
        "start_time": 0,
        "processing_time": 0,
    }
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state)
    
    return final_state.get("final_report", {})
