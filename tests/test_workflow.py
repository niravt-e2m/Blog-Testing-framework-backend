"""Integration tests for LangGraph evaluation workflow"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.workflows.evaluation_workflow import (
    EvaluationWorkflow,
    WorkflowState,
    create_evaluation_graph,
    run_evaluation,
)


# Sample test data
SAMPLE_BLOG_CONTENT = """
# Getting Started with Machine Learning

Machine learning is transforming industries across the globe. In this guide, 
we'll explore the fundamentals of ML and provide practical examples.

## What is Machine Learning?

Machine learning is a subset of artificial intelligence that enables computers 
to learn from data without being explicitly programmed. According to recent surveys, 
over 60% of enterprises are now using ML in production.

## Types of Machine Learning

1. **Supervised Learning**: The algorithm learns from labeled data
2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data  
3. **Reinforcement Learning**: The algorithm learns through trial and error

## Conclusion

Machine learning offers powerful capabilities for solving complex problems. 
Start with small projects and gradually build your expertise.
"""


class TestEvaluationWorkflow:
    """Tests for the evaluation workflow"""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance with mocked agents"""
        wf = EvaluationWorkflow()
        return wf
    
    @pytest.fixture
    def initial_state(self) -> WorkflowState:
        """Create initial workflow state"""
        return {
            "inputs": {
                "blog_title": "Getting Started with Machine Learning",
                "blog_content": SAMPLE_BLOG_CONTENT,
                "tone_guidelines": "Professional and educational",
                "target_audience": "Beginner developers",
                "reference_links": [],
                "reference_texts": [],
                "prompt_text": "",
                "options": {},
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
    
    @pytest.mark.asyncio
    async def test_validate_inputs_success(self, workflow, initial_state):
        """Test input validation node"""
        result = await workflow.validate_inputs(initial_state)
        
        assert "metadata" in result
        assert result["metadata"]["word_count"] > 0
        assert result["metadata"]["paragraph_count"] > 0
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_inputs_short_content(self, workflow):
        """Test input validation with too short content"""
        state: WorkflowState = {
            "inputs": {
                "blog_content": "Too short",
            },
            "errors": [],
        }
        
        result = await workflow.validate_inputs(state)
        
        assert len(result["errors"]) > 0
        assert "100 characters" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_content_quality_agent_node(self, workflow, initial_state):
        """Test content quality agent node"""
        # Setup state after validation
        state = await workflow.validate_inputs(initial_state)
        
        # Mock the agent's evaluate method
        mock_result = {
            "completeness": {"score": 75, "analysis": "Good", "evidence": []},
            "relevance": {"score": 85, "analysis": "Relevant", "evidence": []},
            "clarity": {"score": 88, "analysis": "Clear", "evidence": []},
            "factual_accuracy": {"score": 80, "analysis": "Accurate", "evidence": []},
        }
        
        with patch.object(
            workflow.content_quality_agent, 
            'safe_evaluate', 
            new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_result
            
            result = await workflow.run_content_quality_agent(state)
            
            assert "content_quality" in result["agent_results"]
            assert result["agent_results"]["content_quality"]["completeness"]["score"] == 75
    
    @pytest.mark.asyncio
    async def test_aggregate_insights_node(self, workflow, initial_state):
        """Test score aggregation node"""
        # Setup state with agent results
        state = await workflow.validate_inputs(initial_state)
        state["agent_results"] = {
            "content_quality": {
                "completeness": {"score": 70, "analysis": "Good coverage", "evidence": [], "sub_metrics": {}},
                "relevance": {"score": 85, "analysis": "On topic", "evidence": [], "sub_metrics": {}},
                "clarity": {"score": 90, "analysis": "Clear writing", "evidence": [], "sub_metrics": {}},
                "factual_accuracy": {"score": 80, "analysis": "Accurate", "evidence": [], "sub_metrics": {}},
            },
            "style_compliance": {
                "instruction_following": {"score": 88, "analysis": "Follows guidelines", "evidence": [], "sub_metrics": {}},
                "style_tone": {"score": 85, "analysis": "Consistent tone", "evidence": [], "sub_metrics": {}},
                "context_awareness": {"score": 82, "analysis": "Audience appropriate", "evidence": [], "sub_metrics": {}},
            },
            "safety_detection": {
                "safety": {"score": 95, "analysis": "Safe content", "evidence": [], "sub_metrics": {}},
            },
        }
        state["ai_likelihood"] = {
            "percentage": 40,
            "classification": "Likely human-written",
            "reasoning": "Shows variation",
        }
        
        result = await workflow.aggregate_and_generate_insights(state)
        
        assert "scores" in result
        assert "overall" in result["scores"]
        assert result["scores"]["overall"] > 0
        assert "strengths" in result
        assert len(result["strengths"]) > 0
    
    @pytest.mark.asyncio
    async def test_compile_report_node(self, workflow, initial_state):
        """Test report compilation node"""
        # Setup state after aggregation
        import asyncio
        
        state = await workflow.validate_inputs(initial_state)
        state["start_time"] = asyncio.get_event_loop().time()
        state["scores"] = {
            "completeness": 70,
            "relevance": 85,
            "clarity": 90,
            "factual_accuracy": 80,
            "instruction_following": 88,
            "style_tone": 85,
            "context_awareness": 82,
            "safety": 95,
            "overall": 82.5,
        }
        state["ai_likelihood"] = {
            "percentage": 40,
            "classification": "Likely human-written",
            "reasoning": "Natural variation",
        }
        state["summary"] = "Good content overall."
        state["detailed_metrics"] = []
        state["flags"] = []
        state["suggested_changes"] = []
        state["suggestions"] = []
        state["strengths"] = ["Strong clarity"]
        state["improvements"] = []
        
        result = await workflow.compile_final_report(state)
        
        assert "final_report" in result
        report = result["final_report"]
        
        assert "overallScore" in report
        assert "aiLikelihood" in report
        assert "completeness" in report
        assert "metadata" in report


class TestWorkflowIntegration:
    """Integration tests for full workflow execution"""
    
    @pytest.mark.asyncio
    async def test_create_evaluation_graph(self):
        """Test graph creation"""
        graph = create_evaluation_graph()
        
        # Graph should be compiled and ready
        assert graph is not None
    
    @pytest.mark.asyncio
    async def test_run_evaluation_mocked(self):
        """Test full evaluation with mocked agents"""
        # This test would require extensive mocking of all agents
        # In practice, this would be an integration test with actual LLM calls
        # or a pre-recorded response set
        
        with patch('app.workflows.evaluation_workflow.EvaluationWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            MockWorkflow.return_value = mock_instance
            
            # Setup mock methods to return expected state
            async def mock_validate(state):
                return {
                    **state,
                    "metadata": {"word_count": 100, "paragraph_count": 5},
                }
            
            mock_instance.validate_inputs = mock_validate
            
            # The actual test would verify the workflow runs to completion
            # with all nodes executing in the correct order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
