"""Unit tests for evaluation agents"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.base_agent import BaseAgent, AgentError, LLMResponseError
from app.agents.content_quality_agent import ContentQualityAgent
from app.agents.style_compliance_agent import StyleComplianceAgent
from app.agents.safety_detection_agent import SafetyDetectionAgent


# Sample test data
SAMPLE_BLOG_CONTENT = """
# 10 Python Tips for Data Scientists

Python has become the go-to language for data science. In this comprehensive guide, 
we'll explore powerful tips that will transform your workflow.

## 1. Use List Comprehensions

List comprehensions are faster and more readable than traditional loops. 
For example: `[x**2 for x in range(10)]` creates a list of squares.

## 2. Leverage Pandas Efficiently

Pandas is essential for data manipulation. Use vectorized operations 
instead of iterating through DataFrames for better performance.

According to recent studies, Python usage among data scientists has increased by 25% 
in the last year. This growth demonstrates the language's dominance in the field.

## Conclusion

These tips will help you write more efficient Python code for your data science projects.
"""

SAMPLE_TONE_GUIDELINES = """
- Professional but approachable tone
- Use clear examples and avoid jargon
- Target intermediate developers
- Keep sentences concise
"""


class TestBaseAgent:
    """Tests for BaseAgent class"""
    
    def test_normalize_score_valid(self):
        """Test score normalization with valid inputs"""
        class ConcreteAgent(BaseAgent):
            @property
            def agent_name(self):
                return "TestAgent"
            
            async def evaluate(self, **kwargs):
                return {}
        
        agent = ConcreteAgent()
        assert agent._normalize_score(75) == 75.0
        assert agent._normalize_score(150) == 100.0
        assert agent._normalize_score(-10) == 0.0
        assert agent._normalize_score("80") == 80.0
        assert agent._normalize_score(None) == 50.0
    
    def test_parse_json_response_valid(self):
        """Test JSON parsing with valid input"""
        class ConcreteAgent(BaseAgent):
            @property
            def agent_name(self):
                return "TestAgent"
            
            async def evaluate(self, **kwargs):
                return {}
        
        agent = ConcreteAgent()
        
        # Direct JSON
        result = agent._parse_json_response('{"score": 85}')
        assert result == {"score": 85}
        
        # JSON in markdown code block
        result = agent._parse_json_response('```json\n{"score": 90}\n```')
        assert result == {"score": 90}
    
    def test_parse_json_response_invalid(self):
        """Test JSON parsing with invalid input"""
        class ConcreteAgent(BaseAgent):
            @property
            def agent_name(self):
                return "TestAgent"
            
            async def evaluate(self, **kwargs):
                return {}
        
        agent = ConcreteAgent()
        
        with pytest.raises(LLMResponseError):
            agent._parse_json_response("This is not JSON")


class TestContentQualityAgent:
    """Tests for ContentQualityAgent"""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for content quality evaluation"""
        return """
        {
            "completeness": {
                "score": 75,
                "analysis": "Good coverage of Python tips",
                "missing_topics": ["error handling", "debugging"],
                "underdeveloped_sections": ["conclusion"],
                "coverage_percentage": 80
            },
            "relevance": {
                "score": 85,
                "analysis": "Content aligns well with title",
                "on_topic_percentage": 90,
                "drift_sections": []
            },
            "clarity": {
                "score": 88,
                "analysis": "Clear and well-structured",
                "readability_level": "Intermediate",
                "flow_issues": []
            },
            "factual_accuracy": {
                "score": 82,
                "analysis": "Most claims appear accurate",
                "total_claims": 5,
                "verified_claims": 4,
                "unverified_claims": ["25% increase statistic"],
                "contradictions": []
            }
        }
        """
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_llm_response):
        """Test successful content quality evaluation"""
        agent = ContentQualityAgent()
        
        # Mock the LLM invocation
        with patch.object(agent, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_llm_response
            
            result = await agent.evaluate(
                blog_title="10 Python Tips",
                blog_content=SAMPLE_BLOG_CONTENT,
                target_audience="Data scientists",
            )
            
            assert "completeness" in result
            assert "relevance" in result
            assert "clarity" in result
            assert "factual_accuracy" in result
            
            assert result["completeness"]["score"] == 75
            assert result["relevance"]["score"] == 85
    
    @pytest.mark.asyncio
    async def test_evaluate_fallback_on_error(self):
        """Test fallback scores when evaluation fails"""
        agent = ContentQualityAgent()
        
        with patch.object(agent, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("LLM error")
            
            result = await agent.safe_evaluate(
                blog_title="Test",
                blog_content=SAMPLE_BLOG_CONTENT,
            )
            
            assert result.get("error") is True
            assert result["completeness"]["score"] == 50


class TestStyleComplianceAgent:
    """Tests for StyleComplianceAgent"""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for style compliance evaluation"""
        return """
        {
            "instruction_following": {
                "score": 92,
                "analysis": "Follows guidelines well",
                "guideline_adherence": 90,
                "violations": [],
                "structural_compliance": true
            },
            "style_tone": {
                "score": 88,
                "analysis": "Consistent professional tone",
                "tone_consistency": 85,
                "voice_match": "Good",
                "inconsistencies": []
            },
            "context_awareness": {
                "score": 80,
                "analysis": "Appropriate for intermediate audience",
                "audience_alignment": 82,
                "terminology_appropriateness": "Appropriate",
                "knowledge_level_match": true
            }
        }
        """
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_llm_response):
        """Test successful style compliance evaluation"""
        agent = StyleComplianceAgent()
        
        with patch.object(agent, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_llm_response
            
            result = await agent.evaluate(
                blog_content=SAMPLE_BLOG_CONTENT,
                tone_guidelines=SAMPLE_TONE_GUIDELINES,
                target_audience="Intermediate developers",
            )
            
            assert "instruction_following" in result
            assert "style_tone" in result
            assert "context_awareness" in result
            
            assert result["instruction_following"]["score"] == 92


class TestSafetyDetectionAgent:
    """Tests for SafetyDetectionAgent"""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for safety detection"""
        return """
        {
            "safety": {
                "score": 95,
                "analysis": "Content is safe and appropriate",
                "is_safe": true,
                "flags": [],
                "biases_detected": [],
                "recommendations": []
            },
            "ai_likelihood": {
                "percentage": 35,
                "classification": "Likely human-written",
                "reasoning": "Content shows personal voice and varied structure",
                "linguistic_markers": [],
                "statistical_analysis": {
                    "sentence_length_variance": "High variation",
                    "vocabulary_diversity": "Good diversity",
                    "pattern_score": "Low pattern presence"
                }
            }
        }
        """
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_llm_response):
        """Test successful safety detection evaluation"""
        agent = SafetyDetectionAgent()
        
        with patch.object(agent, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_llm_response
            
            result = await agent.evaluate(
                blog_content=SAMPLE_BLOG_CONTENT,
                target_audience="General audience",
            )
            
            assert "safety" in result
            assert "ai_likelihood" in result
            
            assert result["safety"]["score"] == 95
            assert result["ai_likelihood"]["percentage"] == 35
    
    def test_classify_ai_likelihood(self):
        """Test AI likelihood classification"""
        agent = SafetyDetectionAgent()
        
        assert agent._classify_ai_likelihood(10) == "Definitely AI"
        assert agent._classify_ai_likelihood(30) == "Likely AI"
        assert agent._classify_ai_likelihood(60) == "Likely human-written"
        assert agent._classify_ai_likelihood(85) == "Definitely human"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
