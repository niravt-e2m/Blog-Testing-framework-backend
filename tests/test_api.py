"""API endpoint tests"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


# Test client for synchronous tests
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns correct structure"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "services" in data
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestMetricsDefinitionsEndpoint:
    """Tests for metrics definitions endpoint"""
    
    def test_get_metrics_definitions(self):
        """Test metrics definitions endpoint"""
        response = client.get("/api/v1/metrics/definitions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics" in data
        assert "scoringBands" in data
        assert len(data["metrics"]) == 8
        
        # Check metric structure
        for metric in data["metrics"]:
            assert "name" in metric
            assert "description" in metric
            assert "targetScore" in metric
            assert "weight" in metric


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert data["name"] == "AI Blog Evaluation System"


class TestEvaluateEndpoint:
    """Tests for main evaluation endpoint"""
    
    @pytest.fixture
    def valid_request(self):
        """Valid evaluation request"""
        return {
            "blogText": """
            # Python Best Practices
            
            Python is a versatile programming language used in many domains.
            This guide covers essential best practices for writing clean, 
            maintainable Python code.
            
            ## Code Style
            
            Follow PEP 8 guidelines for consistent code formatting.
            Use meaningful variable names and add docstrings to functions.
            
            ## Testing
            
            Write unit tests for your code using pytest or unittest.
            Aim for high test coverage to catch bugs early.
            
            ## Conclusion
            
            Following these practices will improve your code quality.
            """ * 3,  # Make it longer to pass validation
            "promptText": "Write about Python best practices",
            "toneOfVoice": {
                "type": "text",
                "content": "Professional and educational"
            },
            "references": [],
            "targetAudience": "Python developers",
        }
    
    @pytest.mark.asyncio
    async def test_evaluate_validation_error(self):
        """Test evaluation with invalid request"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/evaluate",
                json={
                    "blogText": "Too short",
                    "toneOfVoice": {"type": "text", "content": "Test"},
                    "targetAudience": "Test",
                }
            )
            
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_evaluate_success_mocked(self, valid_request):
        """Test successful evaluation with mocked workflow"""
        mock_result = {
            "overallScore": 78.5,
            "summary": "Good content overall.",
            "aiLikelihood": 35,
            "metrics": [],
            "flags": [],
            "suggestedChanges": [],
            "suggestions": [],
            "instructionFollowing": 85,
            "factualAccuracy": 80,
            "relevance": 82,
            "completeness": 70,
            "writingStyleTone": 78,
            "clarity": 88,
            "contextAwareness": 75,
            "safety": 95,
            "strengths": ["Clear writing", "Good structure"],
            "improvements": [],
            "metadata": {
                "contentLength": 500,
                "wordCount": 100,
                "paragraphCount": 5,
                "estimatedReadingTime": "1 minute",
                "processingTime": 5.2,
                "evaluationTimestamp": "2024-01-15T10:00:00",
            },
        }
        
        with patch(
            'app.api.routes.run_evaluation',
            new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_result
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(
                    "/api/v1/evaluate",
                    json=valid_request,
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["overallScore"] == 78.5
                assert data["aiLikelihood"] == 35


class TestAsyncEvaluation:
    """Tests for async evaluation functionality"""
    
    @pytest.fixture
    def valid_async_request(self):
        """Valid async evaluation request"""
        return {
            "blogText": "A" * 200,  # Minimum length content
            "promptText": "Test prompt",
            "toneOfVoice": {"type": "text", "content": "Professional"},
            "references": [],
            "targetAudience": "Developers",
            "evaluationOptions": {
                "async_mode": True,
                "include_ai_detection": True,
                "depth_level": "standard"
            }
        }
    
    @pytest.mark.asyncio
    async def test_async_evaluation_returns_job_id(self, valid_async_request):
        """Test async evaluation returns job ID"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/api/v1/evaluate",
                json=valid_async_request,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "jobId" in data
            assert data["status"] == "processing"


class TestQuickEvaluateEndpoint:
    """Tests for quick evaluation endpoint"""
    
    @pytest.mark.asyncio
    async def test_quick_evaluate_mocked(self):
        """Test quick evaluation with mocked workflow"""
        mock_result = {
            "completeness": 70,
            "clarity": 85,
            "safety": 95,
            "aiLikelihood": 30,
            "suggestedChanges": ["Add more examples", "Expand conclusion"],
        }
        
        with patch(
            'app.api.routes.run_evaluation',
            new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_result
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(
                    "/api/v1/evaluate/quick",
                    json={
                        "blogText": "A" * 200,
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "completenessScore" in data
                assert "clarityScore" in data
                assert "safetyScore" in data


class TestReferenceValidationEndpoint:
    """Tests for reference validation endpoint"""
    
    @pytest.mark.asyncio
    async def test_validate_references_mocked(self):
        """Test reference validation with mocked validator"""
        from app.services.reference_validator import ReferenceValidationResult
        
        mock_results = [
            ReferenceValidationResult(
                url="https://example.com/article",
                accessible=True,
                response_time_ms=150,
                content_type="article",
                credibility_score=85.0,
                key_claims=["Claim 1", "Claim 2"],
            ),
            ReferenceValidationResult(
                url="https://invalid.example.com",
                accessible=False,
                error="Connection refused",
            ),
        ]
        
        with patch(
            'app.api.routes.ReferenceValidator.validate_urls',
            new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.return_value = mock_results
            
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(
                    "/api/v1/validate/references",
                    json={
                        "urls": [
                            "https://example.com/article",
                            "https://invalid.example.com",
                        ]
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "results" in data
                assert len(data["results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
