"""Pytest configuration and fixtures"""

import pytest
import asyncio
from typing import Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_blog_content() -> str:
    """Sample blog content for testing"""
    return """
    # Introduction to Machine Learning
    
    Machine learning is a fascinating field that has transformed technology.
    In this comprehensive guide, we will explore the fundamentals and provide
    practical examples for beginners.
    
    ## What is Machine Learning?
    
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    
    ## Types of Machine Learning
    
    There are three main types:
    1. Supervised Learning
    2. Unsupervised Learning  
    3. Reinforcement Learning
    
    ## Conclusion
    
    Machine learning offers powerful tools for solving complex problems.
    """


@pytest.fixture
def sample_tone_guidelines() -> str:
    """Sample tone guidelines for testing"""
    return """
    - Professional but approachable
    - Use clear examples
    - Target intermediate developers
    - Avoid unnecessary jargon
    """


@pytest.fixture
def sample_evaluation_request() -> dict:
    """Sample evaluation request for testing"""
    return {
        "blogText": """
        # Python Best Practices Guide
        
        Python has become one of the most popular programming languages.
        This guide covers essential best practices for Python developers.
        
        ## Code Style
        
        Following PEP 8 ensures consistent, readable code.
        Use meaningful variable names and add docstrings.
        
        ## Testing
        
        Write comprehensive tests using pytest.
        Aim for high code coverage.
        
        ## Performance
        
        Profile your code before optimizing.
        Use appropriate data structures.
        
        ## Conclusion
        
        Following these practices improves code quality significantly.
        """ * 2,
        "promptText": "Write about Python best practices",
        "toneOfVoice": {
            "type": "text",
            "content": "Professional and educational tone"
        },
        "references": [],
        "targetAudience": "Python developers"
    }
