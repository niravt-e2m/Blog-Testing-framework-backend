"""Base agent class for LLM-powered evaluation agents"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from app.config import get_settings, Settings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class AgentError(Exception):
    """Base exception for agent errors"""
    pass


class LLMResponseError(AgentError):
    """Error parsing LLM response"""
    pass


class BaseAgent(ABC):
    """Base class for evaluation agents"""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            settings: Application settings. If None, loads from environment.
            llm: Pre-configured LLM instance. If None, creates based on settings.
        """
        self.settings = settings or get_settings()
        self.llm = llm or self._create_llm()
        self._retry_count = 3
        self._retry_delay = 1.0
    
    def _create_llm(self) -> BaseChatModel:
        """Create LLM instance based on settings"""
        if self.settings.llm_provider == "openrouter":
            return ChatOpenAI(
                model=self.settings.openrouter_model,
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                temperature=0.3,
                max_tokens=4096,
            )
        return ChatOpenAI(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            temperature=0.3,
            max_tokens=4096,
        )
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the agent's name for logging"""
        pass
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Perform evaluation. Must be implemented by subclasses.
        
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    async def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Invoke the LLM with retry logic.
        
        Args:
            system_prompt: System message for the LLM
            user_prompt: User message/query
            
        Returns:
            Raw response text from LLM
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        last_error = None
        for attempt in range(self._retry_count):
            try:
                logger.debug(f"{self.agent_name}: Invoking LLM (attempt {attempt + 1})")
                response = await self.llm.ainvoke(messages)
                if self.settings.debug:
                    preview = (response.content or "")[:500]
                    logger.info(
                        f"{self.agent_name}: LLM response received (len={len(response.content or '')}) "
                        f"preview={preview}"
                    )
                return response.content
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.agent_name}: LLM invocation failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self._retry_count - 1:
                    import asyncio
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
        
        raise AgentError(f"LLM invocation failed after {self._retry_count} attempts: {last_error}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling common formatting issues.
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            Parsed JSON as dictionary
        """
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Log the problematic response for debugging
        logger.error(f"{self.agent_name}: Failed to parse JSON from response: {response[:500]}...")
        raise LLMResponseError(f"Could not parse JSON from LLM response")
    
    def _validate_response(
        self,
        response: Dict[str, Any],
        response_model: Type[T],
    ) -> T:
        """
        Validate response against a Pydantic model.
        
        Args:
            response: Parsed JSON response
            response_model: Pydantic model class for validation
            
        Returns:
            Validated Pydantic model instance
        """
        try:
            return response_model.model_validate(response)
        except Exception as e:
            logger.error(f"{self.agent_name}: Response validation failed: {e}")
            raise LLMResponseError(f"Response validation failed: {e}")
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """
        Get default/fallback scores when evaluation fails.
        Override in subclasses for agent-specific defaults.
        """
        return {
            "error": True,
            "error_message": "Evaluation failed, using default scores",
            "score": 50,
        }
    
    async def safe_evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate with error handling, returning fallback on failure.
        
        Returns:
            Evaluation results or fallback defaults
        """
        try:
            return await self.evaluate(**kwargs)
        except Exception as e:
            logger.error(f"{self.agent_name}: Evaluation failed: {e}")
            defaults = self._get_default_scores()
            defaults["error_message"] = str(e)
            return defaults
    
    def _normalize_score(self, score: Any, min_val: float = 0, max_val: float = 100) -> float:
        """Normalize a score to ensure it's within valid range"""
        try:
            score = float(score)
            return max(min_val, min(max_val, score))
        except (TypeError, ValueError):
            return 50.0  # Default middle score on conversion failure
    
    def _extract_evidence(self, text: str, max_items: int = 5) -> list:
        """Extract evidence snippets from analysis text"""
        # Look for quoted text or specific examples
        quotes = re.findall(r'"([^"]+)"', text)
        if quotes:
            return quotes[:max_items]
        
        # Fall back to extracting key sentences
        sentences = text.split('.')
        evidence = [s.strip() for s in sentences if len(s.strip()) > 20]
        return evidence[:max_items]
