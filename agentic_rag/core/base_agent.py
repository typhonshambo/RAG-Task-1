"""Base agent class that defines the interface for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import uuid

from agentic_rag.core.types import (
    AgentResponse,
    ConfidenceLevel,
    ProcessingContext
)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with optional configuration."""
        self.agent_id = str(uuid.uuid4())
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    async def process(self, context: ProcessingContext) -> AgentResponse:
        """Process the current context and return a response.
        
        Args:
            context: The current processing context containing document segments
                    and previous agent responses.
        
        Returns:
            AgentResponse containing the results of processing.
        """
        pass

    def create_response(
        self,
        success: bool,
        confidence: ConfidenceLevel,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> AgentResponse:
        """Create a standardized agent response.
        
        Args:
            success: Whether the processing was successful
            confidence: Confidence level in the result
            result: Optional dictionary containing the processing result
            error: Optional error message if processing failed
        
        Returns:
            A standardized AgentResponse object
        """
        return AgentResponse(
            agent_id=self.agent_id,
            success=success,
            confidence=confidence,
            result=result or {},
            error=error
        )

    def validate_config(self, required_keys: list[str]) -> None:
        """Validate that the agent has all required configuration keys.
        
        Args:
            required_keys: List of required configuration keys
        
        Raises:
            ValueError: If any required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys for {self.name}: {missing_keys}"
            )

    async def pre_process(self, context: ProcessingContext) -> None:
        """Hook for any preprocessing steps needed before main processing.
        
        Args:
            context: The current processing context
        """
        pass

    async def post_process(
        self, context: ProcessingContext, response: AgentResponse
    ) -> None:
        """Hook for any postprocessing steps needed after main processing.
        
        Args:
            context: The current processing context
            response: The response from the main processing step
        """
        pass

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name}(id={self.agent_id})" 