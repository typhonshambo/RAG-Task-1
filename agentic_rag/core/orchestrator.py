"""Orchestrator for managing the flow of processing between agents."""

from typing import Dict, List
import logging

from agentic_rag.core.base_agent import BaseAgent
from agentic_rag.core.types import ProcessingContext

class Orchestrator:
    """Manages the flow of processing between multiple agents."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(__name__)

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            name: Name to register the agent under
            agent: The agent instance to register
        """
        self.agents[name] = agent
        self.logger.info(f"Registered agent {name}: {agent.__class__.__name__}")

    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process the context through all registered agents."""
        for name, agent in self.agents.items():
            try:
                response = await agent.process(context)
                context.agent_responses.append(response)
                
                if not response.success:
                    context.status = f"failed_at_{name}"
                    self.logger.error(f"Agent {name} failed: {response.error}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error processing with agent {name}: {e}")
                context.status = f"error_at_{name}"
                break

        return context

    def reset(self) -> None:
        """Reset the orchestrator by clearing all registered agents."""
        self.agents.clear()
        self.logger.info("Reset orchestrator") 