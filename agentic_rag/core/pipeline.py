"""Main pipeline class for the Agentic RAG system."""

from typing import Any, Dict, Optional, Union
import uuid
import logging

from agentic_rag.core.orchestrator import Orchestrator
from agentic_rag.core.types import ProcessingContext, DocumentSegment, ContentType

class Pipeline:
    """Main interface for the Agentic RAG system."""

    def __init__(self):
        """Initialize the pipeline."""
        self.orchestrator = Orchestrator()
        self.logger = logging.getLogger(__name__)

    async def process_document(
        self,
        document: Union[str, bytes],
        metadata: Optional[Dict] = None,
        document_type: ContentType = ContentType.TEXT
    ) -> ProcessingContext:
        """Process a document through the RAG pipeline."""
        # Create a new processing context
        context = ProcessingContext(
            document_id=str(uuid.uuid4()),
            metadata=metadata or {}
        )
        
        # Initialize segments with the input document
        context.segments = [
            DocumentSegment(
                id=str(uuid.uuid4()),
                content=document,
                content_type=document_type,
                position=0
            )
        ]
        
        # Process through orchestrator
        try:
            context = await self.orchestrator.process(context)
            if context.status == "initialized":
                context.status = "completed"
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            context.status = "failed"
            context.metadata["error"] = str(e)
            
        return context

    def add_agent(self, agent: Any, name: Optional[str] = None) -> None:
        """Add an agent to the pipeline.
        
        Args:
            agent: The agent instance to add
            name: Optional name for the agent. If not provided, will use the class name
        """
        agent_name = name or agent.__class__.__name__.lower()
        self.orchestrator.register_agent(agent_name, agent)

    def reset(self) -> None:
        """Reset the pipeline by clearing all agents."""
        self.orchestrator.reset() 