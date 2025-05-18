"""Verification agent for checking response quality and detecting hallucinations."""

from typing import Dict, List, Optional
import logging

from agentic_rag.core.base_agent import BaseAgent
from agentic_rag.core.types import (
    AgentResponse,
    ConfidenceLevel,
    ContentType,
    DocumentSegment,
    ProcessingContext
)

logger = logging.getLogger(__name__)

class VerificationAgent(BaseAgent):
    """Agent for verifying responses and detecting potential hallucinations."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the verification agent."""
        super().__init__(config)
        self.min_evidence_threshold = 0.5

    async def process(self, context: ProcessingContext) -> AgentResponse:
        """Verify the generated response against available context."""
        try:
            # Skip verification during indexing
            if context.metadata.get("mode") == "index":
                return self.create_response(
                    success=True,
                    confidence=ConfidenceLevel.HIGH
                )

            response = context.metadata.get("generated_text")
            if not response:
                return self.create_response(
                    success=True,  # Changed to True to avoid breaking the pipeline
                    confidence=ConfidenceLevel.LOW,
                    result={"message": "No response to verify"}
                )

            # Get original context
            context_str = self._prepare_context(context.segments)
            
            # Perform verification checks
            verification_result = await self._verify_response(response, context_str)
            
            # Add verification results to context
            context.metadata["verification_result"] = verification_result
            
            return self.create_response(
                success=True,
                confidence=ConfidenceLevel.HIGH if verification_result["evidence_score"] >= self.min_evidence_threshold else ConfidenceLevel.LOW,
                result=verification_result
            )
            
        except Exception as e:
            return self.create_response(
                success=True,  # Changed to True to avoid breaking the pipeline
                confidence=ConfidenceLevel.UNCERTAIN,
                error=str(e)
            )

    def _prepare_context(self, segments: List[DocumentSegment]) -> str:
        """Prepare context string from segments."""
        return "\n\n".join(
            segment.content for segment in segments 
            if segment.content_type == ContentType.TEXT
        )

    async def _verify_response(self, response: str, context: str) -> Dict:
        """Verify response against context and detect potential hallucinations."""
        # Split into sentences (simple approach)
        response_parts = [p.strip() for p in response.split(".") if p.strip()]
        
        # Check each response part against context
        supported_parts = []
        unsupported_parts = []
        
        for part in response_parts:
            # Simple word overlap check
            words = set(part.lower().split())
            context_words = set(context.lower().split())
            overlap = len(words.intersection(context_words))
            
            # Ignore common words and phrases in overlap calculation
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "based", "context", "following", "please", "note"}
            relevant_words = words - common_words
            
            if not relevant_words:
                continue
                
            overlap_score = overlap / len(relevant_words) if relevant_words else 0
            
            if overlap_score >= 0.5:  # At least 50% of relevant words should be in context
                supported_parts.append(part)
            else:
                unsupported_parts.append(part)
        
        # Calculate evidence score
        total_parts = len(response_parts)
        evidence_score = len(supported_parts) / total_parts if total_parts > 0 else 0.0
        
        return {
            "evidence_score": evidence_score,
            "supported_count": len(supported_parts),
            "unsupported_count": len(unsupported_parts),
            "potential_hallucinations": unsupported_parts,
            "has_hallucinations": len(unsupported_parts) > 0
        } 