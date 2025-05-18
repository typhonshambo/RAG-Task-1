"""Example usage of the Agentic RAG system."""

import logging
from agentic_rag.core.orchestrator import Orchestrator
from agentic_rag.agents.document import DocumentProcessor
from agentic_rag.agents.retrieval import RetrievalAgent
from agentic_rag.agents.generation import GenerationAgent
from agentic_rag.agents.verification import VerificationAgent
from agentic_rag.core.types import ProcessingContext, ContentType, DocumentSegment
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize orchestrator and register agents
    orchestrator = Orchestrator()
    orchestrator.register_agent("document", DocumentProcessor())
    orchestrator.register_agent("retrieval", RetrievalAgent())
    orchestrator.register_agent("generation", GenerationAgent())
    orchestrator.register_agent("verification", VerificationAgent())

    # Create sample document with text and image markers
    document = """
    Machine Learning Types and Applications

    Here are the main types of machine learning:
    1. Supervised Learning: Training with labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through trial and error

    Machine learning is used in many fields including:
    - Image recognition
    - Natural language processing
    - Recommendation systems
    - Fraud detection

    [Image: Neural Network Architecture Diagram]
    Figure 1: A typical neural network showing input layer, hidden layers, and output layer.

    Key Components of Neural Networks:
    - Neurons (nodes)
    - Weights and biases
    - Activation functions
    """

    # Create processing context
    context = ProcessingContext(
        document_id="doc1",  # Required field
        segments=[
            DocumentSegment(
                id="seg1",
                content=document,
                content_type=ContentType.TEXT,
                position=0
            )
        ]
    )

    # Index document
    logger.info("Indexing document...")
    context.metadata["mode"] = "index"
    await orchestrator.process(context)

    # Query examples
    queries = [
        "What are the main types of machine learning?",
        "Explain the neural network diagram.",
        "What are the key components of neural networks?"
    ]

    for query in queries:
        # Create new context for each query
        query_context = ProcessingContext(
            document_id=f"query_{query[:20]}",  # Use part of query as ID
            segments=[
                DocumentSegment(
                    id="query_seg",
                    content=query,
                    content_type=ContentType.TEXT,
                    position=0
                )
            ]
        )
        
        # Process query
        query_context.metadata["mode"] = "query"
        query_context.metadata["query"] = query
        await orchestrator.process(query_context)
        
        # Get response
        response = query_context.metadata.get("generated_text", "No response generated")
        logger.info(f"\nQuery: {query}\n\nResponse: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())