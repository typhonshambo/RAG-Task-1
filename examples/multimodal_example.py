"""Example script demonstrating multimodal content handling in the Agentic RAG system."""

import asyncio
from pathlib import Path
import logging
from PIL import Image
import io
import base64

from agentic_rag import Pipeline
from agentic_rag.agents.document import DocumentProcessor
from agentic_rag.agents.retrieval import RetrievalAgent
from agentic_rag.agents.image import ImageAnalysisAgent
from agentic_rag.agents.verification import VerificationAgent
from agentic_rag.agents.generation import GenerationAgent
from agentic_rag.core.types import ContentType


def create_sample_image() -> bytes:
    """Create a simple test image.
    
    Returns:
        Image bytes
    """
    # Create a simple image with some shapes
    img = Image.new('RGB', (200, 200), color='white')
    
    # Add some sample content (you would replace this with real diagrams)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a simple neural network diagram
    # Nodes
    node_positions = [
        [(50, y) for y in range(50, 200, 50)],  # Input layer
        [(100, y) for y in range(25, 200, 25)],  # Hidden layer 1
        [(150, y) for y in range(50, 200, 50)]   # Output layer
    ]
    
    # Draw nodes
    for layer in node_positions:
        for x, y in layer:
            draw.ellipse([x-10, y-10, x+10, y+10], outline='black')
            
    # Draw connections
    for i, layer in enumerate(node_positions[:-1]):
        next_layer = node_positions[i + 1]
        for x1, y1 in layer:
            for x2, y2 in next_layer:
                draw.line([x1, y1, x2, y2], fill='blue', width=1)
                
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


async def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Initialize the pipeline
    pipeline = Pipeline()
    
    # Add agents in processing order
    pipeline.add_agent(DocumentProcessor())
    pipeline.add_agent(ImageAnalysisAgent())
    pipeline.add_agent(RetrievalAgent())
    pipeline.add_agent(VerificationAgent())
    pipeline.add_agent(GenerationAgent())
    
    # Create a sample document with mixed content
    text_content = """
    # Deep Learning Architecture Analysis
    
    This document analyzes various deep learning architectures and their
    applications in modern AI systems.
    
    ## Neural Network Visualization
    
    Below is a diagram showing a basic feedforward neural network architecture:
    """
    
    # Create the image
    image_content = create_sample_image()
    
    # Process text first
    logger.info("Processing text content...")
    context = await pipeline.process_document(
        text_content,
        metadata={"mode": "index"}
    )
    
    # Process image
    logger.info("Processing image content...")
    context = await pipeline.process_document(
        image_content,
        document_type=ContentType.IMAGE,
        metadata={"mode": "index"}
    )
    
    # Example queries to test the system
    queries = [
        "Describe the neural network architecture shown in the diagram.",
        "What are the key components visible in the network visualization?",
        "How many layers does the neural network have and how are they connected?"
    ]
    
    # Process each query
    for query in queries:
        logger.info(f"\nProcessing query: {query}")
        context = await pipeline.process_document(
            query,
            metadata={"query": query}
        )
        
        if context.status == "completed":
            response = context.metadata.get("generated_text")
            logger.info(f"\nQuery: {query}")
            logger.info(f"Response: {response}")
            
            # Log verification results if available
            verification_results = next(
                (r for r in context.agent_responses
                 if "verification_results" in (r.result or {})),
                None
            )
            if verification_results and verification_results.result:
                results = verification_results.result
                logger.info("\nVerification Results:")
                logger.info(f"Total Claims: {results['total_claims']}")
                logger.info(f"Verified Claims: {results['verified_claims']}")
                logger.info(f"Contradicted Claims: {results['contradicted_claims']}")
                logger.info(f"Uncertain Claims: {results['uncertain_claims']}")
        else:
            logger.error(f"Processing failed: {context.status}")


if __name__ == "__main__":
    asyncio.run(main()) 