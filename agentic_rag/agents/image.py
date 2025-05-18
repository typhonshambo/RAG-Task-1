"""Image analysis agent for processing and understanding image content."""

from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
import logging
from dataclasses import dataclass

from agentic_rag.core.base_agent import BaseAgent
from agentic_rag.core.types import (
    AgentResponse,
    ConfidenceLevel,
    ContentType,
    DocumentSegment,
    ProcessingContext
)

logger = logging.getLogger(__name__)

@dataclass
class ImageInfo:
    """Basic information about detected images."""
    position: int
    description: str = "Image content detected"
    confidence: float = 0.0

class ImageAnalysisAgent(BaseAgent):
    """Agent for analyzing and understanding image content."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the image analysis agent.
        
        Config options:
            model_name: Name of the vision model to use
            confidence_threshold: Minimum confidence score for predictions
            max_generated_tokens: Maximum length of generated descriptions
        """
        super().__init__(config)
        
        # Load configuration
        self.model_name = self.config.get(
            "model_name", "microsoft/git-base-coco"
        )
        self.confidence_threshold = self.config.get(
            "confidence_threshold", 0.7
        )
        self.max_generated_tokens = self.config.get(
            "max_generated_tokens", 50
        )
        
        # Initialize components
        self._init_model()

    def _init_model(self):
        """Initialize the vision model and processor."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    async def process(self, context: ProcessingContext) -> AgentResponse:
        """Process images in the context.
        
        Args:
            context: The current processing context
            
        Returns:
            AgentResponse with the processing results
        """
        try:
            processed_segments = []
            
            for segment in context.segments:
                if segment.content_type != ContentType.IMAGE:
                    processed_segments.append(segment)
                    continue
                    
                # Analyze the image
                analysis_result = self._analyze_image(segment)
                if analysis_result:
                    description, confidence = analysis_result
                    
                    # Create a new text segment with the image description
                    text_segment = DocumentSegment(
                        id=f"{segment.id}_description",
                        content=description,
                        content_type=ContentType.TEXT,
                        position=segment.position,
                        parent_id=segment.id,
                        metadata={
                            **segment.metadata,
                            "is_image_description": True,
                            "confidence_score": confidence
                        }
                    )
                    
                    # Keep both the original image and its description
                    processed_segments.extend([segment, text_segment])
                else:
                    # If analysis failed, mark for human review
                    segment.requires_human_review = True
                    segment.metadata["analysis_failed"] = True
                    processed_segments.append(segment)
                    
            # Update context with processed segments
            context.segments = processed_segments
            
            return self.create_response(
                success=True,
                confidence=ConfidenceLevel.HIGH,
                result={
                    "processed_images": len([
                        s for s in processed_segments
                        if s.content_type == ContentType.IMAGE
                    ]),
                    "generated_descriptions": len([
                        s for s in processed_segments
                        if s.metadata.get("is_image_description", False)
                    ])
                }
            )
            
        except Exception as e:
            return self.create_response(
                success=False,
                confidence=ConfidenceLevel.UNCERTAIN,
                error=str(e)
            )

    def _analyze_image(
        self, segment: DocumentSegment
    ) -> Optional[Tuple[str, float]]:
        """Analyze an image and generate a description.
        
        Args:
            segment: The image segment to analyze
            
        Returns:
            Tuple of (description, confidence) or None if analysis failed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(segment.content))
            
            # Prepare image for the model
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate description
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_generated_tokens,
                    num_beams=4,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
            # Decode the generated text
            description = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )[0].strip()
            
            # Calculate confidence score
            scores = outputs.sequences_scores.cpu().numpy()
            confidence = float(np.exp(scores[0]))  # Convert log prob to prob
            
            if confidence < self.confidence_threshold:
                return None
                
            return description, confidence
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return None

    def _get_image_features(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract features from an image for potential similarity matching.
        
        Args:
            image: PIL Image to extract features from
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            features = outputs.cpu().numpy()[0]
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting image features: {e}")
            return None

class ImageDetectionAgent(BaseAgent):
    """Agent for detecting and handling images in documents."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the image detection agent."""
        super().__init__(config)

    async def process(self, context: ProcessingContext) -> AgentResponse:
        """Process document segments to detect and handle images."""
        try:
            detected_images = []
            
            for idx, segment in enumerate(context.segments):
                # Simple check for common image markers
                if self._has_image_markers(segment.content):
                    image_info = ImageInfo(
                        position=idx,
                        description="Image content detected - requires human review",
                        confidence=0.8
                    )
                    detected_images.append(image_info)
                    
                    # Add image detection metadata to segment
                    segment.metadata = segment.metadata or {}
                    segment.metadata["has_image"] = True
                    segment.metadata["requires_review"] = True
            
            context.metadata["image_sections"] = [
                {"position": img.position, "description": img.description}
                for img in detected_images
            ]
            
            return self.create_response(
                success=True,
                confidence=ConfidenceLevel.HIGH,
                result={"detected_images": len(detected_images)}
            )
            
        except Exception as e:
            return self.create_response(
                success=False,
                confidence=ConfidenceLevel.UNCERTAIN,
                error=str(e)
            )

    def _has_image_markers(self, content: str) -> bool:
        """Check for common indicators of image content."""
        image_markers = [
            "[image]", "(image)", "figure:", "fig.", 
            "diagram", "chart:", "graph:", "picture:"
        ]
        content_lower = content.lower()
        return any(marker in content_lower for marker in image_markers) 