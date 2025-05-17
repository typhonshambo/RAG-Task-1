"""Retrieval agent for semantic search and content retrieval."""

from typing import Dict, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from agentic_rag.core.base_agent import BaseAgent
from agentic_rag.core.types import (
    AgentResponse,
    ConfidenceLevel,
    ContentType,
    DocumentSegment,
    ProcessingContext
)


class RetrievalAgent(BaseAgent):
    """Agent for semantic search and content retrieval."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the retrieval agent."""
        super().__init__(config)
        
        # Use smallest model available
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.collection_name = "rag_collection"
        self.top_k = 5  # Retrieve more context
        
        # Initialize components
        self._init_retrieval_system()

    def _init_retrieval_system(self):
        """Initialize the embedding model and vector store."""
        # Initialize with CPU
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device="cpu"
        )
        
        # Use in-memory ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            is_persistent=False,  # In-memory only
            anonymized_telemetry=False
        ))
        
        # Create new collection each time
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    async def process(self, context: ProcessingContext) -> AgentResponse:
        """Process the context for retrieval or indexing."""
        try:
            if context.metadata.get("mode") == "index":
                return await self._handle_indexing(context)
            return await self._handle_retrieval(context)
            
        except Exception as e:
            return self.create_response(
                success=False,
                confidence=ConfidenceLevel.UNCERTAIN,
                error=str(e)
            )

    async def _handle_indexing(self, context: ProcessingContext) -> AgentResponse:
        """Handle document indexing."""
        indexed_count = 0
        
        for segment in context.segments:
            if segment.content_type != ContentType.TEXT:
                continue
                
            try:
                # Clean and normalize text
                text = segment.content.strip()
                if not text:
                    continue
                    
                embeddings = await self._get_embeddings([text])
                # Convert numpy array to list for ChromaDB
                embeddings_list = embeddings.tolist()
                
                self.collection.add(
                    documents=[text],
                    embeddings=embeddings_list,
                    ids=[segment.id],
                    metadatas=[{"position": segment.position}]
                )
                
                indexed_count += 1
            except Exception as e:
                continue  # Skip failed segments
            
        return self.create_response(
            success=True,
            confidence=ConfidenceLevel.HIGH,
            result={"indexed_segments": indexed_count}
        )

    async def _handle_retrieval(self, context: ProcessingContext) -> AgentResponse:
        """Handle query retrieval."""
        query = context.metadata.get("query")
        if not query:
            return self.create_response(
                success=False,
                confidence=ConfidenceLevel.UNCERTAIN,
                error="No query provided"
            )
            
        try:
            query_embedding = await self._get_embeddings([query])
            # Convert numpy array to list for ChromaDB
            query_embedding_list = query_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding_list,
                n_results=self.top_k
            )
            
            if not results["documents"]:
                return self.create_response(
                    success=True,
                    confidence=ConfidenceLevel.LOW,
                    result={"message": "No relevant documents found"}
                )
                
            # Sort by position to maintain document order
            retrieved_segments = []
            for doc, metadata in sorted(
                zip(results["documents"][0], results["metadatas"][0]),
                key=lambda x: x[1]["position"]
            ):
                segment = DocumentSegment(
                    id=metadata.get("id", ""),
                    content=doc,
                    content_type=ContentType.TEXT,
                    position=metadata.get("position", 0)
                )
                retrieved_segments.append(segment)
                
            context.segments = retrieved_segments
            
            return self.create_response(
                success=True,
                confidence=ConfidenceLevel.HIGH,
                result={"retrieved_segments": len(retrieved_segments)}
            )
        except Exception as e:
            return self.create_response(
                success=False,
                confidence=ConfidenceLevel.UNCERTAIN,
                error=str(e)
            )

    async def _get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        ) 