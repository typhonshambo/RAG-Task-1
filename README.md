# Agentic RAG with Multi-Agent Orchestration

A simple implementation of a RAG system that tries to handle images better and reduce hallucinations.

## What it does

- Uses multiple agents to process documents and generate responses
- Tries to handle images without converting them to text
- Attempts to reduce hallucinations through verification
- Basic but functional implementation

### Current Features

- Document processing with basic image detection
- Simple retrieval using sentence-transformers
- Response generation with confidence scoring
- Basic verification to catch obvious hallucinations

### Known Limitations

- Very basic image handling (just detection)
- Simple verification approach
- Limited testing
- No advanced features

## Quick Start

```bash
# Get the code
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Try it out
python examples/basic_usage.py
```

## Basic Usage

```python
from agentic_rag import Pipeline

# Simple query
pipeline = Pipeline()
response = pipeline.query("What are the main types of machine learning?")

# With a document
pipeline.process_document("my_doc.pdf")
response = pipeline.query("What does the diagram show?")
```

## Development Notes

- Built with Python 3.9+
- Uses sentence-transformers for embeddings
- ChromaDB for vector storage
- Basic FastAPI endpoints

### TODO

- [ ] Better image handling
- [ ] Improved verification
- [ ] Documentation improvements

## Documentation
Note : This is a work in progress.
- [Technical Details](docs/technical_architecture.md)
- [API Guide](docs/api.md)
- [Agent Info](docs/agents.md)

## Contributing

It's pretty basic. Feel free to improve it!

