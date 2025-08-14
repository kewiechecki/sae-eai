# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SAE (Sparse Autoencoders) chat application that combines EleutherAI's sparsify library with RAG (Retrieval-Augmented Generation) capabilities for medical diagnostics. The system uses DeepSeek-R1-Distill-Qwen-1.5B as the base model with SAE interpretability features.

## Architecture

### Core Components

- **`featurized.py`**: Contains the `Featurized` class that handles SAE model loading, tokenization, and layer-wise feature extraction using EleutherAI's sparsify library
- **`chat.py`**: Implements the `Chat` class for conversational AI with medical diagnostics focus, integrating both the featurized model and RAG database
- **`ragdb.py`**: Provides the `RAGDB` class for vector database operations using Milvus, handling document chunking and similarity search
- **`main.py`**: Entry point with model configuration and example usage patterns for SAE feature analysis
- **`mkdb.py`**: Utility for creating and populating the RAG database from text documents
- **`embed.py`** and **`embed_layer.py`**: Lower-level embedding utilities for SAE layer analysis
- **`qwen2hook.py`**: Model patching and hook utilities for intercepting model internals
- **`features.py`**: Feature processing utilities for SAE analysis

### Data Flow

1. Text input is tokenized and passed through the SAE-enhanced model
2. Hidden states from specified layers are extracted and processed through SAE encoders
3. RAG system retrieves relevant context from medical flowcharts/documentation
4. Chat system combines SAE interpretability with retrieved context for responses

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment with SAE dependencies
make sae-eai

# Alternative Goodfire environment setup
make goodfire

# Clean environments
make clean
```

### Package Management
This project uses `uv` for dependency management. Dependencies are defined in `pyproject.toml`:
```bash
# Install dependencies (handled by make targets)
uv sync
```

### Running the Application
```bash
# Main entry point
python main.py

# Direct chat interface
python chat.py

# Database creation
python mkdb.py
```

## Key Dependencies

- **eai-sparsify**: EleutherAI's SAE library (from git)
- **transformers**: HuggingFace transformers for base model
- **pymilvus**: Vector database for RAG functionality
- **sentence-transformers**: For text embeddings (BAAI/bge-small-en-v1.5)
- **torch**: Deep learning framework

## Model Configuration

- **Base Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **SAE Model**: `EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k`
- **Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Database**: Uses Milvus with IP (Inner Product) distance metric

## Data Sources

The system processes medical diagnostic flowcharts (`flowcharts.md`) for RAG context. The database is stored in `hf_milvus.db` with collection name "flowcharts".

## Key Implementation Notes

### Class Interfaces

- **`Featurized`**: Initialize with `device`, `layer_idx`, `model_path`, and `sae_path`. Provides `embed()` and `unembed()` methods for feature analysis
- **`Chat`**: Accepts `featurized` model instance and optional `ragdb`. Implements diagnostic system prompt for medical analysis
- **`RAGDB`**: Initialize with `db_path` and `collection_name`. Provides `search()` method for context retrieval

### Common Patterns

- Device management: Most classes accept a `device` parameter (defaults to "cuda" if available)
- Layer selection: SAE analysis typically uses layers 5-11 of the 27-layer model
- Context window: Default chunk size is 512 tokens with 50 token overlap for RAG
- Feature dimensions: SAE encoders produce 65536-dimensional sparse features