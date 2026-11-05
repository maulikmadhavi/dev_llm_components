# LLM Development Project

A collection of tools and workflows for working with large language models, video processing, and AI applications.

## Overview

This project includes:
- VLM (Vision Language Model) applications for video and image analysis
- ASR (Automatic Speech Recognition) training with Whisper
- RAG (Retrieval Augmented Generation) systems with LangChain
- Fine-tuning scripts using TRL and transformers
- Data generation and processing utilities

## Project Structure

```
vlm_application/     - Video and image processing with VLM APIs
langchain/          - RAG and research agent workflows
ASR/                - Speech recognition model training
gradio_app/        - Node backend + React Flow canvas API (see gradio_app/README.md for chat assistant settings)
finetuning/         - Model fine-tuning scripts
n8n/                - Integration utilities
asr_text_collection.py - ASR data collection
```

## Requirements

- Python 3.10+
- pytorch, transformers, langchain
- See individual script imports for specific dependencies

## Usage

1. Set up environment variables in `config.yaml`
2. Run individual scripts from their directories
3. Check each module's docstrings for function usage

## Key Files

- `vlm_application/utils.py` - Video processing and API utilities
- `langchain/nvidia_rag.py` - RAG pipeline implementation
- `ASR/train.py` - Whisper model training
- `finetuning/trl_dpo.py` - DPO fine-tuning example

## Notes

All functions include type hints and docstrings for easy understanding and IDE support.
