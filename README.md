# Agentic-AI-Prototype

This project is an intelligent, agentic chatbot designed to provide both emotional support and logical assistance. It is powered by Google's Gemini LLM and built using LangGraph, RAG-based retrieval, and PostgreSQL for memory.

## Key Features

- **Emotion Classification**: Distinguishes between emotional and logical messages.
- **RAG System**: Uses LangGraph + Chroma + PDF knowledge base.
- **Long-Term Memory**: Stores conversations and user context in a local PostgreSQL DB.
- **Therapist Support**: If the user frequently expresses emotional distress, the bot proactively suggests therapists based on their location.
- **Live Search Integration**: Searches the web for therapist contacts using Google Search API.
- **Contextual Memory**: Summarizes past conversations and personal details to maintain continuity.

## Tech Stack

- **LLM**: Google Gemini via API
- **Memory**: PostgreSQL + LangGraph checkpointing
- **Embedding**: Ollama LLaMA 3 (locally)
- **RAG**: Chroma vector store with PDF ingestion
- **Search Tools**: Google Search API wrapper
