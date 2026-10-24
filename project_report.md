# Project Report

## Title
Retrieval-Augmented Generation (RAG) System with Semantic Search and Domain-Specific Knowledge Graphs

## Abstract
This project presents a document question-answering prototype that combines semantic retrieval with a simple domain-specific knowledge graph. Users upload text-based PDF or TXT documents. The system extracts text, divides it into overlapping chunks, encodes the chunks with a sentence-embedding model, and retrieves relevant passages using cosine similarity. It also identifies concepts from a predefined vocabulary and connects concepts that appear in the same sentence. An optional Gemini integration can generate an answer using retrieved passages as context; without an API key, the system displays the best matching passage as an extractive fallback.

## Problem Statement
Keyword-only search may fail when a user's question and a document use different wording. A semantic retrieval system can compare meaning-related embeddings and retrieve more relevant passages. A graph can also make relationships between recognized domain concepts easier to inspect.

## Objectives
1. Build a document upload and text extraction interface.
2. Implement chunking and semantic embedding generation.
3. Retrieve relevant document passages for a user question.
4. Display source passages to support answer verification.
5. Construct and visualize a simple concept relationship graph.
6. Provide optional context-grounded language generation.

## Tools and Technologies
- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn cosine similarity
- PyPDF
- NetworkX
- Matplotlib
- Optional Gemini API for generated answers

## Methodology
1. Upload a PDF or TXT file.
2. Extract the document text.
3. Divide the text into overlapping chunks.
4. Generate normalized sentence embeddings for the chunks.
5. Encode the question and calculate cosine similarity against chunk embeddings.
6. Retrieve the top three passages.
7. If configured, send the retrieved passages and question to a language model; otherwise, show the top passage.
8. Recognize built-in domain concepts and connect concepts appearing in the same sentence.
9. Display the answer, source passages, and graph through Streamlit.

## System Modules
- Document ingestion
- Text chunking
- Semantic search
- Source retrieval
- Optional answer generation
- Knowledge graph construction
- Web interface

## Expected Results
The application should accept supported documents, retrieve passages related to a question, show similarity scores and source text, and visualize recognized concepts. Actual results should be recorded after running the app and testing it with several questions.

## Limitations
The initial graph uses a predefined list of concepts and sentence co-occurrence rather than advanced entity and relation extraction. The no-key mode is extractive, not generative. Scanned PDFs may require OCR. Optional LLM output must be checked against source passages.

## Future Enhancements
- Automated entity and relation extraction
- Graph-aware retrieval
- Persistent vector database
- Support for multiple documents
- More robust citation and page-number tracking
- Evaluation using retrieval metrics and a question-answer test set

## Conclusion
The project demonstrates the core stages of semantic document retrieval and a simple graph representation in an accessible web interface. The prototype can be extended into a more complete RAG system by improving knowledge graph extraction, adding graph-aware retrieval, and evaluating generated answers against source documents.
