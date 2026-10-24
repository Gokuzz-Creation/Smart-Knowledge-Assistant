# Smart Knowledge Assistant

**Project title:** Retrieval-Augmented Generation (RAG) System with Semantic Search and Domain-Specific Knowledge Graphs

A student-friendly Streamlit prototype that accepts PDF/TXT documents, retrieves relevant passages using semantic embeddings, displays source passages, and visualizes a simple concept knowledge graph.

## Features

- Upload text-based PDF or TXT files
- Split document text into overlapping chunks
- Generate sentence embeddings with `all-MiniLM-L6-v2`
- Retrieve relevant chunks using cosine similarity
- Display retrieved source passages and similarity scores
- Build a basic concept graph from recognized terms
- Optional answer generation with Gemini when an API key is supplied
- Extractive fallback when no API key is supplied

## Important scope note

The no-key mode retrieves and displays the most relevant passage; it does **not** generate a new answer. The optional Gemini mode uses the retrieved passages as context for generation. The current graph uses a predefined concept vocabulary and links concepts that occur in the same sentence; it is a prototype rather than a fully automated knowledge graph extractor.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit.

## Deploy on Streamlit Community Cloud

1. Upload these project files to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the main file path to `app.py`.
5. Deploy. The first launch may take several minutes while dependencies and the embedding model download.

Do not upload API keys or private documents to GitHub. If using Gemini, enter your API key in the app's password field for the current session. For a production deployment, use Streamlit secrets rather than hard-coding credentials.

## Test with the sample document

Upload `sample_knowledge.txt`, then ask:
- What is semantic search?
- What is Retrieval-Augmented Generation?
- How are AI and Machine Learning related?
- What does a knowledge graph represent?

## Architecture

1. Document upload
2. Text extraction
3. Chunking
4. Embedding generation
5. Semantic similarity retrieval
6. Optional LLM generation using retrieved context
7. Concept graph construction and visualization
8. Answer and source display

## Limitations

- Scanned PDFs may need OCR before upload.
- The first model download requires internet and can take time.
- Retrieval quality depends on document text and chunking.
- Similarity scores are not probabilities.
- The graph recognizes only the concepts listed in the code.
- Generated answers can still be incorrect; check the displayed sources.

## Project report

See `project_report.md`.
