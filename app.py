import re
from itertools import combinations

import networkx as nx
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Smart Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Smart Knowledge Assistant")
st.caption("RAG prototype • Semantic Search • Domain-Specific Knowledge Graphs")
st.write(
    "Upload a PDF or TXT file, ask a question, and inspect the passages "
    "retrieved from your document."
)

with st.sidebar:
    st.header("About this project")
    st.markdown(
        "- **Semantic search:** finds passages by meaning.\n"
        "- **RAG workflow:** retrieves relevant context for answering.\n"
        "- **Knowledge graph:** connects recognized concepts that appear "
        "together in a sentence."
    )
    st.info(
        "This repository includes an extractive-answer fallback. "
        "For generated answers, add a Gemini API key in the optional field."
    )

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="replace")

def split_into_chunks(text, chunk_size=120, overlap=25):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]

CONCEPTS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Neural Networks",
    "Data Science",
    "Semantic Search",
    "Knowledge Graph",
    "Retrieval-Augmented Generation",
    "RAG",
    "Education",
    "Healthcare",
    "Computer Vision",
    "Sentence Embeddings",
]

def build_knowledge_graph(text):
    graph = nx.Graph()
    found = []
    for concept in CONCEPTS:
        if re.search(re.escape(concept), text, flags=re.IGNORECASE):
            found.append(concept)
            graph.add_node(concept)

    # Create an edge when two recognized concepts occur in the same sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        present = [
            concept for concept in found
            if re.search(re.escape(concept), sentence, flags=re.IGNORECASE)
        ]
        for a, b in combinations(present, 2):
            graph.add_edge(a, b)
    return graph

def generate_with_gemini(api_key, question, context):
    """Optional answer generation; returns None if the optional API fails."""
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = (
            "Answer the user's question using only the supplied document context. "
            "If the context does not contain the answer, say that clearly. "
            "Do not invent facts.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return getattr(response, "text", None)
    except Exception as exc:
        st.warning(
            "The optional AI generation request failed. Showing the best "
            f"retrieved passage instead. Details: {exc}"
        )
        return None

uploaded_file = st.file_uploader(
    "Upload a text-based PDF or TXT document",
    type=["pdf", "txt"],
)

api_key = st.text_input(
    "Optional Gemini API key (leave blank to use retrieved passages only)",
    type="password",
    help="The key is used for this session only. Do not commit API keys to GitHub.",
)

if uploaded_file:
    try:
        document_text = extract_text(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.stop()

    if not document_text.strip():
        st.error(
            "No readable text was found. If this is a scanned PDF, run OCR "
            "first or upload a text-based PDF/TXT file."
        )
        st.stop()

    chunks = split_into_chunks(document_text)
    if not chunks:
        st.error("The document did not contain enough readable text.")
        st.stop()

    with st.spinner("Loading the embedding model and processing your document..."):
        try:
            model = load_model()
            embeddings = model.encode(chunks, normalize_embeddings=True)
            graph = build_knowledge_graph(document_text)
        except Exception as exc:
            st.error(
                "The model could not load. Check your internet connection and "
                f"try again. Details: {exc}"
            )
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Text chunks", len(chunks))
    col2.metric("Graph concepts", graph.number_of_nodes())
    col3.metric("Graph connections", graph.number_of_edges())
    st.success("Document processed successfully.")

    question = st.text_input("Ask a question about your document")

    if st.button("Search and Answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching for relevant passages..."):
                q_embedding = model.encode(
                    [question], normalize_embeddings=True
                )
                scores = cosine_similarity(q_embedding, embeddings)[0]
                top_indices = np.argsort(scores)[::-1][:3]
                context = "\n\n".join(chunks[i] for i in top_indices)

            st.subheader("Answer")
            generated = generate_with_gemini(api_key.strip(), question, context)
            if generated:
                st.write(generated)
                st.caption("Generated using the retrieved document context.")
            else:
                st.write(chunks[int(top_indices[0])])
                st.caption(
                    "Extractive fallback: this is the highest-ranked source "
                    "passage, not a newly generated answer."
                )

            st.subheader("Retrieved source passages")
            for rank, idx in enumerate(top_indices, start=1):
                with st.expander(
                    f"Source {rank} • similarity {scores[idx]:.3f}",
                    expanded=(rank == 1),
                ):
                    st.write(chunks[int(idx)])

    st.subheader("Knowledge graph")
    if graph.number_of_nodes():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        positions = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(
            graph,
            positions,
            ax=ax,
            with_labels=True,
            node_size=2200,
            font_size=8,
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Graph edges indicate that two recognized concepts appeared in "
            "the same sentence. This is a simple prototype, not a verified "
            "domain ontology."
        )
    else:
        st.info(
            "No concepts from the prototype's built-in vocabulary were found. "
            "Try the included sample_knowledge.txt file."
        )
else:
    st.info("Upload a PDF or TXT document to begin.")
    st.markdown(
        "### Try these questions with the sample document\n"
        "- What is semantic search?\n"
        "- What is Retrieval-Augmented Generation?\n"
        "- How are AI and Machine Learning related?\n"
        "- What does a knowledge graph represent?"
    )

st.divider()
st.caption("Student project prototype • Verify answers against the displayed sources.")
