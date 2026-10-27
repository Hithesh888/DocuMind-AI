
import io
import uuid
from pathlib import Path

import chromadb
import ollama
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "llama3.2:1b"
EMBEDDING_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: #F7F5ED !important;
    color: #45483F !important;
}

[data-testid="stHeader"] {
    background: #F7F5ED !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #FFFFFF !important;
    color: #45483F !important;
    border-right: 1px solid #E7E4D9 !important;
}

[data-testid="stSidebar"] * {
    color: #45483F;
}

.hero {
    background: linear-gradient(135deg, #FFFFFF, #F0F4E9);
    border: 1px solid #E3E8DA;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 22px;
}

.eyebrow {
    color: #718567;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.hero h1 {
    font-size: 34px;
    line-height: 1.25;
    color: #424A3C;
    margin: 10px 0;
}

.hero p {
    color: #77796F;
    font-size: 15px;
    margin-bottom: 0;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E7E4D9;
    border-radius: 16px;
    padding: 18px;
    min-height: 100px;
}

.metric-label {
    font-size: 13px;
    color: #858579;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #4E5E46;
    margin-top: 8px;
}

.section-title {
    color: #45483F;
    font-size: 19px;
    font-weight: 700;
    margin: 10px 0 14px;
}

.sidebar-brand {
    font-size: 24px;
    font-weight: 800;
    color: #4D5D44;
    letter-spacing: -0.8px;
}

.sidebar-subtitle {
    color: #8C8C7F;
    font-size: 12px;
    margin-top: 2px;
}

.info-note {
    color: #858579;
    font-size: 12px;
    line-height: 1.6;
}

.footer {
    color: #969587;
    font-size: 12px;
    text-align: center;
    padding: 24px 0 8px;
}

/* Buttons */
.stButton > button {
    background: #E7EEDF !important;
    color: #4D6045 !important;
    border: 1px solid #D7E1CD !important;
    border-radius: 11px !important;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #D9E6CF !important;
    color: #394B32 !important;
    border-color: #C6D6BA !important;
}

.stButton > button[kind="primary"] {
    background: #829875 !important;
    color: #FFFFFF !important;
    border-color: #829875 !important;
}

/* Chat messages */
[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"],
.stChatMessage {
    background: #FFFFFF !important;
    color: #45483F !important;
    border: 1px solid #E7E4D9 !important;
    border-radius: 16px !important;
}

/* Chat input */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #45483F !important;
    border-color: #D7E1CD !important;
    box-shadow: none !important;
    -webkit-text-fill-color: #45483F !important;
}

/* Text inputs */
input,
textarea,
[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-baseweb="textarea"],
[data-baseweb="textarea"] > div {
    background: #FFFFFF !important;
    color: #45483F !important;
    border-color: #D7E1CD !important;
    box-shadow: none !important;
    -webkit-text-fill-color: #45483F !important;
}

input::placeholder,
textarea::placeholder {
    color: #969587 !important;
    -webkit-text-fill-color: #969587 !important;
}

/* FIX: Sidebar document selection dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #F0F4E9 !important;
    border: 1px solid #D7E1CD !important;
    border-radius: 10px !important;
    color: #4D5D44 !important;
    box-shadow: none !important;
}

/* FIX: Selected document tags instead of red tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #DDE8D3 !important;
    color: #45583D !important;
    border: 1px solid #C8D8BD !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #45583D !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] svg {
    fill: #617654 !important;
    color: #617654 !important;
}

/* Dropdown text and icons */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #4D5D44 !important;
    -webkit-text-fill-color: #4D5D44 !important;
}

/* Dropdown menus and options */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #FFFFFF !important;
    color: #45483F !important;
}

[role="option"]:hover {
    background-color: #F0F4E9 !important;
}

/* File uploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {
    background: #F0F4E9 !important;
    color: #45483F !important;
    border-color: #D7E1CD !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #E7EEDF !important;
    color: #4D6045 !important;
    border-color: #D7E1CD !important;
}

/* Expanders and bordered containers */
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > summary,
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    color: #45483F !important;
    border-color: #E7E4D9 !important;
}

/* FIX: chroma_db inline code label */
[data-testid="stSidebar"] code {
    background: #E7EEDF !important;
    color: #4D6045 !important;
    border: 1px solid #D7E1CD !important;
    border-radius: 5px !important;
    padding: 2px 5px !important;
}

/* Alerts and metrics */
[data-testid="stMetric"],
[data-testid="stAlert"] {
    background: #FFFFFF !important;
    color: #45483F !important;
    border-color: #E7E4D9 !important;
}

/* General text */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"],
label {
    color: #45483F;
}

hr {
    border-color: #E7E4D9 !important;
}

:focus-visible {
    outline-color: #829875 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_resources():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
    model = SentenceTransformer(EMBEDDING_NAME)
    return collection, model


def split_text(text):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def extract_text(uploaded_file):
    filename = uploaded_file.name
    raw = uploaded_file.getvalue()
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(f"[Page {page_number}]\n{page_text}")

        return "\n\n".join(pages)

    if suffix == ".txt":
        return raw.decode("utf-8", errors="replace")

    if suffix == ".csv":
        dataframe = pd.read_csv(io.BytesIO(raw)).fillna("")
        return dataframe.to_string(index=False)

    raise ValueError("Upload a PDF, TXT, or CSV file.")


def index_document(uploaded_file, collection, embedding_model):
    filename = uploaded_file.name
    text = extract_text(uploaded_file)
    chunks = split_text(text)

    if not chunks:
        raise ValueError(
            "No readable text found. Scanned PDFs may require OCR."
        )

    existing = collection.get(
        where={"source": filename},
        include=["metadatas"],
    )
    old_ids = existing.get("ids", [])

    if old_ids:
        collection.delete(ids=old_ids)

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
    ).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": filename, "chunk": i + 1}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def get_document_names(collection):
    result = collection.get(include=["metadatas"])
    names = set()

    for metadata in result.get("metadatas", []):
        if metadata and metadata.get("source"):
            names.add(metadata["source"])

    return sorted(names)


def retrieve(question, selected_documents, collection, embedding_model):
    if not selected_documents:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where={"source": {"$in": selected_documents}},
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        retrieved.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "chunk": metadata.get("chunk", "?"),
            "distance": distance,
        })

    return retrieved


def generate_answer(question, retrieved):
    if not retrieved:
        return (
            "I couldn't find relevant text in the selected documents."
        ), []

    context_parts = []

    for i, item in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source S{i}: {item['source']}, chunk {item['chunk']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DocuMind AI, a helpful document "
                    "question-answering assistant. Answer using the "
                    "supplied document context. Do not invent facts "
                    "that are not supported by the context. If the "
                    "answer is not present, say you could not find it "
                    "in the selected documents. Keep answers clear "
                    "and concise. Refer to sources as S1, S2, and so "
                    "on when useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based on the context above."
                ),
            },
        ],
    )

    return response["message"]["content"], retrieved


def display_sources(sources):
    if not sources:
        return

    with st.expander("View sources"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**S{i} · {source['source']} · Chunk {source['chunk']}**"
            )
            st.caption(source["text"])


try:
    collection, embedding_model = get_resources()
except Exception as error:
    st.error(f"Could not initialize the chatbot: {error}")
    st.info(
        "Check your dependencies and internet connection for "
        "downloading the embedding model."
    )
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">DocuMind '
        '<span style="color:#829875">AI</span></div>'
        '<div class="sidebar-subtitle">YOUR DOCUMENT ASSISTANT</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Add documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV files",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Documents are indexed in your local ChromaDB database.",
    )

    if st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Choose at least one document first.")
        else:
            success_count = 0

            with st.spinner("Reading and indexing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        count = index_document(
                            uploaded_file, collection, embedding_model
                        )
                        st.success(
                            f"{uploaded_file.name}: {count} chunks indexed"
                        )
                        success_count += 1
                    except Exception as error:
                        st.error(f"{uploaded_file.name}: {error}")

            if success_count:
                st.rerun()

    st.divider()
    st.markdown("### Your documents")

    available_docs = get_document_names(collection)

    if available_docs:
        previous_selection = [
            name for name in st.session_state.selected_docs
            if name in available_docs
        ]

        selected_docs = st.multiselect(
            "Choose documents to search",
            options=available_docs,
            default=previous_selection or available_docs,
        )
        st.session_state.selected_docs = selected_docs

        if st.button(
            "Remove selected documents",
            use_container_width=True,
        ):
            if not selected_docs:
                st.warning("Select a document to remove.")
            else:
                for name in selected_docs:
                    existing = collection.get(
                        where={"source": name},
                        include=["metadatas"],
                    )
                    ids = existing.get("ids", [])

                    if ids:
                        collection.delete(ids=ids)

                st.session_state.selected_docs = []
                st.session_state.messages = []
                st.rerun()
    else:
        st.session_state.selected_docs = []
        st.caption("Your indexed documents will appear here.")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="info-note">Your vector database is stored locally in '
        '<code>chroma_db</code>. Ollama must be running for answers.</div>',
        unsafe_allow_html=True,
    )


all_chunks = collection.count()
document_count = len(get_document_names(collection))

st.markdown("""
<div class="hero">
  <div class="eyebrow">DOCUMENT INTELLIGENCE</div>
  <h1>Ask your documents anything.</h1>
  <p>Find answers from your files with AI-powered search and source
  references. Your documents stay in your local workspace.</p>
</div>
""", unsafe_allow_html=True)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Indexed documents</div>'
        f'<div class="metric-value">{document_count}</div></div>',
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Text chunks</div>'
        f'<div class="metric-value">{all_chunks}</div></div>',
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Language model</div>'
        '<div class="metric-value" style="font-size:20px">Llama 3.2 1B</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-title">Conversation</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("#### Welcome to DocuMind AI")
    st.write("Choose a suggested question or type your own below.")

    suggestions = [
        "Summarize the selected documents",
        "What are the main topics?",
        "Find important facts and details",
    ]

    suggestion_cols = st.columns(len(suggestions))

    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"])


question = st.chat_input("Ask a question about your documents...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    if not st.session_state.selected_docs:
        st.warning(
            "Upload and index a document, then select it in the sidebar."
        )
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching your documents and generating an answer..."
            ):
                try:
                    retrieved = retrieve(
                        question,
                        st.session_state.selected_docs,
                        collection,
                        embedding_model,
                    )

                    answer, sources = generate_answer(question, retrieved)
                    st.markdown(answer)
                    display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as error:
                    error_message = (
                        f"Something went wrong: {error}\n\n"
                        "Check that Ollama is running and the model "
                        f"`{MODEL_NAME}` is installed."
                    )
                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    })

st.markdown(
    '<div class="footer">DocuMind AI · Built with Streamlit, ChromaDB and Ollama</div>',
    unsafe_allow_html=True,
)
