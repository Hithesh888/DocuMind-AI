
import streamlit as st
import pandas as pd
import ollama
import chromadb

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

MODEL_NAME = "llama3.2:1b"
EMBEDDING_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K_PER_DOCUMENT = 4
MAX_CONTEXT_CHUNKS = 8


# ==========================================
# PROFESSIONAL CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #7667F5;
    --primary-light: #A99EFF;
    --background: #0B1020;
    --surface: #121A2B;
    --border: #29344A;
    --muted: #9BA9BF;
    --text: #F1F5FB;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 70% 0%, #18203B 0%, transparent 42%),
        var(--background);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0E1525;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.3rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
}

.brand-icon {
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #8B7CFF, #5D4AE8);
    border-radius: 14px;
    font-size: 24px;
    box-shadow: 0 8px 24px #7667F533;
}

.brand-title {
    font-size: 22px;
    font-weight: 800;
    color: #F5F6FF;
    letter-spacing: -0.7px;
}

.brand-subtitle {
    font-size: 11px;
    color: var(--muted);
    margin-top: 3px;
}

.hero {
    background: linear-gradient(125deg, #171E37, #14192B 65%, #1B1938);
    border: 1px solid #353455;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.hero-tag {
    display: inline-block;
    color: #C0B8FF;
    background: #7667F51C;
    border: 1px solid #7667F555;
    border-radius: 20px;
    padding: 6px 11px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 15px;
}

.hero h1 {
    font-size: clamp(27px, 4vw, 39px);
    line-height: 1.2;
    letter-spacing: -1.3px;
    color: #FFFFFF;
    margin: 0 0 13px 0;
}

.hero p {
    color: #AAB6CD;
    font-size: 14px;
    line-height: 1.8;
    max-width: 650px;
    margin-bottom: 0;
}

.section-label {
    color: #A9B6CD;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 10px;
    font-weight: 700;
}

.metric-card {
    background: linear-gradient(145deg, #161F32, #111827);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px;
    min-height: 112px;
}

.metric-icon {
    font-size: 19px;
    margin-bottom: 12px;
}

.metric-number {
    font-size: 25px;
    font-weight: 800;
    color: #F5F6FF;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
}

div[data-testid="stChatMessage"] {
    background: #131B2C;
    border: 1px solid #263249;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
}

div[data-testid="stChatInput"] {
    border: 1px solid #3B4165;
    border-radius: 17px;
    background: #151C2D;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #8275FF;
    box-shadow: 0 0 0 1px #8275FF44;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #343D55;
    background: #182135;
    color: #E9EDFA;
    font-weight: 600;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #8678FF;
    color: #FFFFFF;
    background: #252445;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #7667F5, #6454E5);
    border: none;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(110deg, #8B7CFF, #7667F5);
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #141D2E;
    border-color: #303B53;
}

div[data-testid="stExpander"] {
    background: #111A2A;
    border: 1px solid #29344A;
    border-radius: 12px;
}

div[data-testid="stFileUploader"] {
    background: #131B2B;
    border: 1px dashed #45466B;
    border-radius: 14px;
    padding: 10px;
}

[data-testid="stCaptionContainer"] {
    color: #94A3B8;
}

hr {
    border-color: #29344A;
}

.footer {
    text-align: center;
    color: #718099;
    font-size: 11px;
    padding: 28px 0 8px 0;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ADEA8;
    margin-right: 6px;
}

@media (max-width: 700px) {
    .hero {
        padding: 22px;
    }
    .block-container {
        padding-top: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODELS AND DATABASE
# ==========================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_NAME)


@st.cache_resource
def load_database():
    db_client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )
    db_collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return db_client, db_collection


model = load_embedding_model()
client, collection = load_database()


# ==========================================
# DOCUMENT READING
# ==========================================

def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()
    parts = []

    if extension == ".pdf":
        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            text = page.extract_text() or ""

            if text.strip():
                parts.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_number
                    }
                })

    elif extension == ".txt":
        text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

        if text.strip():
            parts.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": 1
                }
            })

    elif extension == ".csv":
        df = pd.read_csv(uploaded_file).fillna("")

        for row_number, row in enumerate(
            df.to_dict(orient="records"),
            start=1
        ):
            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            if row_text.strip():
                parts.append({
                    "text": row_text,
                    "metadata": {
                        "source": filename,
                        "row": row_number
                    }
                })

    return parts


# ==========================================
# TEXT CHUNKING
# ==========================================

def make_chunks(parts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for part in parts:
        for text in splitter.split_text(part["text"]):
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": part["metadata"].copy()
                })

    return chunks


# ==========================================
# INDEX DOCUMENTS
# ==========================================

def process_documents(files):
    for uploaded_file in files:
        filename = uploaded_file.name

        try:
            parts = read_uploaded_file(uploaded_file)

            if not parts:
                st.warning(
                    f"No readable text found in {filename}."
                )
                continue

            chunks = make_chunks(parts)

            if not chunks:
                st.warning(
                    f"No text chunks created for {filename}."
                )
                continue

            texts = [chunk["text"] for chunk in chunks]
            metadatas = [
                chunk["metadata"] for chunk in chunks
            ]

            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            # Replace previous chunks from the same file.
            collection.delete(
                where={"source": filename}
            )

            ids = [
                f"{filename}_{index}"
                for index in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            st.success(
                f"✓ {filename} — {len(chunks)} chunks indexed"
            )

        except Exception as error:
            st.error(
                f"Could not process {filename}: {error}"
            )


# ==========================================
# DOCUMENT LIST
# ==========================================

def get_available_files():
    if collection.count() == 0:
        return []

    result = collection.get(
        include=["metadatas"]
    )

    return sorted({
        metadata["source"]
        for metadata in (result.get("metadatas") or [])
        if metadata and metadata.get("source")
    })


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve_context(question, selected_files):
    if not selected_files:
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    retrieved = []

    for filename in selected_files:
        file_result = collection.get(
            where={"source": filename},
            include=["metadatas"]
        )

        file_count = len(file_result.get("ids", []))

        if file_count == 0:
            continue

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=min(TOP_K_PER_DOCUMENT, file_count),
            where={"source": filename},
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        texts = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for text, metadata, distance in zip(
            texts, metadatas, distances
        ):
            retrieved.append({
                "text": text,
                "metadata": metadata or {},
                "distance": distance
            })

    retrieved.sort(
        key=lambda item: item["distance"]
    )

    return retrieved[:MAX_CONTEXT_CHUNKS]


# ==========================================
# SOURCE FORMATTING
# ==========================================

def source_location(metadata):
    filename = metadata.get("source", "Unknown file")

    if "page" in metadata:
        return f"{filename} · Page {metadata['page']}"

    if "row" in metadata:
        return f"{filename} · Row {metadata['row']}"

    return filename


def build_context(retrieved):
    context_parts = []
    sources = []

    for index, item in enumerate(retrieved, start=1):
        label = f"S{index}"
        location = source_location(item["metadata"])

        context_parts.append(
            f"[{label}] Source: {location}\n"
            f"{item['text']}"
        )

        sources.append({
            "label": label,
            "location": location,
            "text": item["text"],
            "distance": item["distance"]
        })

    return "\n\n".join(context_parts), sources


# ==========================================
# OLLAMA ANSWER GENERATION
# ==========================================

def generate_answer(question, context):
    prompt = f"""
You are a helpful AI document assistant.

Instructions:
1. Answer the user's exact question.
2. Use only information in the supplied excerpts.
3. Cite relevant excerpts using labels such as [S1] and [S2].
4. Never invent facts, quotations, or citations.
5. If the answer is not in the excerpts, say:
   "I could not find this information in the selected documents."
6. Keep the response clear, organized, and easy to understand.
7. Use bullets when they improve readability.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"].strip()


# ==========================================
# CLEAR DOCUMENT DATABASE
# ==========================================

def clear_database():
    global collection

    client.delete_collection(
        name=COLLECTION_NAME
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    st.session_state.messages = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">✳</div>
        <div>
            <div class="brand-title">DocuMind AI</div>
            <div class="brand-subtitle">YOUR LOCAL KNOWLEDGE HUB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">DOCUMENT WORKSPACE</div>',
        unsafe_allow_html=True
    )

    st.write("")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or CSV files."
    )

    if st.button(
        "＋  Index documents",
        type="primary",
        use_container_width=True
    ):
        if uploaded_files:
            with st.spinner("Reading and indexing documents..."):
                process_documents(uploaded_files)

            st.rerun()
        else:
            st.warning("Choose at least one file first.")

    st.divider()

    available_files = get_available_files()

    st.markdown(
        '<div class="section-label">SEARCH SCOPE</div>',
        unsafe_allow_html=True
    )

    if available_files:
        selected_files = st.multiselect(
            "Search within",
            options=available_files,
            default=available_files,
            label_visibility="collapsed",
            help="Only selected documents will be searched."
        )
    else:
        selected_files = []
        st.caption("Your indexed documents will appear here.")

    st.divider()

    st.markdown(
        '<div class="section-label">WORKSPACE OVERVIEW</div>',
        unsafe_allow_html=True
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Documents",
            len(available_files)
        )

    with metric_col2:
        st.metric(
            "Text chunks",
            collection.count()
        )

    with st.expander("View indexed files"):
        if available_files:
            for filename in available_files:
                st.markdown(f"📄 `{filename}`")
        else:
            st.caption("No files indexed yet.")

    st.divider()

    if st.button(
        "＋  New conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🗑  Clear document database",
        use_container_width=True
    ):
        try:
            clear_database()
            st.rerun()
        except Exception as error:
            st.error(f"Could not clear database: {error}")

    st.markdown("""
    <div style="padding-top:18px; color:#8190A9; font-size:11px;">
        <span class="status-dot"></span> Local workspace
        <br><br>
        Powered by Ollama · ChromaDB
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ YOUR PERSONAL AI RESEARCH ASSISTANT</div>
    <h1>Your documents,<br>understood.</h1>
    <p>
        Turn your files into answers. Ask questions, explore ideas,
        and discover useful information from your own documents.
        Your responses are generated using your local AI model.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# METRIC CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📁</div>
        <div class="metric-number">{len(available_files)}</div>
        <div class="metric-label">Indexed documents</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🧩</div>
        <div class="metric-number">{collection.count()}</div>
        <div class="metric-label">Searchable text chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-number">Local</div>
        <div class="metric-label">Answer generation</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ==========================================
# CHAT HEADER
# ==========================================

st.markdown(
    '<div class="section-label">DOCUMENT CHAT</div>',
    unsafe_allow_html=True
)

st.subheader("What would you like to know?")

st.caption(
    "Ask a question about your indexed documents. "
    "Answers can include references to the retrieved sources."
)


# ==========================================
# INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# STARTER QUESTIONS
# ==========================================

if not st.session_state.messages:
    st.markdown("**Try asking**")

    prompt_col1, prompt_col2 = st.columns(2)

    with prompt_col1:
        if st.button(
            "✦  Summarize my document",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Summarize the main points of the selected documents."
            )
            st.rerun()

    with prompt_col2:
        if st.button(
            "⌕  Find key information",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What are the most important facts in the selected documents?"
            )
            st.rerun()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources and retrieved excerpts"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['label']}] {source['location']}**"
                    )
                    st.write(source["text"])


# ==========================================
# PROCESS QUESTION
# ==========================================

typed_question = st.chat_input(
    "Message DocuMind AI...",
    key="chat_input"
)

question = typed_question

if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    if not available_files:
        st.warning(
            "Upload and index a document in the sidebar before chatting."
        )

    elif not selected_files:
        st.warning(
            "Select at least one document in the sidebar."
        )

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant passages..."):
                    retrieved = retrieve_context(
                        question,
                        selected_files
                    )

                if not retrieved:
                    answer = (
                        "I could not find relevant information "
                        "in the selected documents."
                    )
                    sources = []

                else:
                    context, sources = build_context(retrieved)

                    with st.spinner("Preparing your answer..."):
                        answer = generate_answer(
                            question,
                            context
                        )

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources and retrieved excerpts"):
                        for source in sources:
                            st.markdown(
                                f"**[{source['label']}] "
                                f"{source['location']}**"
                            )
                            st.write(source["text"])
                            st.caption(
                                f"Retrieval distance: "
                                f"{source['distance']:.4f}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                answer = f"An error occurred: {error}"
                st.error(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": []
                })


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    DOCUMIND AI &nbsp;·&nbsp; LOCAL DOCUMENT INTELLIGENCE
    <br><br>
    Built with Streamlit, Sentence Transformers, ChromaDB and Ollama
</div>
""", unsafe_allow_html=True)
