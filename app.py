import streamlit as st
import chromadb
import os
import time
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv  # <-- added

# =========================
# CONFIGURATION
# =========================
FILE_PATH = 'data/rag_database.txt'
COLLECTION_NAME = "local_knowledge_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LANGUAGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 512

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()  # reads from .env in same directory
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    st.error("❌ Missing Hugging Face API key. Add `HF_API_KEY=your_key_here` in a `.env` file.")
    st.stop()

# Hugging Face inference client
client = InferenceClient(model=LANGUAGE_MODEL, token=HF_API_KEY)

# =========================
# STREAMLIT STATE INIT
# =========================
if "collection" not in st.session_state:
    st.session_state.collection = None
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# =========================
# EMBEDDING FUNCTION
# =========================
class HuggingFaceEmbeddingFunction(chromadb.api.types.EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input_texts: chromadb.api.types.Documents) -> chromadb.api.types.Embeddings:
        return self.model.encode(input_texts, convert_to_numpy=True).tolist()


# =========================
# LOAD DATASET
# =========================
def load_data(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found. Create it and add your knowledge base text.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = [line.strip() for line in file.readlines() if line.strip()]
            if not dataset:
                st.error(f"'{file_path}' is empty. Add text content.")
                return []
            return dataset
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


# =========================
# INITIALIZE CHROMADB
# =========================
def initialize_knowledge_base(dataset):
    with st.spinner("Loading ChromaDB collection..."):
        chroma_client = chromadb.PersistentClient(path="./chromadb_store")
        hf_embed_func_instance = HuggingFaceEmbeddingFunction()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=hf_embed_func_instance
        )

        if collection.count() == 0:
            st.info("Embedding and adding documents to the knowledge base...")
            progress_bar = st.progress(0)
            for i in range(0, len(dataset), BATCH_SIZE):
                batch_docs = dataset[i:i + BATCH_SIZE]
                batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_docs))]
                collection.add(documents=batch_docs, ids=batch_ids)
                progress_bar.progress((i + len(batch_docs)) / len(dataset))
                time.sleep(0.05)
            progress_bar.empty()
        else:
            st.info(f"Loaded existing collection with {collection.count()} documents from disk.")

        st.success("✅ Knowledge base initialized successfully.")
        return collection


# =========================
# RETRIEVE CONTEXT
# =========================
def retrieve(collection, query: str, top_n: int = 3):
    # Always reattach embedding function when querying
    chroma_client = chromadb.PersistentClient(path="./chromadb_store")
    hf_embed_func_instance = HuggingFaceEmbeddingFunction()
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=hf_embed_func_instance
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_n,
        include=['documents', 'distances']
    )

    retrieved_knowledge = []
    if results and 'documents' in results and results['documents']:
        for chunk, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = 1.0 - distance
            retrieved_knowledge.append((chunk, similarity))
    return retrieved_knowledge


# =========================
# GENERATE RESPONSE
# =========================
def generate_response(input_query: str, joined_chunks: str):
    instruction_prompt = f"""SYSTEM:
You are 'InterventionGPT', a certified road safety design engineer from the National Road Safety Hackathon 2025.
Your role is to generate *engineering-grade road safety interventions* with full technical details based on the issue described.

TASK:
Recommend specific, implementable interventions as per Indian Roads Congress (IRC) standards and good engineering practice.

RULES:
1. Respond in **plain text**, but use a *structured report format*.
2. Do NOT describe the problem or talk about danger — go straight to technical actions.
3. Each answer must be practical, measurable, and standards-based.
4. Include exact sign types, dimensions, materials, installation, and placement details.
5. Reference relevant IRC standards (e.g., IRC:67–2012, IRC:35–2015, IRC:SP:84–2019).
6. Avoid JSON, bullet chaos, or redundant repetition.

OUTPUT FORMAT (STRICT):

INTERVENTION REPORT
Issue: <repeat the safety issue briefly>

Recommended Interventions:
1. <Detailed first intervention with IRC reference, dimensions, and materials>
2. <Detailed second intervention>
3. <Any supplementary measure>

Engineering Rationale:
<Explain how these measures address the issue, using technical reasoning>

Relevant Standards:
<IRC references and clauses used>

Confidence Level: <Low / Medium / High>

Next Engineering Actions:
- <Procurement or installation step 1>
- <Inspection or verification step 2>
- <Monitoring step 3>

CONTEXT (retrieved documents):
{joined_chunks}
"""

    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": input_query},
    ]

    response_text = ""

    try:
        # Safer: non-streaming call first
        response = client.chat_completion(messages=messages, max_tokens=1024, temperature=0.3)
        if hasattr(response, "choices") and response.choices:
            response_text = response.choices[0].message["content"]
        else:
            response_text = str(response)
        yield response_text

    except Exception as e:
        yield f"[Error generating response: {e}]"

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="InterventionGPT - Road Safety", page_icon="🛣️", layout="wide")
st.title("🧠 InterventionGPT - Road Safety Intervention Assistant")

st.sidebar.header("Configuration")
st.sidebar.write(f"**LLM Model:** {LANGUAGE_MODEL}")
st.sidebar.write(f"**Embedding Model:** {EMBEDDING_MODEL}")

if st.button("Load Knowledge Base"):
    dataset = load_data(FILE_PATH)
    if dataset:
        st.session_state.collection = initialize_knowledge_base(dataset)
        st.session_state.dataset_loaded = True
        st.success(f"Loaded {len(dataset)} records into ChromaDB!")


if not st.session_state.dataset_loaded:
    st.warning("⚠️ Please load your dataset first.")
else:
    query = st.text_area("Describe a road safety issue:", placeholder="e.g., Pedestrian crossings near a school lack visibility and signage.")
    top_n = st.slider("Number of retrieved context chunks:", 1, 5, 3)
    st.write("Docs in collection:", st.session_state.collection.count())
    if st.button("Generate Intervention"):
        with st.spinner("Retrieving relevant context..."):
            retrieved = retrieve(st.session_state.collection, query, top_n=top_n)

        if not retrieved:
            st.error("No relevant context found.")
        else:
            st.subheader("Retrieved Knowledge Chunks:")
            for chunk, sim in retrieved:
                st.markdown(f"- **(similarity {sim:.2f})** {chunk}")

            joined_chunks = "\n".join([chunk for chunk, _ in retrieved])

            st.subheader("📄 InterventionGPT Output")
            response_container = st.empty()
            full_response = ""
            with st.spinner("Generating response using Hugging Face..."):
                for piece in generate_response(query, joined_chunks):
                    full_response += piece
                    response_container.markdown(f"```\n{full_response}\n```")

st.markdown("---")
st.caption("Developed for NRSH 2025 Hackathon | Powered by Hugging Face + ChromaDB + Streamlit")
