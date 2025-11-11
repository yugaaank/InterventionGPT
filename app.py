import streamlit as st
import ollama
import chromadb
import os
from typing import List
import time

# --- Configuration ---
FILE_PATH = 'data/rag_database.txt'
COLLECTION_NAME = "local_knowledge_collection"
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest'
BATCH_SIZE = 512

# --- Initialize Session State ---
if "collection" not in st.session_state:
    st.session_state.collection = None
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# 🧠 Embedding Function Wrapper for Ollama
class OllamaEmbeddingFunction(chromadb.api.types.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, input_texts: chromadb.api.types.Documents) -> chromadb.api.types.Embeddings:
        result = ollama.embed(model=self.model_name, input=input_texts)
        return result['embeddings']


# 📂 Load Dataset
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


# ⚙️ Setup ChromaDB
def initialize_knowledge_base(dataset):
    with st.spinner("Loading ChromaDB collection..."):
        chroma_client = chromadb.PersistentClient(path="./chromadb_store")
        ollama_embed_func_instance = OllamaEmbeddingFunction(model_name=EMBEDDING_MODEL)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ollama_embed_func_instance
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



# 🔍 Retrieve Context
def retrieve(collection, query: str, top_n: int = 3):
    results = collection.query(query_texts=[query], n_results=top_n, include=['documents', 'distances'])
    retrieved_knowledge = []

    if results and 'documents' in results and results['documents']:
        for chunk, distance in zip(results['documents'][0], results['distances'][0]):
            similarity = 1.0 - distance
            retrieved_knowledge.append((chunk, similarity))
    return retrieved_knowledge


# 💬 Generate Response
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
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ]

    response_text = ""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3}
    )
    for chunk in stream:
        if chunk.get('message', {}).get('content'):
            response_text += chunk['message']['content']

    return response_text


# 🚀 Streamlit UI
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

            with st.spinner("Generating response using Qwen 4B..."):
                output = generate_response(query, joined_chunks)

            st.subheader("📄 InterventionGPT Output")
            st.code(output, language="json")

st.markdown("---")
st.caption("Developed for NRSH 2025 Hackathon | Powered by Ollama + ChromaDB + Streamlit")
