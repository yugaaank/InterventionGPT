# InterventionGPT – Road Safety RAG System

InterventionGPT is a RAG-based Streamlit application using ChromaDB and Hugging Face Llama 3.1 to generate engineering-grade road safety interventions for the National Road Safety Hackathon 2025 (IITM).

***

### 🚦 Overview

InterventionGPT is a Retrieval-Augmented Generation (RAG) system designed to scan a curated knowledge base of road safety problems and generate implementation-ready engineering interventions following Indian Roads Congress (IRC) standards.
The system combines Streamlit for the UI, ChromaDB for vector storage and retrieval, SentenceTransformers for embeddings, and Llama 3.1 8B Instruct (Hugging Face) for technical intervention generation with a custom structured prompt enforcing engineering-grade output.

***

### 🔧 Key Features

- Loads a custom road-safety knowledge base from `data/rag_database.txt`.
- Embeds text using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores and retrieves context with persistent ChromaDB collections.
- Generates intervention reports using `meta-llama/Llama-3.1-8B-Instruct` via the Hugging Face Inference API.
- Enforces strict output structure including IRC references, dimensions, materials, and rationale.
- Designed specifically for the National Road Safety Hackathon 2025 – IIT Madras.

***

### 📁 Project Structure

```text
InterventionGPT/
│
├── app.py                    # Main Streamlit RAG application
├── chromadb_store/           # Persistent ChromaDB storage
├── data/
│   └── rag_database.txt      # Road safety knowledge base (one entry per line)
├── .env                      # HF_API_KEY stored here
├── requirements.txt
└── README.md
```


***

### ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yugaaank/InterventionGPT
   cd InterventionGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Hugging Face API key by creating a `.env` file:
   ```bash
   HF_API_KEY=your_key_here
   ```

4. Add your knowledge base by placing your data inside `data/rag_database.txt`, with one road safety issue per line.

***

### ▶️ Run the Application

```bash
streamlit run app.py
```


***

### 🧠 How It Works (RAG Pipeline)

1. **Embedding**  
   Each database entry is embedded using `sentence-transformers/all-MiniLM-L6-v2`.

2. **Storage**  
   Documents and embeddings are stored in a persistent ChromaDB collection named `local_knowledge_collection`.

3. **Retrieval**  
   Given a user query, the system returns the top‑N most similar chunks with similarity scores.

4. **LLM Generation**  
   A structured engineering prompt is sent to `meta-llama/Llama-3.1-8B-Instruct`, which produces interventions with dimensions, signage specifications, materials, installation procedures, IRC references (e.g., 67–2012, 35–2015, SP:84–2019), engineering rationale, and next steps in a strict standardized format.

***

### 🛣️ Sample Query

**Input**:  
“Pedestrian crossings near a school lack visibility and signage.”

**Output**:  
- Full structured engineering report tailored to the scenario.
- At least 3 interventions with dimensions, signage, and materials.
- Rationale and references to relevant IRC standards.
- Next actions and confidence level for each recommendation.

***

### 🧩 Configuration

All key configuration options are editable inside the code:

```python
FILE_PATH = "data/rag_database.txt"
COLLECTION_NAME = "local_knowledge_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LANGUAGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
```

***

### 🏗️ Future Improvements

- Map-based visualizations of interventions for better spatial understanding.
- Support for multi-modal inputs, such as images of roads and junctions.
- Auto-classification of problems using supervised ML models.
- Export of PDF engineering reports for submission and documentation.

***

### 📜 License

This project is licensed under the MIT License.
[8](https://github.com/RichardLitt/standard-readme)
[9](https://dev.to/cicirello/badges-tldr-for-your-repositorys-readme-3oo3)
