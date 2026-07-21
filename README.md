# InterventionGPT

RAG pipeline for road safety intervention analysis, built for the National Road Safety Hackathon 2025 (IIT Madras).

Takes a road safety problem description, retrieves relevant context from a curated knowledge base, and generates engineering-grade intervention reports with IRC standard references, dimensions, materials, and rationale.

Built with Streamlit, ChromaDB, SentenceTransformers, and Llama 3.1 8B.

## Install

```bash
git clone https://github.com/yugaaank/InterventionGPT
cd InterventionGPT
pip install -r requirements.txt
```

Add your Hugging Face API key:

```bash
echo "HF_API_KEY=your_key_here" > .env
```

## Usage

```bash
streamlit run app.py
```

Enter a road safety problem (e.g. "Pedestrian crossings near a school lack visibility and signage") and the system returns structured intervention reports with dimensions, signage specs, materials, and IRC references.

## How it works

1. Entries in `data/rag_database.txt` are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB
2. On query, the top-N most similar entries are retrieved
3. A structured prompt is sent to Llama 3.1 8B, which produces interventions with engineering specifications

## License

MIT
