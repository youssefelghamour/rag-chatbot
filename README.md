# Tableau RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about an Introduction to Tableau course using its PDF content as a knowledge source.

Instead of relying solely on an LLM's general knowledge, the application retrieves relevant course content and uses it to generate context-aware answers.

## Features

- **PDF ingestion:** Loads and extracts content from the Tableau course PDF.
- **Text processing:** Splits content by Markdown headers and token size.
- **Transcript correction:** Uses an LLM to correct transcription errors and caches the results.
- **Vector embeddings:** Generates local embeddings using FastEmbed.
- **Vector database:** Stores and retrieves document embeddings using ChromaDB.
- **Semantic retrieval:** Uses Maximal Marginal Relevance (MMR) to retrieve relevant and diverse chunks.
- **Question answering:** Uses retrieved context to generate answers through Groq.
- **Streaming:** Streams responses token by token.

## Tech Stack

- **Language:** Python
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Framework:** LangChain
- **Embeddings:** FastEmbed
- **Vector Database:** ChromaDB
- **Document Processing:** PyPDFLoader, MarkdownHeaderTextSplitter, TokenTextSplitter

## Project Structure

```text
.
├── app/
│   ├── data/
│   │   ├── Introduction_to_Tableau.pdf
│   │   └── formatted_lectures.json
│   ├── __init__.py
│   ├── main.py
│   ├── llm.py
│   ├── ingestion.py
│   ├── vectorstore.py
│   ├── prompts.py
│   └── rag_chain.py
├── chroma_db/
├── .env
├── .gitignore
└── requirements.txt
```

## How It Works

1. Loads the course PDF and extracts its text.
2. Splits the extracted content using Markdown headers.
3. Uses an LLM to correct transcription errors and caches the formatted content.
4. Splits the content into smaller token-based chunks.
5. Generates embeddings locally and stores them in ChromaDB.
6. Retrieves relevant chunks for each question using MMR.
7. Passes the retrieved context and question to the LLM.
8. Streams the generated answer.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv langchain_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

From the project root:

```bash
python -m app.main
```

## Notes

- The LLM requires an internet connection and a valid Groq API key.
- Embeddings are generated locally using FastEmbed.
- ChromaDB persists the vector database locally in `chroma_db/`.
- The initial setup downloads the embedding model.
- If the source PDF or processed content changes, rebuild the vector database and transcript cache.