# PDF Question-Answering with RAG

Ask questions about your own PDFs and get answers grounded in them, with file and page citations. If the answer isn't in the documents, the model is instructed to say "I don't know".

## How it works

1. **Load:** pypdf extracts text page by page.
2. **Chunk:** fixed-size chunks of 800 characters with 150 characters of overlap.
3. **Embed and store:** ChromaDB embeds each chunk (all-MiniLM-L6-v2) and stores it with its file name and page number.
4. **Retrieve:** the question is embedded with the same model, and the top 4 chunks are returned by semantic similarity.
5. **Generate:** Gemini answers only from the retrieved chunks and cites them as `[file p#]`.

Indexing runs once and is batched (200 chunks per call). API calls retry with exponential backoff (5s, 10s, 20s) when the model is overloaded, and a failed call doesn't crash the app.

## Stack

Python · ChromaDB · Gemini API (google-genai) · pypdf

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key. Put your PDFs in the project folder, list them in `FILES` at the top of `rag.py`, then run:

```bash
python rag.py
```

The first run indexes the PDFs into a local `rag_db/` folder. Later runs go straight to the question prompt. After changing the chunk size, delete `rag_db/` to re-index.

## Example

Indexed 3 PDFs into 305 chunks. For "What is the max flow min cut theorem?", the top chunks came from the right section of the source paper:

```
Ask (or 'quit'): What is the max flow min cut theorem?

--- Retrieved pieces (lower distance = closer) ---
[Hochbaum-OR.pdf p7] distance=0.695  Hochbaum: The Pseudoflow algorithm ... 3. The Maximum Flow and the ...
[Hochbaum-OR.pdf p7] distance=0.768  ...
[Hochbaum-OR.pdf p7] distance=0.769  ...
[Hochbaum-OR.pdf p1] distance=0.807  ...
```

## Next steps

- Hybrid search (BM25 + vectors) for exact terms like part numbers
- Cross-encoder reranking of a wider candidate set
- Recursive or section-aware chunking
- An evaluation set to measure recall@k and answer faithfulness
