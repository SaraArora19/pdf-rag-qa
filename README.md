# PDF Question-Answering with RAG

Ask questions about your own PDFs and get answers grounded in them, with file and page citations. If the answer isn't in the documents, the model is instructed to say "I don't know".

## How it works

1. **Load:** pypdf extracts text page by page.
2. **Chunk:** fixed-size chunks of 800 characters with 150 characters of overlap.
3. **Embed and store:** ChromaDB embeds each chunk (all-MiniLM-L6-v2) and stores it with its file name and page number.
4. **Retrieve (hybrid):** semantic vector search and BM25 keyword search each return 20 candidates, merged with Reciprocal Rank Fusion.
5. **Rerank:** a cross-encoder (ms-marco-MiniLM-L-6-v2, via fastembed) rescores the fused candidates and keeps the top 4.
6. **Generate:** Gemini answers only from those chunks and cites them as `[file p#]`.

Indexing runs once and is batched (200 chunks per call). API calls retry with exponential backoff (5s, 10s, 20s) when the model is overloaded, and a failed call doesn't crash the app.

## Evaluation

`eval.py` runs 26 labelled questions over 3 PDFs (305 chunks). A question counts as a hit if a chunk from its answer page is in the top 4.

| Retrieval mode | Recall@4 | MRR@4 | Latency per query |
|---|---|---|---|
| Vector only | 0.92 | 0.88 | 241 ms |
| Hybrid (vector + BM25, RRF) | 0.92 | 0.88 | 224 ms |
| Hybrid + cross-encoder rerank | **0.96** | **0.91** | 892 ms |

Reranking found one more correct page and ranked answers higher, at about 4x the latency. Hybrid search alone gave no gain on this question set, which is mostly conceptual rather than exact-term lookups. With 26 questions, one question is about 4 points of recall, so these numbers are indicative, not definitive.

## Stack

Python · ChromaDB · Gemini API (google-genai) · fastembed (cross-encoder) · pypdf

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key. Put your PDFs in the project folder and list them in `FILES` at the top of `retrieval.py`, then:

```bash
python rag.py      # ask questions
python eval.py     # compare retrieval modes (edit eval_set.json for your own PDFs)
```

The first run indexes the PDFs into a local `rag_db/` folder and downloads the reranker model. After changing the chunk size, delete `rag_db/` to re-index.

## Files

- `retrieval.py`: loading, chunking, indexing, vector search, BM25, RRF and reranking
- `rag.py`: the question-answering loop
- `eval.py` and `eval_set.json`: the retrieval evaluation

## Next steps

- Recursive or section-aware chunking
- A larger evaluation set with more exact-term questions (codes, names, numbers)
- Answer-level evaluation (faithfulness) in addition to retrieval
