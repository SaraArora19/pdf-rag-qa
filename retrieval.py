"""Indexing and retrieval: vector search, BM25 keyword search, RRF fusion and cross-encoder reranking."""
import math
import re
from collections import Counter

import chromadb
from pypdf import PdfReader

FILES = ["Hochbaum-OR.pdf", "IoT.pdf", "Quality Assurance.pdf"]
CHUNK_SIZE, OVERLAP = 800, 150
CANDIDATES = 20        # how many results each search returns before fusion/reranking
TOP_K = 4              # chunks sent to the LLM


# ---------- 1. LOAD + 2. CHUNK ----------
def load_pdf(path):
    reader = PdfReader(path)
    return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]


def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


# ---------- 3 + 4. EMBED and STORE ----------
def build_index(path="rag_db"):
    db = chromadb.PersistentClient(path=path)
    col = db.get_or_create_collection("docs")
    if col.count() == 0:                          # index only the first time
        for f in FILES:
            ids, docs, metas = [], [], []
            for text, page in load_pdf(f):
                for j, piece in enumerate(chunk(text)):
                    ids.append(f"{f}-p{page}-c{j}")
                    docs.append(piece)
                    metas.append({"source": f, "page": page})
            for i in range(0, len(ids), 200):     # store in batches of 200
                col.upsert(ids=ids[i:i+200], documents=docs[i:i+200], metadatas=metas[i:i+200])
            print(f"Indexed {f}: {len(ids)} chunks")
    return col


# ---------- BM25 keyword search (no extra package needed) ----------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, ids, docs, k1=1.5, b=0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k):
        terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len))
            if s > 0:
                scores.append((s, self.ids[i]))
        scores.sort(reverse=True)
        return [doc_id for _, doc_id in scores[:k]]


# ---------- Reciprocal Rank Fusion ----------
def rrf(result_lists, k=60):
    scores = Counter()
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] += 1 / (k + rank)
    return [doc_id for doc_id, _ in scores.most_common()]


class Retriever:
    """mode: 'vector', 'hybrid', or 'hybrid+rerank'."""

    def __init__(self, col):
        self.col = col
        data = col.get(include=["documents", "metadatas"])
        self.text = dict(zip(data["ids"], data["documents"]))
        self.meta = dict(zip(data["ids"], data["metadatas"]))
        self.bm25 = BM25(data["ids"], data["documents"])
        self.reranker = None
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        except Exception as e:                    # reranking is optional
            print(f"Reranker unavailable ({e.__class__.__name__}); install fastembed to enable it.")

    def vector(self, query, k):
        return self.col.query(query_texts=[query], n_results=k)["ids"][0]

    def search(self, query, mode="hybrid+rerank", k=TOP_K):
        if mode == "vector":
            return self.vector(query, k)
        fused = rrf([self.vector(query, CANDIDATES), self.bm25.search(query, CANDIDATES)])[:CANDIDATES]
        if mode == "hybrid" or self.reranker is None:
            return fused[:k]
        scores = list(self.reranker.rerank(query, [self.text[i] for i in fused]))
        ranked = sorted(zip(scores, fused), reverse=True)
        return [doc_id for _, doc_id in ranked[:k]]
