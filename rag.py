import time
import chromadb
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()                         # reads GEMINI_API_KEY from .env

FILES = ["Hochbaum-OR.pdf", "IoT.pdf", "Quality Assurance.pdf"]
CHUNK_SIZE, OVERLAP = 800, 150
MODEL = "gemini-3.8-flash"

# 1. LOAD
def load_pdf(path):
    reader = PdfReader(path)
    return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]

# 2. CHUNK
def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces

# 3 + 4. EMBED and STORE (Chroma embeds with its built-in model)
db = chromadb.PersistentClient(path="rag_db")
col = db.get_or_create_collection("docs")

if col.count() == 0:                  # index only the first time
    for f in FILES:
        ids, docs, metas = [], [], []
        for text, page in load_pdf(f):
            for j, piece in enumerate(chunk(text)):
                ids.append(f"{f}-p{page}-c{j}")
                docs.append(piece)
                metas.append({"source": f, "page": page})
        for i in range(0, len(ids), 200):      # store in batches of 200
            col.upsert(ids=ids[i:i+200], documents=docs[i:i+200], metadatas=metas[i:i+200])
        print(f"Indexed {f}: {len(ids)} chunks")
print("Stored chunks:", col.count())

client = genai.Client()               # uses GEMINI_API_KEY loaded from .env

while True:
    question = input("\nAsk (or 'quit'): ")
    if question == "quit":
        break

    # 5. RETRIEVE
    res = col.query(query_texts=[question], n_results=4)
    print("\n--- Retrieved pieces (lower distance = closer) ---")
    for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"[{m['source']} p{m['page']}] distance={dist:.3f}  {d[:120]}...")

    context = "\n\n".join(f"[{m['source']} p{m['page']}] {d}"
                          for d, m in zip(res["documents"][0], res["metadatas"][0]))

    # 6. GENERATE (retry with increasing waits if the model is busy)
    prompt = f"""Answer using only the context below. Cite sources like [file p#].
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""

    for attempt in range(3):
        try:
            reply = client.models.generate_content(model=MODEL, contents=prompt)
            print("\n--- Answer ---\n", reply.text)
            break
        except Exception as e:
            wait = 5 * 2 ** attempt          # waits 5s, then 10s, then 20s
            print(f"Model error ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        print("Couldn't get an answer right now. Try again in a minute.")
