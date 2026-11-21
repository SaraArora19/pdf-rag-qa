import time

from dotenv import load_dotenv
from google import genai

from retrieval import Retriever, build_index

load_dotenv()                         # reads GEMINI_API_KEY from .env
MODEL = "gemini-3.8-flash"
MODE = "hybrid+rerank"                # or "hybrid" / "vector"

col = build_index()
print("Stored chunks:", col.count())
retriever = Retriever(col)
client = genai.Client()

while True:
    question = input("\nAsk (or 'quit'): ")
    if question == "quit":
        break

    # 5. RETRIEVE: vector + BM25, fused with RRF, then reranked
    ids = retriever.search(question, mode=MODE)
    print(f"\n--- Retrieved pieces ({MODE}) ---")
    for i in ids:
        m = retriever.meta[i]
        print(f"[{m['source']} p{m['page']}]  {retriever.text[i][:120]}...")

    context = "\n\n".join(
        f"[{retriever.meta[i]['source']} p{retriever.meta[i]['page']}] {retriever.text[i]}" for i in ids)

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
