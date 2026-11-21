"""Compare retrieval modes on a labelled question set: recall@k and MRR@k."""
import json
import time

from retrieval import TOP_K, Retriever, build_index

questions = json.load(open("eval_set.json", encoding="utf-8"))
retriever = Retriever(build_index())
modes = ["vector", "hybrid"] + (["hybrid+rerank"] if retriever.reranker else [])

print(f"{len(questions)} questions, k = {TOP_K}\n")
print(f"{'mode':<15}{'recall@k':>10}{'MRR@k':>8}{'ms/query':>10}")
for mode in modes:
    hits, rr, start = 0, 0.0, time.perf_counter()
    misses = []
    for item in questions:
        ids = retriever.search(item["q"], mode=mode, k=TOP_K)
        rank = next((r for r, i in enumerate(ids, 1)
                     if retriever.meta[i]["source"] == item["source"]
                     and retriever.meta[i]["page"] in item["pages"]), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(item["q"])
    ms = (time.perf_counter() - start) * 1000 / len(questions)
    print(f"{mode:<15}{hits / len(questions):>10.2f}{rr / len(questions):>8.2f}{ms:>10.0f}")
    for q in misses:
        print(f"   missed: {q}")
