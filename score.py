import json
import ollama

import numpy as np
from retrieval import reciprocal_rank_fusion
from pipeline import load_or_build_pipeline
from retrieval import baseline_retrieve
from generation import generate_answer, judge_answer  
import matplotlib.pyplot as plt


index, bm25, all_chunks = load_or_build_pipeline()

with open('gold_questions.json') as f:
    gold_questions = json.load(f)

def judge_sufficiency(query, chunks):
    context = "\n\n".join(c['text'] for c in chunks)

    prompt = f"""Question: {query}

    Context: {context}

    Does the retrieved context contain enough information to fully answer the question?
    if yes, respond with exactly : SUFFICIENT
    if no, respond with exactly : SEARCH <a better search query to find the missing information>
    """

    response = ollama.generate(model="llama3.1:8b", prompt=prompt, options={'temperature': 0})
    return response['response'].strip()

def retrieval_loop(query, index, bm25, all_chunks, k=10):
    query_embedding = ollama.embed(
        model='nomic-embed-text',
        input=[query]
    )['embeddings']
    query_vec = np.array(query_embedding, dtype='float32')
    distances, faiss_ranked = index.search(query_vec, k)
    faiss_ranked = faiss_ranked[0]

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    bm25_ranked = np.argsort(scores)[::-1][:k]

    ranks = reciprocal_rank_fusion(faiss_ranked, bm25_ranked)

    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    top_chunks = []
    for chunk_idx, score in sorted_ranks[:5]:
        chunk_idx = int(chunk_idx)
        top_chunks.append(all_chunks[chunk_idx])

    return top_chunks

def agentic_rag(query, index, bm25, all_chunks, max_iterations=3):
    top_chunks_retrieved = []
    curr_query = query

    for i in range(max_iterations):
        chunks = retrieval_loop(curr_query, index, bm25, all_chunks)
        top_chunks_retrieved.extend(chunks)

        decision = judge_sufficiency(query, top_chunks_retrieved)
        if "SUFFICIENT" in decision:
            break
        curr_query = decision.replace("SEARCH", "").strip()

    return generate_answer(query, top_chunks_retrieved)

results = []
for q in gold_questions:
    question = q['question']

    baseline_chunks = baseline_retrieve(question, index, all_chunks)
    baseline_answer = generate_answer(question, baseline_chunks)

    agentic_answer = agentic_rag(question, index, bm25, all_chunks)

    baseline_score = judge_answer(question, baseline_answer, q['reference_answer'])
    agentic_score = judge_answer(question, agentic_answer, q['reference_answer'])

    results.append({
        'question':question,
        'type':q['type'],
        'baseline_score':baseline_score,
        'agentic_score':agentic_score
    })

def avg_by_type(results, qtype, key):
    subset = [r for r in results if r['type'] == qtype]
    return sum(r[key] for r in subset) / len(subset)

b_single = avg_by_type(results, 'single-topic', 'baseline_score')
b_multi = avg_by_type(results, 'multi-hop', 'baseline_score')
b_overall = sum(r['baseline_score'] for r in results) / len(results)

a_single = avg_by_type(results, 'single-topic', 'agentic_score')
a_multi = avg_by_type(results, 'multi-hop', 'agentic_score')
a_overall = sum(r['agentic_score'] for r in results) / len(results)

categories = ['Single-topic', 'Multi-hop', 'Overall']
baseline = [b_single, b_multi, b_overall]
agentic = [a_single, a_multi, a_overall]

x = range(len(categories))
plt.bar([i - 0.2 for i in x], baseline, width=0.4, label='Baseline (FAISS-only)')
plt.bar([i + 0.2 for i in x], agentic, width=0.4, label='Hybrid + Agentic')
plt.xticks(x, categories)
plt.ylabel('Avg Score (1-5)')
plt.title('Retrieval Method Comparison')
plt.legend()
plt.savefig('eval_comparison.png', dpi=150, bbox_inches='tight')

with open('eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)