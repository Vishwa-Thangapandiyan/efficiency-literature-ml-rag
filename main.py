import urllib.request
import feedparser
import fitz
import os
import numpy as np
import re
import ollama
import faiss
from rank_bm25 import BM25Okapi

from ingestion import search_arxiv, download_pdf, extract_text, clean_text, chunk_paper, PAPERS_DIR
from retrieval import embed_chunks, build_bm25_index, reciprocal_rank_fusion
from generation import generate_answer

BASE_URL = "http://export.arxiv.org/api/query?"

#-------------------------------------
#-------------------------------------


papers=[]
topics = {
    'quantization': 'cat:cs.LG AND all:"quantization" AND all:"neural network"',
    'pruning': 'cat:cs.LG AND all:"pruning" AND all:"neural network"',
    'distillation': 'cat:cs.LG AND all:"knowledge distillation"',
}
for i in topics:
    papers += search_arxiv(topics[i], max_results=10)

paper_ids=[]

for paper in papers:
    paper_ids.append(paper['pdf_url'].split('/')[-1])


for i in range(len(papers)):
    download_pdf(papers[i]['pdf_url'], paper_ids[i])
    papers[i]['text'] = extract_text(os.path.join(PAPERS_DIR, paper_ids[i]))

for i in range(len(papers)):
    papers[i]['text'] = clean_text(papers[i]['text'])

for i in range(len(papers)):
    papers[i]['chunks'] = chunk_paper(papers[i]['text'], 1000, 200)

all_chunks = []

for paper in papers:
    for chunk in paper['chunks']:
        all_chunks.append({
            'text':chunk,
            'paper_title':paper['title'],
            'pdf_url':paper['pdf_url']
        })

chunk_text = [c['text'] for c in all_chunks]

embeddings = embed_chunks(chunk_text)
#-------------------------------------

#first, converting list to numpy
embeddings_np = np.array(embeddings, dtype='float32')
print("Embedding shape: ", embeddings_np.shape)

#FAISS

d = embeddings_np.shape[1] #dimension of first chunk vector
index = faiss.IndexFlatL2(d)

index.add(embeddings_np)

bm25 = build_bm25_index(all_chunks)

#-----------------------------------

query = "how does quantization neural network work"
query_embedding = ollama.embed(
    model = 'nomic-embed-text',
    input = [query]
)['embeddings']
query_vec = np.array(query_embedding, dtype='float32')
distances, faiss_ranked = index.search(query_vec, k=10)
faiss_ranked = faiss_ranked[0]

tokenised_query = query.lower().split()
scores = bm25.get_scores(tokenised_query)
bm25_ranked = np.argsort(scores)[::-1][:10]

ranks = reciprocal_rank_fusion(faiss_ranked, bm25_ranked)

sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

top_chunks=[]
for chunk_idx, score in sorted_ranks[:5]:
    chunk_idx = int(chunk_idx)
    top_chunks.append(all_chunks[chunk_idx])

#-----------------------------------
#-----------------------------------

answer = generate_answer(query, top_chunks)
print(answer)