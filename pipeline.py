import os
import pickle
import faiss
import numpy as np

from ingestion import search_arxiv, download_pdf, extract_text, clean_text, chunk_paper, PAPERS_DIR
from retrieval import embed_chunks, build_bm25_index, reciprocal_rank_fusion

CHECKPOINT_DIR = "checkpoint"

def load_or_build_pipeline():
    chunks_path = os.path.join(CHECKPOINT_DIR, "all_chunks.pkl")

    if os.path.exists(chunks_path):
        print("Checkpoint found — loading from disk, skipping ingestion.")
        all_chunks = pickle.load(open(chunks_path, "rb"))
        index = faiss.read_index(os.path.join(CHECKPOINT_DIR, "faiss_index.bin"))
        bm25 = pickle.load(open(os.path.join(CHECKPOINT_DIR, "bm25.pkl"), "rb"))
        return index, bm25, all_chunks

    print("No checkpoint — running full ingestion pipeline (this will take a while).")

    # ---- EVERYTHING BELOW IS YOUR EXISTING main.py CODE, UNCHANGED ----
    papers = []
    topics = {
        'quantization': 'cat:cs.LG AND all:"quantization" AND all:"neural network"',
        'pruning': 'cat:cs.LG AND all:"pruning" AND all:"neural network"',
        'distillation': 'cat:cs.LG AND all:"knowledge distillation"',
    }
    for i in topics:
        papers += search_arxiv(topics[i], max_results=100)

    paper_ids = []
    for paper in papers:
        paper_ids.append(paper['pdf_url'].split('/')[-1])

    for i in range(len(papers)):
        try:
            download_pdf(papers[i]['pdf_url'], paper_ids[i])
            papers[i]['text'] = extract_text(os.path.join(PAPERS_DIR, paper_ids[i]))
        except Exception as e:
            print(f"Skipping paper (failed): {papers[i]['title']} — {e}")
            papers[i]['text'] = None
    papers = [p for p in papers if p['text'] is not None]

    for i in range(len(papers)):
        papers[i]['text'] = clean_text(papers[i]['text'])

    for i in range(len(papers)):
        papers[i]['chunks'] = chunk_paper(papers[i]['text'], 1000, 200)

    all_chunks = []
    for paper in papers:
        for chunk in paper['chunks']:
            all_chunks.append({
                'text': chunk,
                'paper_title': paper['title'],
                'pdf_url': paper['pdf_url']
            })

    chunk_text = [c['text'] for c in all_chunks]
    embeddings = embed_chunks(chunk_text)
    embeddings_np = np.array(embeddings, dtype='float32')

    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)

    bm25 = build_bm25_index(all_chunks)
    # ---- END OF YOUR EXISTING CODE ----

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pickle.dump(all_chunks, open(chunks_path, "wb"))
    faiss.write_index(index, os.path.join(CHECKPOINT_DIR, "faiss_index.bin"))
    pickle.dump(bm25, open(os.path.join(CHECKPOINT_DIR, "bm25.pkl"), "wb"))

    return index, bm25, all_chunks