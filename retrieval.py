import urllib.request
import feedparser

import fitz
import os
import numpy as np

import re

import ollama
import faiss

from rank_bm25 import BM25Okapi

#EMBEDDING

def embed_chunks(chunks):
    response = ollama.embed(
        model = 'nomic-embed-text',
        input = chunks
    )

    return response['embeddings']

#BM25
def build_bm25_index(all_chunks):
    tokenised_chunks = []
    for chunk in all_chunks:
        tokens = chunk['text'].lower().split()
        tokenised_chunks.append(tokens)
    
    bm25 = BM25Okapi(tokenised_chunks)

    return bm25

def reciprocal_rank_fusion(faiss_ranked, bm25_ranked, k=60):
    rrf_scores={}
    for rank, chunk_idx in enumerate(faiss_ranked):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1/(k+rank+1)
    for rank, chunk_idx in enumerate(bm25_ranked):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1/(k+rank+1)
    
    return rrf_scores

