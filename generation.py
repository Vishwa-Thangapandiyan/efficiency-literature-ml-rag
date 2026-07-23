import urllib.request
import feedparser

import fitz
import os
import numpy as np

import re

import ollama
import faiss

from rank_bm25 import BM25Okapi

def generate_answer(query, top_chunks):
    context = "\n\n".join([c['text'] for c in top_chunks])

    prompt = f"""Answer the question using only the context below.
    
    Context:
    {context}

    Question : {query}

    Answer:"""

    response = ollama.generate(model='llama3.1:8b', prompt = prompt)
    return response['response']
