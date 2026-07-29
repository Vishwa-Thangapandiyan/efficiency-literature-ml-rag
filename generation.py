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

    prompt = f"""Answer the question thoroughly using only the context below.
            Structure your answer with:
            - A brief direct answer first
            - Supporting details organized under clear headings
            - Cite which paper each point comes from, by title

            Context:
            {context}

            Question: {query}

            Answer:"""

    response = ollama.generate(model='llama3.1:8b', prompt = prompt)
    return response['response']

def judge_answer(question, answer, reference_answer):
    prompt = f"""You are evaluating an AI-generated answer against a reference answer.

Question: {question}

Reference Answer (ground truth): {reference_answer}

Generated Answer: {answer}

Rate the Generated Answer on a scale of 1-5 for each criterion:
- Faithfulness: Does it avoid contradicting or inventing facts not in the reference?
- Completeness: Does it cover the key points from the reference answer?
- Relevance: Does it directly address the question?

Respond in EXACTLY this format, nothing else:
Faithfulness: <score>
Completeness: <score>
Relevance: <score>

ENSURE TO STRICTLY FOLLOW THE FORMAT
"""
    response = ollama.generate(model='llama3.1:8b', prompt=prompt)['response']

    try:
        faithfulness = int(re.search(r'Faithfulness:\s*(\d)', response).group(1))
        completeness = int(re.search(r'Completeness:\s*(\d)', response).group(1))
        relevance = int(re.search(r'Relevance:\s*(\d)', response).group(1))
        avg_score = (faithfulness + completeness + relevance) / 3
        return avg_score
    except AttributeError:
        print(f"Judge response didn't match expected format: {response}")
        return None
