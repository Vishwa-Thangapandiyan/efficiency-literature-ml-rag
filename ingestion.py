import urllib.request
import feedparser

import fitz
import os
import numpy as np

import re

import ollama
import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


BASE_URL = "http://export.arxiv.org/api/query?"
PAPERS_DIR = "papers"

def search_arxiv(query, max_results=50):
    url_dict={"search_query":query, "start":0, "max_results":max_results}
    url = BASE_URL + urllib.parse.urlencode(url_dict)
    #urlencode and glue it back

    retrieved = urllib.request.urlopen(url)
    parsed_url = feedparser.parse(retrieved)

    results=[]
    for i in parsed_url.entries:
        pdf_url = ''
        for link in i.links:
            if link.get('title') == 'pdf':
                pdf_url = link['href']
                break
        results.append({
            'title': i.title,
            'summary': i.summary, 
            'pdf_url': pdf_url,
            'published': i.published
        })

    return results

def download_pdf(pdf_url, paper_id):
    os.makedirs(PAPERS_DIR, exist_ok=True)
    urllib.request.urlretrieve(pdf_url, os.path.join(PAPERS_DIR, paper_id))


def extract_text(pdf_path):
    content=""
    doc = fitz.open(pdf_path)
    for page in doc:
        content += page.get_text()
    
    return content

def clean_text(text):
    result_text = re.sub(r'\s+', ' ', text)
    result_text = re.sub(r'-\s', '', result_text)

    return result_text.strip()


def chunk_paper(text, c_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = c_size, chunk_overlap = overlap)
    chunk_text = text_splitter.split_text(text)

    return chunk_text


