import streamlit as st
import requests
from urllib.parse import urljoin
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT = os.getenv("RUNPOD_ENDPOINT")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
    

def get_relevant_text(collection, query='', nresults=3, sim_th=None):
    """Get relevant text from a collection for a given query"""
    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th is not None:
        similarities = [1 - d for d in query_result.get("distances")[0]]
        relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
        return ''.join(relevant_docs)
    return ''.join([doc for doc in docs if doc is not None])


def generate_answer(prompt, context=[], top_k=5, top_p=0.9, temp=0.5):
    url = base_url + "/generate"
    data = {
        "prompt": prompt,
        "model": model,
        "stream": False,
        "context": context,
        "options": {"temperature": temp, "top_p": top_p, "top_k": top_k},
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        response_dict = response.json()
        return response_dict.get('response', ''), response_dict.get('context', [])
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return "", []


def get_contextual_prompt(question, context):
    """
    Optimized prompt format for Mistral 7B RAG applications
    Uses Mistral's preferred chat template format
    """
    # Option 1: Mistral Chat Template (Recommended)
    contextual_prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on the provided context. Use only the information given in the context to answer the question. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question} [/INST]"""
    
    return contextual_prompt
