import streamlit as st
import requests
from urllib.parse import urljoin
import warnings


def is_ollama_running(base_url, logger):
    """
    Check if the Ollama server is running
    """
    try:
        url = urljoin(base_url, "/api/tags")
        logger.debug(f"Checking Ollama server at {url}")
        response = requests.get(url, timeout=5)
        # Add debug output
        logger.debug(f"Ollama response status: {response.status_code}")
        logger.debug(f"Ollama response content: {response.text[:100]}...")  # Show first 100 chars
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Exception when connecting to Ollama: {e}")
        return False
    

def get_relevant_text(collection, query='', nresults=2, sim_th=None):
    """Get relevant text from a collection for a given query"""

    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th is not None:
        similarities = [1 - d for d in query_result.get("distances")[0]]
        relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
        return ''.join(relevant_docs)
    return ''.join([doc for doc in docs if doc is not None])


def generate_answer(base_url, model, prompt, context=[], top_k=5, top_p=0.9, temp=0.5):
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
    contextual_prompt = (
        "You are a helpful assistant. Use the information provided in the context below to answer the question. "
        "Ensure your answer is accurate, concise, and directly addresses the question. "
        "If the context does not provide enough information to answer the question, state that explicitly.\n\n"
        "### Context:\n"
        f"{context}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:"
    )
    return contextual_prompt