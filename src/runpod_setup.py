import requests
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
    """
    Get relevant text from a collection for a given query
    """
    query_result = collection.query(query_texts=query, n_results=nresults)
    docs = query_result.get('documents')[0]
    if sim_th is not None:
        similarities = [1 - d for d in query_result.get("distances")[0]]
        relevant_docs = [d for d, s in zip(docs, similarities) if s >= sim_th]
        return ''.join(relevant_docs)
    return ''.join([doc for doc in docs if doc is not None])


def get_contextual_prompt(question, context):
    """
    Optimized prompt format for Mistral 7B 
    """
    # Option 1: Mistral Chat Template (Recommended)
    contextual_prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on the provided context. Use only the information given in the context to answer the question. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question} [/INST]"""
    
    return contextual_prompt


def generate_answer(prompt, max_tokens=150, temperature=0.7, HEADERS=HEADERS, ENDPOINT=ENDPOINT):
    """
    Submit a prompt to the RunPod SYNC endpoint and get back a response string.
    """
    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        # Use /runsync instead of /run - immediate response!
        response = requests.post(f"{ENDPOINT}/runsync", headers=HEADERS, json=payload, timeout=65)
        response.raise_for_status()
        result = response.json()
        
        print(f"[RunPod] Request completed successfully")
        
        if result.get("status") == "COMPLETED":
            return result["output"]["response"]
        else:
            error_msg = result.get("error", "Unknown error")
            raise RuntimeError(f"RunPod job failed: {error_msg}")
            
    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out (>60s). Try reducing prompt length or max_tokens.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RunPod API error: {e}")

