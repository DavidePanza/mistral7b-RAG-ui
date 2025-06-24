import os
import time
import requests
from dotenv import load_dotenv
import json
import codecs
from pathlib import Path

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT = os.getenv("RUNPOD_ENDPOINT")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def format_messages_as_prompt(messages):
    """Convert messages list to a single prompt string for the model."""
    parts = []
    for message in messages:
        parts.append(f"{message['role'].capitalize()}: {message['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def run_prompt(prompt: str) -> str:
    """Submit a prompt to the RunPod endpoint and get back a response string."""
    payload = {"input": 
               {"prompt": prompt}
               }

    # Start job
    response = requests.post(f"{ENDPOINT}/run", headers=HEADERS, json=payload)
    job_id = response.json().get("id")
    print(f"[RunPod] Job started: {job_id}")

    # Poll for status
    while True:
        status_res = requests.get(f"{ENDPOINT}/status/{job_id}", headers=HEADERS).json()
        status = status_res.get("status")
        print(f"[RunPod] Status: {status}")
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)

    if status == "COMPLETED":
        return status_res["output"]["response"]
    else:
        raise RuntimeError("RunPod job failed.")


def clean_and_parse_json(raw_text: str):
    """Clean and parse model output into JSON."""
    cleaned = raw_text.strip().strip("```json").strip("```").strip("'")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            # Handle escaped quotes
            unescaped = codecs.decode(cleaned, 'unicode_escape')
            return json.loads(unescaped)
        except Exception as e:
            raise ValueError("Could not parse JSON output") from e
        
