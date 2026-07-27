import os
import requests
from dotenv import load_dotenv

# Load the token from .env
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
payload = {"inputs": "Hello world", "parameters": {"max_new_tokens": 10}}

response = requests.post(
    f"https://api-inference.huggingface.co/models/{MODEL_ID}",
    headers=headers,
    json=payload
)

print("Status code:", response.status_code)
print("Response:", response.text)
