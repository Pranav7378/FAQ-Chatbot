import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("API_SECRET_KEY")

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

bad_headers = {
    "X-API-Key": "wrong_key",
    "Content-Type": "application/json"
}

print("1. Testing without API key...")
r = requests.post(f"{BASE_URL}/chat", json={"question": "Test"}, headers={"Content-Type": "application/json"})
print("Status:", r.status_code, "->", r.json() if r.status_code != 403 else "403 Forbidden (Success)")

print("\n2. Testing with wrong API key...")
r = requests.post(f"{BASE_URL}/chat", json={"question": "Test"}, headers=bad_headers)
print("Status:", r.status_code, "->", r.json() if r.status_code != 403 else "403 Forbidden (Success)")

print("\n3. Testing valid request (ensure server is running)...")
try:
    r = requests.post(f"{BASE_URL}/chat", json={"question": "What are Pranav's skills?"}, headers=headers)
    print("Status:", r.status_code)
    print("Response:", r.json())
except requests.exceptions.ConnectionError:
    print("ConnectionError: Is the FastAPI server running?")
