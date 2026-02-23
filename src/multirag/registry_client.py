import os, requests

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8000")

def get_active_rags():
    r = requests.get(f"{REGISTRY_URL}/rags/active", timeout=10)
    r.raise_for_status()
    return r.json()