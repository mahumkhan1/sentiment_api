import requests

BASE = "http://localhost:8000"

r = requests.get(f"{BASE}/health")
print("Health:", r.json())

r = requests.post(f"{BASE}/predict", json={"text": "This is fantastic!"})
print("Single:", r.json())

r = requests.post(f"{BASE}/predict/batch", json={"texts": [
    "I absolutely love this.",
    "Worst thing I've ever used.",
    "It's fine, nothing special.",
]})
for item in r.json()["results"]:
    print(f"  [{item['sentiment']:8s} {item['score']:.2f}] {item['text']}")