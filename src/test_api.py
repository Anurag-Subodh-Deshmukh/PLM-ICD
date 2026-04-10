import urllib.request
import json

data = json.dumps({
    "text": "Patient admitted with chest pain and atrial fibrillation. History of diabetes type 2 and hypertension."
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=data,
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())

for c in result["codes"][:10]:
    print(f"  {c['code']:>10}  {c['probability']:.4f}  {c['description'][:70]}")
