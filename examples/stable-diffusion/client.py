import base64
from pathlib import Path
import requests

response = requests.post('http://localhost:7777/predict', json={
    "text": "Harry potter fighting Voldemort on top of statue of liberty"
})

img = response.json()["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
