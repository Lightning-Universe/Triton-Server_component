import base64
import requests
from pathlib import Path

# img = Path("catimage.png").read_bytes()
img = requests.get("https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png").content
img = base64.b64encode(img).decode("UTF-8")
response = requests.post("http://127.0.0.1:7777/predict", json={
    "image": img
})
print(response.json())
