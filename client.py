import base64
from pathlib import Path
import requests

img = Path("catimage.png").read_bytes()
img = base64.b64encode(img).decode("UTF-8")
response = requests.post("http://127.0.0.1:7777/predict", json={
    "image": img
})
print(response.json())
