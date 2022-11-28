# LAI-Triton-Serve-Component

Triton Serve Component for lightning.ai

## Introduction

Triton serve component enables you to deploy your model to Triton Inference Server and setup a FastAPI interface
for converting api datatypes (string, int, float etc) to and from Triton datatypes (DT_STRING, DT_INT32 etc).

## Example

### Install the component

```bash
pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
```

### Save the app file

Save the following code as `app.py`

```python
import lightning as L
from lightning_triton import TritonServer
import base64, io, torchvision
from PIL import Image as PILImage
from pydantic import BaseModel


class Image(BaseModel):
    image: str


class Number(BaseModel):
    prediction: int


class TorchVisionServer(TritonServer):
    def __init__(self, input_type=Image, output_type=Number):
        super().__init__(input_type=input_type, output_type=output_type)
        self._model = None

    def setup(self):
        self._model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self._model.to(self.device)

    def infer(self, request):
        image = base64.b64decode(request.image.encode("utf-8"))
        image = PILImage.open(io.BytesIO(image))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transforms(image)
        image = image.to(self.device)
        prediction = self._model(image.unsqueeze(0))
        return {"prediction": prediction.argmax().item()}


app = L.LightningApp(TorchVisionServer())
```

### Run the app

Run the app locally using the following command

```bash
lightning run app app.py
```

### Run the app in cloud

Run the app in cloud using the following command

```bash
lightning run app app.py --cloud
```
