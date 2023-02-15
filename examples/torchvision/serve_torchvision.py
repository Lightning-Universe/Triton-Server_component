# !pip install torchvision pillow
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
import base64
import io

import lightning as L
import torch
import torchvision
from PIL import Image as PILImage

import lightning_triton as lt


class TorchvisionServer(lt.TritonServer):
    def __init__(self, input_type=lt.Image, output_type=lt.Category, **kwargs):
        super().__init__(input_type=input_type, output_type=output_type, max_batch_size=8, **kwargs)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None

    def setup(self):
        self._model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self._model.to(torch.device(self._device))

    def predict(self, request):
        image = base64.b64decode(request.image.encode("utf-8"))
        image = PILImage.open(io.BytesIO(image))
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = transforms(image)
        image = image.to(self._device)
        prediction = self._model(image.unsqueeze(0))
        return {"category": prediction.argmax().item()}


cloud_compute = L.CloudCompute("gpu", shm_size=512)
app = L.LightningApp(TorchvisionServer(cloud_compute=cloud_compute))
