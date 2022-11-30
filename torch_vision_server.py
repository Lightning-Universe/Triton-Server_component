# !pip install torchvision pillow
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
import lightning as L
import base64, io, torchvision, lightning_triton
from PIL import Image as PILImage


class TorchvisionServer(lightning_triton.TritonServer):
    def __init__(self, input_type=lightning_triton.Image, output_type=lightning_triton.Number, **kwargs):
        super().__init__(input_type=input_type,
                         output_type=output_type,
                         max_batch_size=8,
                         **kwargs)
        self._model = None

    def setup(self):
        self._model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self._model.to(self.device)

    def predict(self, request):
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


cloud_compute = L.CloudCompute("gpu", shm_size=512)
app = L.LightningApp(TorchvisionServer(cloud_compute=cloud_compute))
