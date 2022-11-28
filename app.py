import lightning as L
import base64, io, torchvision
from PIL import Image as PILImage
from pydantic import BaseModel


class Image(BaseModel):
    image: str


class Number(BaseModel):
    prediction: int


class TorchVisionServer(L.app.components.TritonServer):
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
