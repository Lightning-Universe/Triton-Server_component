# LAI-Triton-Serve-Component

Triton Serve Component for lightning.ai

## Introduction

Triton serve component enables you to deploy your model to Triton Inference Server and setup a FastAPI interface
for converting api datatypes (string, int, float etc) to and from Triton datatypes (DT_STRING, DT_INT32 etc).

## Example

### Install the component

Install the component using pip

```bash
pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
```

### Install docker (for running the component locally)

Since installing triton can be tricky in different operating systems, we use docker internally to run
the triton server. This component expects the docker is already installed in your system. Note that you
don't need to install docker if you are running the component only on cloud.

### Save the app file

Save the following code as `app.py`

```python
# !pip install torchvision pillow
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
import lightning as L
import base64, io, torchvision, lightning_triton
from PIL import Image as PILImage


class TorchVisionServer(lightning_triton.TritonServer):
    def __init__(self, input_type=L.app.components.Image, output_type=L.app.components.Number):
        super().__init__(input_type=input_type,
                         output_type=output_type,
                         cloud_compute=L.CloudCompute("gpu", shm_size=512),
                         max_batch_size=8)
        self._model = None

    def setup(self):
        self._model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
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


app = L.LightningApp(TorchVisionServer())
```

### Run the app

Run the app locally using the following command

```bash
lightning run app app.py --setup
```

### Run the app in cloud

Run the app in cloud using the following command

```bash
lightning run app app.py --setup --cloud
```

## known Limitations

- [ ] When running locally, ctrl-c not terminating all the processes
- [ ] Running locally requires docker to be installed
- [ ] Only python backend is supported for the triton server
- [ ] Not all the features of triton are configurable through the component
- [ ] Only four datatypes are supported at the API level (string, int, float, bool)
- [ ] Providing the model_repository directly to the component is not supported yet
