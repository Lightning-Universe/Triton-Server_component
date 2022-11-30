# LAI-Triton-Serve-Component

Triton Serve Component for lightning.ai

## Introduction

Triton serve component enables you to deploy your model to Triton Inference Server and setup a FastAPI interface
for converting api datatypes (string, int, float etc) to and from Triton datatypes (DT_STRING, DT_INT32 etc).

## Example building a TorchVisionServe Component


### Install docker (for running the component locally)

Since installing triton can be tricky (and not officially supported) in different operating systems, 
we use docker internally to run the triton server. This component expects the docker is already installed in
your system.
Note that you don't need to install docker if you are running the component only on cloud.

### Save the component into a file

Save the following code as `torch_vision_server.py`

```python
# !pip install torchvision pillow
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
import lightning as L
import base64, io, torchvision, lightning_triton
from PIL import Image as PILImage


class TorchvisionServer(lightning_triton.TritonServer):
    def __init__(self, input_type=lightning_triton.Image, output_type=lightning_triton.Category, **kwargs):
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
```

### Run it locally

Run it locally using

```bash
lightning run app torch_vision_server.py --setup
```

### Run it in the cloud

Run it in the cloud using

```bash
lightning run app torch_vision_server.py --setup --cloud
```


## known Limitations

- [ ] When running locally, ctrl-c not terminating all the processes
- [ ] Running locally requires docker to be installed
- [ ] Only python backend is supported for the triton server
- [ ] Not all the features of triton are configurable through the component
- [ ] Only four datatypes are supported at the API level (string, int, float, bool)
- [ ] Providing the model_repository directly to the component is not supported yet
