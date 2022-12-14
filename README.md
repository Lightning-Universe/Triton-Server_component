# Triton Server on Lightning AI

## Introduction

Triton server component enables you to deploy your model to Triton Inference Server and setup a FastAPI interface
for converting api datatypes (`string`, `integer`, `float` etc) to and from Triton datatypes (`DT_STRING`, `DT_INT32` etc).

## What is Triton

Triton Inference Server is an open-source deep learning inference server designed by Nvidia to make AI model 
deployment easy and efficient.  It supports multiple model formats and hardware platforms, and help utilize the compute
efficiently by batching requests and optimizing the model execution. For more details, refer the
[developer blog](https://developer.nvidia.com/nvidia-triton-inference-server) from Nvidia

## Let's do an example

We'll use the Triton Server component in this example to serve a torch vision model

Save the following code as `torch_vision_server.py`

```python
# !pip install torch torchvision pillow
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Server-Component.git
import lightning as L
import base64, io, torch, torchvision, lightning_triton as lt
from PIL import Image


class TorchvisionServer(lt.TritonServer):
    def __init__(self, input_type=lt.Image, output_type=lt.Category,**kwargs):
        super().__init__(input_type=input_type,
                         output_type=output_type,
                         max_batch_size=8, **kwargs)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None

    def setup(self):
        self._model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self._model.to(self._device)

    def predict(self, request):
        image = base64.b64decode(request.image.encode("utf-8"))
        image = Image.open(io.BytesIO(image))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transforms(image)
        image = image.to(self._device)
        prediction = self._model(image.unsqueeze(0))
        return {"category": prediction.argmax().item()}


cloud_compute = L.CloudCompute("gpu", shm_size=512)
app = L.LightningApp(TorchvisionServer(cloud_compute=cloud_compute))
```

### Install lightning

If you don't have lightning installed yet, install it using

```bash
pip install -U lightning
```

### Run it locally

Since installing Triton can be tricky (and not officially supported) in different operating systems, 
we use docker internally to run the Triton server. This component expects the docker is already installed in
your system. If you don't have docker installed, you can install it from [here](https://docs.docker.com/get-docker/)

Note that you don't need to install docker if you are running the component only on cloud. 
Keep in mind that the docker image is very huge (about 20 GB) and can affect the startup time on the
first time you run it. 

Run it locally using

```bash
lightning run app torch_vision_server.py --setup
```

### Run it in the cloud

Run it in the cloud using

```bash
lightning run app torch_vision_server.py --setup --cloud
```

## More examples

Check out more examples that serve different model types in the example directory. 
Follow the instructions for each of those here

- [Stable Diffusion](examples/stable-diffusion/README.md)
- [Image Classification using Torch vision](examples/torchvision/README.md)
- [Audio Transcription using Torch Audio](examples/torchaudio/README.md)


## Benchmark

Triton Server is in very early stages of development and is not yet optimized for performance. But we'll be tracking the
progress with the help of benchmarks provided in this section. Here we are comparing the performance of Triton Server with
[PythonServer](https://github.com/Lightning-AI/lightning/blob/master/src/lightning_app/components/serve/python_server.py).
Below given are the results of benchmarking on two different GPU instances using the stable diffusion component. 
For more details, refer the [benchmarking](https://github.com/Lightning-AI/lightning-diffusion-component/blob/main/benchmarks/README.md)
section of stable diffusion component.

|      **Device**       | **Server Type** | **Req/Sec** | **Latency** | **Batch Size** |
|:---------------------:|-----------------|:-----------:|:-----------:|:--------------:|
| gpu-rtx (g5.2xlarge)  | PythonServer    |    ~0.2     |     7s      |       1        |
| gpu-rtx (g5.2xlarge)  | TritonServer    |    ~0.1     |    7.3s     |       1        |
| gpu-fast (p3.2xlarge) | PythonServer    |    ~0.2     |     6s      |       1        |
| gpu-fast (p3.2xlarge) | TritonServer    |    ~0.1     |    7.5s     |       1        |


## Next Steps

At present, our focus is on improving the performance of Triton Server and that includes tackling the following issues

- [ ] Dynamic batching with python backend
- [ ] Supporting TensorRT backend
- [ ] Dynamic batching with TensorRT backend
- [ ] Concurrent model execution


## Known Limitations

This component is still in the early stages of development. Here are some of the known limitations that are being
worked on: If you have issues with any of these or if you find other issues, please create a 
[Github issue](https://github.com/Lightning-AI/LAI-Triton-Server-Component/issues/new) so we can prioritise them.

- [ ] When running locally, it requires ctrl-c to be pressed twice to stop all the processes
- [ ] Running locally requires docker to be installed
- [ ] Only python backend is supported for the Triton server. This means, a lot of optimizations
      specific to other backends, like TensorRT for example, cannot be utilized with this component yet
- [ ] Not all the features of Triton are configurable through the component yet.
- [ ] Only four datatypes are supported at the API level (`string`, `integer`, `float`, `bool`)
- [ ] Providing a pre-created Model Repository to the component is not supported yet. This means if you have an existing
      model repository, you cannot use it with this component yet
