import multiprocessing
import time

import sys
import abc
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import requests
import torch
import tritonclient.http as httpclient
import uvicorn
from fastapi import FastAPI
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.app.utilities.network import find_free_network_port
from lightning.app.utilities.packaging.build_config import BuildConfig
from lightning.app.utilities.packaging.cloud_compute import CloudCompute
from tritonclient.utils import np_to_triton_dtype

from lightning_triton import safe_pickle
from lightning_triton.base import ServeBase

logger = Logger(__name__)

MODEL_NAME = "lightning-triton"
LIGHTNING_TRITON_BASE_IMAGE = os.getenv(
    "LIGHTNING_TRITON_BASE_IMAGE", "ghcr.io/gridai/lightning-triton:v0.22"
)
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
MIN_NVIDIA_DRIVER_REQUIREMENT_MAP = {"22.10": "520"}


environment = jinja2.Environment()
template = environment.from_string(
    """name: "lightning-triton"
backend: "{{ backend }}"
max_batch_size: {{ max_batch_size }}
default_model_filename: "__lightningapp_triton_model_file.py"

input [
{% for input in inputs %}
  {
    name: "{{ input.name }}"
    data_type: {{ input.type }}
    dims: {{ input.dim }}
  }{{ "," if not loop.last else "" }}
{% endfor %}
]
output [
{% for output in outputs %}
  {
    name: "{{ output.name }}"
    data_type: {{ output.type }}
    dims: {{ output.dim }}
  }{{ "," if not loop.last else "" }}
{% endfor %}
]

instance_group [
  {
    kind: KIND_{{ kind }}
  }
]
"""
)


triton_model_file_template = """
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class Request:
    pass


class TritonPythonModel:

    def __init__(self):
        self.work = None
        self.model_config = {}

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        sys.path.insert(1, str(Path.cwd()))
        self.work = pickle.load(open("__model_repository/lightning-triton/1/__lightning_work.pkl", "rb"))
        sys.path.pop(1)
        self.work.setup()

    def execute(self, requests):
        responses = []
        for request in requests:
            req = Request()
            for inp in self.model_config['input']:
                i = pb_utils.get_input_tensor_by_name(request, inp['name'])
                if inp['data_type'] == 'TYPE_STRING':
                    ip = i.as_numpy()[0][0].decode()
                else:
                    ip = i.as_numpy()[0][0]
                setattr(req, inp['name'], ip)
            resp = self.work.predict(req)
            for out in self.model_config['output']:
                if out['name'] not in resp:
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(f"Output {out['name']} not found in response")))
                    continue
                val = [resp[out['name']]]
                dtype = pb_utils.triton_string_to_numpy(out['data_type'])
                out = pb_utils.Tensor(
                    out['name'],
                    np.array(val, dtype=dtype),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
"""


PYDANTIC_TO_NUMPY = {
    "integer": np.int32,
    "number": np.float64,
    "string": np.dtype("object"),
    "boolean": np.dtype("bool"),
}


def pydantic_to_numpy_dtype(pydantic_obj_string):
    return PYDANTIC_TO_NUMPY[pydantic_obj_string]


PYDANTIC_TO_TRITON = {
    "integer": "TYPE_INT32",
    "number": "TYPE_FP64",
    "string": "TYPE_STRING",
    "boolean": "TYPE_BOOL",
}


def pydantic_to_triton_dtype_string(pydantic_obj_string):
    return PYDANTIC_TO_TRITON[pydantic_obj_string]


class TritonServer(ServeBase, abc.ABC):
    def __init__(self, *args, max_batch_size=8, backend="python", **kwargs):
        cloud_build_config = kwargs.get(
            "cloud_build_config", BuildConfig(image=LIGHTNING_TRITON_BASE_IMAGE)
        )
        compute_config = kwargs.get("cloud_compute", CloudCompute("cpu", shm_size=512))
        if compute_config.shm_size < 256:
            raise ValueError(
                "Triton expects the shared memory size (shm_size) to be at least 256MB"
            )
        kwargs["cloud_build_config"] = cloud_build_config
        kwargs["cloud_compute"] = compute_config
        super().__init__(
            *args,
            **kwargs,
        )
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be greater than 0")
        self.max_batch_size = max_batch_size
        if backend != "python":
            raise ValueError(
                "Currently only python backend is supported. But we are looking for user feedback to"
                "support other backends too. Please reach out to our slack channel or "
                "support@lightning.ai"
            )
        self.backend = backend
        self._triton_server_process = None
        self._fastapi_process = None

    @abc.abstractmethod
    def predict(self, request: Any) -> Any:
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic
        """
        pass

    def _attach_triton_proxy_fn(self, fastapi_app: FastAPI, triton_port: int):
        input_type: Any = self.configure_input_type()
        output_type: Any = self.configure_output_type()

        client = httpclient.InferenceServerClient(
            url=f"127.0.0.1:{triton_port}", connection_timeout=1200.0, network_timeout=1200.0
        )

        def proxy_fn(request: input_type):  # type: ignore
            # TODO - test with multiple input and multiple output
            inputs = []
            outputs = []
            for property_name, property in input_type.schema()["properties"].items():
                val = [getattr(request, property_name)]
                dtype = pydantic_to_numpy_dtype(property["type"])
                arr = np.array(val, dtype=dtype).reshape((-1, 1))
                data = httpclient.InferInput(
                    property_name, arr.shape, np_to_triton_dtype(arr.dtype)
                )
                data.set_data_from_numpy(arr)
                inputs.append(data)
            for property_name, property in output_type.schema()["properties"].items():
                output = httpclient.InferRequestedOutput(property_name)
                outputs.append(output)
            query_response = client.infer(
                model_name="lightning-triton",
                inputs=inputs,
                outputs=outputs,
                timeout=1200,
            )
            response = {}
            for property_name, property in output_type.schema()["properties"].items():
                # TODO - test with image output if decode is required
                response[property_name] = query_response.as_numpy(property_name).item()
            return response

        fastapi_app.post("/predict", response_model=output_type)(proxy_fn)

    def _get_config_file(self) -> str:
        """Create config.pbtxt file specific for triton-python backend"""
        kind = "GPU" if self.device.type == "cuda" else "CPU"
        input_types = self.configure_input_type()
        output_types = self.configure_output_type()
        inputs = []
        outputs = []
        for k, v in input_types.schema()["properties"].items():
            inputs.append(
                {
                    "name": k,
                    "type": pydantic_to_triton_dtype_string(v["type"]),
                    "dim": "[1]",
                }
            )
        for k, v in output_types.schema()["properties"].items():
            outputs.append(
                {
                    "name": k,
                    "type": pydantic_to_triton_dtype_string(v["type"]),
                    "dim": "[1]",
                }
            )
        return template.render(
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            max_batch_size=self.max_batch_size,
            backend=self.backend,
        )

    def _setup_model_repository(self):
        # create the model repository directory
        cwd = Path.cwd()
        if (cwd / "__model_repository").is_dir():
            shutil.rmtree(cwd / "__model_repository")
        repo_path = cwd / f"__model_repository/{MODEL_NAME}/1"
        repo_path.mkdir(parents=True, exist_ok=True)

        # setting the model file
        (repo_path / "__lightningapp_triton_model_file.py").write_text(
            triton_model_file_template
        )

        with open(repo_path / "__lightning_work.pkl", "wb+") as f:
            safe_pickle.dump(self, f)

        # setting the config file
        config = self._get_config_file()
        config_path = repo_path.parent
        with open(config_path / "config.pbtxt", "w") as f:
            f.write(config)

    @staticmethod
    def _check_nvidia_driver_compatibility():
        cmd = shlex.split("nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0")
        try:
            version = subprocess.check_output(cmd).decode().strip()
        except FileNotFoundError:
            raise RuntimeError(
                "nvidia-smi is not found. Please make sure that nvidia-smi is installed and available in the PATH"
            )
        triton_version = "22.10"
        if version < MIN_NVIDIA_DRIVER_REQUIREMENT_MAP[triton_version]:
            raise RuntimeError(
                f"Your nvidia driver version is {version}."
                f"Lightning Triton {triton_version} requires nvidia driver "
                f"version >= {MIN_NVIDIA_DRIVER_REQUIREMENT_MAP[triton_version]}"
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run method takes care of configuring and setting up a FastAPI server behind the scenes.

        Normally, you don't need to override this method.
        """
        if self.device == torch.device("cuda"):
            self._check_nvidia_driver_compatibility()

        self._setup_model_repository()

        triton_port = find_free_network_port()

        # start triton server in subprocess
        TRITON_SERVE_COMMAND = f"tritonserver --model-repository __model_repository --http-port {triton_port}"
        if is_running_in_cloud():
            _triton_server_process = subprocess.Popen(shlex.split(TRITON_SERVE_COMMAND))
        else:
            # locally, installing triton is painful and hence we'll call into docker
            base_image = LIGHTNING_TRITON_BASE_IMAGE

            # we try to get the entrypoint file name from sys.argv. Ideally, the AppRef should have
            # the information about entrypoint file too. TODO @sherin
            for i, arg in enumerate(sys.argv):
                if arg == "run" and sys.argv[i + 1] == "app":
                    entrypoint_file = sys.argv[i + 2]
                    break
            else:
                raise ValueError("Could not find the entrypoint file. Please create an issue "
                                 "on github with a reproducible script")
            cmd = f'bash -c "bash /usr/local/bin/docker_script.sh {entrypoint_file}; {TRITON_SERVE_COMMAND}"'
            first = f"docker run -it --shm-size=256m --rm -p {triton_port}:{triton_port} -v {Path.cwd()}:/__model_artifacts/ "
            middle = ""
            if self.device == torch.device("cuda"):
                middle += " --gpus all --env NVIDIA_VISIBLE_DEVICES=all "
            last = f"{base_image} {cmd}"
            docker_cmd = shlex.split(first + middle + last)
            _triton_server_process = subprocess.Popen(docker_cmd)

        # check if triton server is up
        while True:
            try:
                requests.get(f"http://127.0.0.1:{triton_port}")
            except requests.exceptions.ConnectionError:
                time.sleep(0.3)
            else:
                time.sleep(0.5)
                break

        url = self._future_url if self._future_url else self.url
        if not url:
            # if the url is still empty, point it to localhost
            url = f"http://127.0.0.1:{self.port}"
        url = f"{url}/predict"

        fastapi_proc = multiprocessing.Process(
            target=_run_fast_api,
            args=(
                safe_pickle.get_picklable_work(self),
                triton_port,
                url,
                self.host,
                self.port,
            )
        )
        fastapi_proc.start()
        logger.info(
            f"Your app has started. View it in your browser: http://{self.host}:{self.port}"
        )

        self._triton_server_process = _triton_server_process
        self._fastapi_process = fastapi_proc

        fastapi_proc.join()

    def on_exit(self):
        if self._triton_server_process:
            self._triton_server_process.kill()
        if self._fastapi_process:
            self._fastapi_process.kill()


def _run_fast_api(self, triton_port, url, fastapi_host, fastapi_port):
    # setting and exposing the fast api service that sits in front of triton server
    fastapi_app = FastAPI()
    self._attach_triton_proxy_fn(fastapi_app, triton_port)
    self._attach_frontend(fastapi_app, url)
    config = uvicorn.Config(fastapi_app, host=fastapi_host, port=fastapi_port, log_level="error")
    server = uvicorn.Server(config=config)
    server.run()
