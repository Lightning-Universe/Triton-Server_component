import abc
import base64
from typing import Any, Dict, Optional

import requests
import torch
from fastapi import FastAPI
from lightning.app import LightningWork
from lightning.app.utilities.app_helpers import Logger
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

logger = Logger(__name__)


class _DefaultInputData(BaseModel):
    payload: str


class _DefaultOutputData(BaseModel):
    prediction: str


class Image(BaseModel):
    image: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        url = "https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png"
        img = requests.get(url).content
        img = base64.b64encode(img).decode("UTF-8")
        return {"image": img}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return """import base64
from pathlib import Path
import requests

img = requests.get("https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png").content
img = base64.b64encode(img).decode("UTF-8")
response = requests.post('""" + url + """', json={
    "image": img
})"""

    @staticmethod
    def response_code_sample() -> str:
        return """img = response.json()["image"]
img = base64.b64decode(img.encode("utf-8"))
Path("response.png").write_bytes(img)
"""


class Category(BaseModel):
    category: Optional[int]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"prediction": 463}

    @staticmethod
    def response_code_sample() -> str:
        return """print("Predicted category is: ", response.json()["category"]) 
"""


class Text(BaseModel):
    text: Optional[str]

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        return {"text": "A portrait of a person looking away from the camera"}

    @staticmethod
    def request_code_sample(url: str) -> str:
        return """import base64
from pathlib import Path
import requests

response = requests.post('""" + url + """', json={
    "text": "A portrait of a person looking away from the camera"
})
"""


class ServeBase(LightningWork, abc.ABC):
    def __init__(  # type: ignore
        self,
        host: str = "127.0.0.1",
        port: int = 7777,
        input_type: type = _DefaultInputData,
        output_type: type = _DefaultOutputData,
        **kwargs,
    ):
        super().__init__(parallel=True, host=host, port=port, **kwargs)
        if not issubclass(input_type, BaseModel):
            raise TypeError("input_type must be a pydantic BaseModel class")
        if not issubclass(output_type, BaseModel):
            raise TypeError("output_type must be a pydantic BaseModel class")
        self._supported_pydantic_types = {"string", "number", "integer", "boolean"}
        self._input_type = self._verify_type(input_type)
        self._output_type = self._verify_type(output_type)

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def configure_input_type(self) -> type:
        """Override this method to configure the input type for the API.

        By default, it is set to `_DefaultInputData`
        """
        return self._input_type

    def configure_output_type(self) -> type:
        """Override this method to configure the output type for the API.

        By default, it is set to `_DefaultOutputData`
        """
        return self._output_type

    def _verify_type(self, datatype: Any):
        props = datatype.schema()["properties"]
        for k, v in props.items():
            if v["type"] not in self._supported_pydantic_types:
                raise TypeError("Unsupported type")
        return datatype

    def setup(self, *args, **kwargs) -> None:
        """This method is called before the server starts. Override this if you need to download the model or
        initialize the weights, setting up pipelines etc.

        Note that this will be called exactly once on every work machines. So if you have multiple machines for serving,
        this will be called on each of them.
        """
        return

    @abc.abstractmethod
    def predict(self, request: Any) -> Any:
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic
        """
        pass

    def get_code_sample(self, url: str) -> Optional[str]:
        input_type: Any = self.configure_input_type()
        output_type: Any = self.configure_output_type()

        if not (
                hasattr(input_type, "request_code_sample") and
                hasattr(output_type, "response_code_sample")
        ):
            return None
        return f"{input_type.request_code_sample(url)}\n{output_type.response_code_sample()}"

    @staticmethod
    def _get_sample_dict_from_datatype(datatype: Any) -> dict:
        if hasattr(datatype, "get_sample_data"):
            return datatype.get_sample_data()

        datatype_props = datatype.schema()["properties"]
        out: Dict[str, Any] = {}
        for k, v in datatype_props.items():
            if v["type"] == "string":
                out[k] = "data string"
            elif v["type"] == "number":
                out[k] = 0.0
            elif v["type"] == "integer":
                out[k] = 0
            elif v["type"] == "boolean":
                out[k] = False
            else:
                raise TypeError("Unsupported type")
        return out

    def _attach_infer_fn(self, fastapi_app: FastAPI) -> None:
        input_type: type = self.configure_input_type()
        output_type: type = self.configure_output_type()

        def infer_fn(request: input_type):  # type: ignore
            with torch.inference_mode():
                return self.predict(request)

        fastapi_app.post("/predict", response_model=output_type)(infer_fn)

    def _attach_frontend(self, fastapi_app: FastAPI) -> None:
        from lightning_api_access import APIAccessFrontend

        class_name = self.__class__.__name__
        url = self._future_url if self._future_url else self.url
        if not url:
            # if the url is still empty, point it to localhost
            url = f"http://127.0.0.1:{self.port}"
        url = f"{url}/predict"
        request, response = {}, {}
        datatype_parse_error = False
        try:
            request = self._get_sample_dict_from_datatype(self.configure_input_type())
        except TypeError:
            datatype_parse_error = True

        try:
            response = self._get_sample_dict_from_datatype(self.configure_output_type())
        except TypeError:
            datatype_parse_error = True

        if datatype_parse_error:

            @fastapi_app.get("/")
            def index() -> str:
                return (
                    "Automatic generation of the UI is only supported for simple, "
                    "non-nested datatype with types string, integer, float and boolean"
                )

            return

        frontend_payload = {
            "name": class_name,
            "url": url,
            "method": "POST",
            "response": response,
            "request": request
        }
        code_sample = self.get_code_sample(url)
        if code_sample:
            frontend_payload["code_sample"] = code_sample

        frontend = APIAccessFrontend(apis=[frontend_payload])
        fastapi_app.mount(
            "/", StaticFiles(directory=frontend.serve_dir, html=True), name="static"
        )
