# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml

import base64
import io
import os

import lightning as L
import torch
from ldm.lightning import LightningStableDiffusion, PromptDataset

import lightning_triton as lt


class StableDiffusionServer(lt.TritonServer):
    def __init__(self, input_type=lt.Text, output_type=lt.Image, **kwargs):
        super().__init__(input_type=input_type, output_type=output_type, max_batch_size=8, **kwargs)
        self._model = None
        self._trainer = None

    def setup(self):
        os.system(
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        )
        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        self._model = LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=768,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        imgs_ = self._trainer.predict(
            self._model,
            torch.utils.data.DataLoader(PromptDataset([request.text])),
        )
        image = imgs_[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"{img_str}"}


app = L.LightningApp(StableDiffusionServer())
