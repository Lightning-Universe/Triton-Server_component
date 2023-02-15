# !pip install torchaudio
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git

import lightning as L
import torch
import torchaudio

import lightning_triton as lt


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


class TorchAudioServe(lt.TritonServer):
    def __init__(self, input_type=lt.WaveForm, output_type=lt.Text):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            cloud_compute=L.CloudCompute("gpu-rtx", shm_size=512),
            max_batch_size=8,
        )
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None

    def setup(self):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self._model = bundle.get_model().to(self._device)
        self._model._decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        self._model.sample_rate = bundle.sample_rate

    def predict(self, request):
        aux = request.waveform.split(" ")
        waveform = torch.FloatTensor([float(i) for i in aux[:-1:]]).to(self._device)
        print("Fixing sample rate")
        if int(aux[-1]) != self._model.sample_rate:
            waveform = torchaudio.functional.resample(waveform, int(aux[-1]), self._model.sample_rate)

        with torch.inference_mode():
            emission, _ = self._model(waveform.unsqueeze(0))
        print("transcription")
        transcripts = self._model._decoder(emission[0])
        return {"text": str(transcripts)}


app = L.LightningApp(TorchAudioServe())
