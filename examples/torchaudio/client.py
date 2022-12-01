import torchaudio, requests

waveform, sample_rate = torchaudio.load("../../audio_file.wav")
waveform =  waveform[0,:].tolist()
waveform.append(sample_rate)
waveform = ' '.join(str(x) for x in waveform)

response = requests.post("http://127.0.0.1:7777/predict", json={"waveform": waveform})
print(response.json())
