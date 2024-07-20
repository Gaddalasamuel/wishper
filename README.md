## Setup
We used Python 3.9.9 and PyTorch 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably OpenAI's tiktoken for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

```sh
pip install -U openai-whisper
```

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

```sh
pip install git+https://github.com/openai/whisper.git 
```

It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:
```sh
 #on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

#on Windows using Chpip install git+https://github.com/openai/whisper.git ocolatey (https://chocolatey.org/)
choco install ffmpeg

#on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Python usage

Transcription can also be performed within Python:

```python
import whisper  
model = whisper.load_model("base")
result = model.transcribe("audio.mp3") #repalce audio with file path and insert audio file   
print(result["text"])
```

Internally, the transcribe() method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of whisper.detect_language() and whisper.decode() which provide lower-level access to the model.

 ```python
 import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```
