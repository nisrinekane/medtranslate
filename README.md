# MedTranslate

A local-first medical interpreter for clinics that lack on-shift human interpreters and reliable internet. The doctor and patient speak into a microphone, the app transcribes the speech, translates between languages, plays the translation back through the browser's speech synthesis, and can generate a SOAP note plus a structured chart from the conversation.

All inference happens locally. No patient data ever leaves the machine running the app.

## What it does

* Real-time two-way speech translation across multiple language pairs (English, Spanish, Mandarin, Arabic, Armenian).
* Editable transcripts. Either side can click a message and edit the text. Source edits trigger a fresh translation, target edits replay the audio.
* SOAP note generation. One click at the end of the visit turns the conversation into a Subjective/Objective/Assessment/Plan note ready to paste into a chart.
* Live structured extraction. As the conversation progresses, a side panel auto-fills chief complaint, symptoms with onset and duration, medications, allergies and vitals.

## Architecture

```
microphone -> browser MediaRecorder -> POST /transcribe
                                           |
                                           v
                                  Whisper (faster-whisper, CUDA)
                                           |
                                     transcribed text
                                           |
                                           v
                                  Gemma 4 via Ollama
                                           |
                                     translated text
                                           |
                                           v
                                  browser SpeechSynthesis
                                  (falls back to /tts -> eSpeak NG
                                   for languages with no OS voice)
```

The same Gemma 4 model powers three jobs through different system prompts (see `backend/prompts.py`):

1. **Translation** for `/transcribe` and `/retranslate`, with a formal-register medical interpreter prompt.
2. **SOAP note** for `/soap`, with a clinical-scribe prompt that refuses to fabricate findings.
3. **Structured extraction** for `/extract`, using Ollama's JSON mode to return a typed chart object.

The FastAPI backend serves the HTML frontend at `/` and assets under `/static`. The frontend is a single file (`frontend/index.html`) with no build step.

## Languages supported

* English <> Spanish, both directions
* English <> Mandarin, both directions
* English <> Arabic, both directions
* English <> Armenian, both directions

## Requirements

* Python 3.10 or newer.
* An NVIDIA GPU with CUDA 12 drivers for the production Whisper + Gemma 4 setup. CPU is supported via environment variables for development.
* Ollama installed and running, with the Gemma 4 model pulled (`ollama pull gemma4:e2b` for an 8 GB GPU, `gemma4:e4b` for 12+ GB).
* eSpeak NG installed, used as a fallback TTS engine for languages the browser does not have an OS-installed voice for (e.g. Armenian on Windows). On Windows: `winget install eSpeak-NG.eSpeak-NG`. On Linux: `apt install espeak-ng`. On Mac: `brew install espeak-ng`.

## Running it

You can run the app two ways. Pick whichever fits your setup.

### Option A: Docker Compose (recommended for clean reproducible deploy)

Requires Docker, an NVIDIA GPU, and the `nvidia-container-toolkit` installed on the host so containers can use the GPU.

1. Start both services.
   ```
   docker compose up -d
   ```

2. Pull the Gemma 4 model into the Ollama container (one time only, the volume persists across restarts).
   ```
   docker compose exec ollama ollama pull gemma4:e2b
   ```

3. Open `http://127.0.0.1:8080` in any modern browser. Allow microphone access when prompted.

To stop:
```
docker compose down
```

To swap the Gemma variant, change `OLLAMA_MODEL` in `docker-compose.yml`, pull the new model with `docker compose exec ollama ollama pull <tag>`, then `docker compose restart backend`.

### Option B: Native uvicorn + native Ollama (recommended for development)

Requires Python 3.10+ and an NVIDIA GPU with CUDA 12 drivers. CPU is supported via environment variables.

1. Install Ollama from `https://ollama.com` and pull the model.
   ```
   ollama pull gemma4:e2b
   ```

2. Create a virtual environment and install dependencies.
   ```
   python -m venv .venv
   .venv\Scripts\activate     (Windows)
   source .venv/bin/activate   (Linux or Mac)
   pip install -r requirements.txt
   ```

3. Start the backend.
   ```
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

4. Open `http://127.0.0.1:8000` in any modern browser. Allow microphone access when prompted.

## Configuration

All runtime config is driven by environment variables so the same code runs on a dev laptop and a clinic mini-PC.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama serves the model. Point at a remote Ollama for split deployment. |
| `OLLAMA_MODEL` | `gemma4:e2b` | Gemma variant. Swap for `gemma4:e4b` or larger on bigger GPUs. |
| `WHISPER_MODEL` | `large-v3` | Whisper checkpoint. Use `small` or `base` on CPU. |
| `WHISPER_DEVICE` | `cuda` | Set to `cpu` for CPU-only machines. |
| `WHISPER_COMPUTE_TYPE` | `float16` | Use `int8` for CPU. |

## Project structure

```
backend/
  main.py        FastAPI app, routes, model loading, lifespan
  prompts.py     System prompts for translation, SOAP, extraction
frontend/
  index.html    Single-file UI with embedded CSS and JS
validation/
  dataset-en-es.csv   40 medical phrases with reference translations
  score_gemma.py      Benchmarks Gemma 4 against the dataset with BLEU and chrF
requirements.txt
README.md
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Serves the frontend |
| GET | `/health` | Reports GPU, model load state and Ollama reachability |
| POST | `/transcribe` | Audio in, transcription plus translation out |
| POST | `/retranslate` | Re-translates text after a user edits a source bubble |
| POST | `/soap` | Conversation in, SOAP note out |
| POST | `/extract` | Conversation in, JSON chart out |
| POST | `/tts` | Server-side eSpeak NG synthesis, used by the frontend as a fallback when the browser has no OS-installed voice for the target language |

## Privacy

The production deployment runs entirely on a clinic-local machine. No audio, no transcript, no chart and no patient data is sent to any third party. Ollama runs the model on local hardware. Browser speech synthesis is also local. The only data crossing the network is between the patient's device and the clinic's own server on the local network.