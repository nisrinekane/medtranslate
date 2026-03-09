import subprocess
import tempfile
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer


whisper_model = None
translator_en_es = None
translator_es_en = None
tokenizer_en_es = None
tokenizer_es_en = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    global translator_en_es, translator_es_en
    global tokenizer_en_es, tokenizer_es_en

    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    # load translation models
    tokenizer_en_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    translator_en_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    tokenizer_es_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    translator_es_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    yield


app = FastAPI(title="medtranslate", lifespan=lifespan)


def translate(text: str, direction: str) -> str:
    if direction == "en-es":
        tokens = tokenizer_en_es(text, return_tensors="pt", padding=True)
        output = translator_en_es.generate(**tokens)
        return tokenizer_en_es.decode(output[0], skip_special_tokens=True)
    else:
        tokens = tokenizer_es_en(text, return_tensors="pt", padding=True)
        output = translator_es_en.generate(**tokens)
        return tokenizer_es_en.decode(output[0], skip_special_tokens=True)


@app.get("/health")
async def health():
    gpu_available = False
    gpu_name = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_available = True
            gpu_name = result.stdout.strip()
    except FileNotFoundError:
        pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "whisper_loaded": whisper_model is not None,
        "translators_loaded": translator_en_es is not None and translator_es_en is not None
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    direction: str = Form(...)
):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        lang = "en" if direction == "en-es" else "es"
        segments, info = whisper_model.transcribe(tmp_path, language=lang)
        source_text = " ".join([s.text.strip() for s in segments])
    finally:
        os.unlink(tmp_path)

    translated_text = translate(source_text, direction)

    return {
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": "en" if direction == "en-es" else "es",
        "target_lang": "es" if direction == "en-es" else "en"
    }


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")