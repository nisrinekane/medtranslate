import subprocess
import sys
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path

# 1) KMP_DUPLICATE_LIB_OK works around the libiomp5 vs libomp140 OpenMP conflict
#    that occurs when torch (Intel OMP) and ctranslate2 (LLVM OMP) coexist
# 2) prepend the bundled nvidia DLL dirs to PATH so ctranslate2 can find
#    cuBLAS and cuDNN when it LoadLibrary's them lazily at inference time
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _site_pkgs = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if _site_pkgs.exists():
        _bin_dirs = [str(sub / "bin") for sub in _site_pkgs.iterdir() if (sub / "bin").is_dir()]
        if _bin_dirs:
            os.environ["PATH"] = os.pathsep.join(_bin_dirs) + os.pathsep + os.environ.get("PATH", "")
            for _bin in _bin_dirs:
                os.add_dll_directory(_bin)

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

from backend.prompts import GEMMA_TRANSLATE_SYSTEM, SOAP_SYSTEM, EXTRACT_SYSTEM


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
USE_GEMMA = os.environ.get("USE_GEMMA", "1") not in ("0", "false", "False")

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

LANG_NAME = {"en": "English", "es": "Spanish", "zh": "Mandarin Chinese"}

whisper_model = None
translator_en_es = None
translator_es_en = None
tokenizer_en_es = None
tokenizer_es_en = None
ollama_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    global translator_en_es, translator_es_en
    global tokenizer_en_es, tokenizer_es_en
    global ollama_client

    whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    # MarianMT as a degradation fallback if Ollama is unreachable
    # benchmark on validation/dataset-40.csv: Gemma 4 BLEU 72.7 vs MarianMT 49.9
    tokenizer_en_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    translator_en_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    tokenizer_es_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    translator_es_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    ollama_client = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=120.0)

    yield

    await ollama_client.aclose()


app = FastAPI(title="medtranslate", lifespan=lifespan)

# serve frontend assets (logo, images) under /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")


async def gemma_translate(text: str, direction: str) -> str:
    src, tgt = direction.split("-")
    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": GEMMA_TRANSLATE_SYSTEM.format(
                src=LANG_NAME.get(src, src), tgt=LANG_NAME.get(tgt, tgt))},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    r = await ollama_client.post("/api/chat", json=body)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def marian_translate(text: str, direction: str) -> str:
    if direction == "en-es":
        tokens = tokenizer_en_es(text, return_tensors="pt", padding=True)
        output = translator_en_es.generate(**tokens)
        return tokenizer_en_es.decode(output[0], skip_special_tokens=True)
    elif direction == "es-en":
        tokens = tokenizer_es_en(text, return_tensors="pt", padding=True)
        output = translator_es_en.generate(**tokens)
        return tokenizer_es_en.decode(output[0], skip_special_tokens=True)
    else:
        raise ValueError(f"MarianMT fallback only supports en-es / es-en, got {direction}")


async def translate(text: str, direction: str) -> str:
    if USE_GEMMA:
        try:
            return await gemma_translate(text, direction)
        except Exception:
            # fall through to MarianMT for en-es / es-en, re-raise for zh and other pairs
            if direction not in ("en-es", "es-en"):
                raise
    return marian_translate(text, direction)


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

    ollama_reachable = False
    if ollama_client is not None:
        try:
            r = await ollama_client.get("/api/tags", timeout=2.0)
            ollama_reachable = r.status_code == 200
        except Exception:
            pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "whisper_loaded": whisper_model is not None,
        "translators_loaded": translator_en_es is not None and translator_es_en is not None,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_reachable": ollama_reachable,
        "use_gemma": USE_GEMMA,
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
        src_lang = direction.split("-")[0]
        tgt_lang = direction.split("-")[1]
        segments, info = whisper_model.transcribe(tmp_path, language=src_lang)
        source_text = " ".join([s.text.strip() for s in segments])
    finally:
        os.unlink(tmp_path)

    translated_text = await translate(source_text, direction)

    return {
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
    }


class RetranslateRequest(BaseModel):
    text: str
    direction: str


@app.post("/retranslate")
async def retranslate(req: RetranslateRequest):
    translated_text = await translate(req.text, req.direction)
    return {"translated_text": translated_text}


class ConversationTurn(BaseModel):
    speaker: str  # "doctor" or "patient"
    source_text: str
    source_lang: str
    translated_text: str
    target_lang: str


class SoapRequest(BaseModel):
    conversation: list[ConversationTurn]
    chart_lang: str = "en"


@app.post("/soap")
async def soap_note(req: SoapRequest):
    chart_lang = req.chart_lang if req.chart_lang in LANG_NAME else "en"
    lines = []
    for turn in req.conversation:
        if turn.source_lang == chart_lang:
            text = turn.source_text
        elif turn.target_lang == chart_lang:
            text = turn.translated_text
        else:
            text = f"{turn.source_text} (translated: {turn.translated_text})"
        lines.append(f"{turn.speaker.upper()}: {text}")
    transcript = "\n".join(lines) if lines else "(empty visit)"

    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SOAP_SYSTEM.format(chart_lang_name=LANG_NAME[chart_lang])},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }
    r = await ollama_client.post("/api/chat", json=body)
    r.raise_for_status()
    return {"soap_note": r.json()["message"]["content"].strip(), "chart_lang": chart_lang}


@app.post("/extract")
async def extract(req: SoapRequest):
    chart_lang = "en"  # extraction always normalizes to english for the chart
    lines = []
    for turn in req.conversation:
        if turn.source_lang == chart_lang:
            text = turn.source_text
        elif turn.target_lang == chart_lang:
            text = turn.translated_text
        else:
            text = f"{turn.source_text} (translated: {turn.translated_text})"
        lines.append(f"{turn.speaker.upper()}: {text}")
    transcript = "\n".join(lines) if lines else "(empty visit)"

    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 1024},
    }
    r = await ollama_client.post("/api/chat", json=body)
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()
    # Ollama returns the JSON as a string, parse it server-side so the frontend gets a real object
    import json as _json
    try:
        chart = _json.loads(content)
    except _json.JSONDecodeError:
        chart = {"raw": content, "parse_error": True}
    return chart


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")