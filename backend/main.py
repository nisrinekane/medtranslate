import subprocess
import sys
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path

# this block is skipped on linux and mac where neither workaround is needed
# on windows, two extra setup steps run before torch and faster-whisper are imported
# KMP_DUPLICATE_LIB_OK works around the libiomp5 (torch) vs libomp140 (ctranslate2)
# openmp conflict that otherwise crashes the boot with "OMP: Error #15"
# the PATH prepend lets ctranslate2 find cuBLAS and cuDNN dlls bundled via pip
# (it loads them lazily at inference so os.add_dll_directory alone is not enough)
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _site_pkgs = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if _site_pkgs.exists():
        _bin_dirs = [str(sub / "bin") for sub in _site_pkgs.iterdir() if (sub / "bin").is_dir()]
        if _bin_dirs:
            os.environ["PATH"] = os.pathsep.join(_bin_dirs) + os.pathsep + os.environ.get("PATH", "")
            for _bin in _bin_dirs:
                os.add_dll_directory(_bin)

import shutil

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

from backend.prompts import GEMMA_TRANSLATE_SYSTEM, SOAP_SYSTEM, EXTRACT_SYSTEM


# resolve the espeak-ng binary at import time so each /tts call avoids a PATH lookup
# falls back to the standard Windows install path if not on PATH
ESPEAK_BIN = (
    shutil.which("espeak-ng")
    or shutil.which("espeak-ng.exe")
    or r"C:\Program Files\eSpeak NG\espeak-ng.exe"
)


# all runtime config is env-driven so the same code runs on a dev laptop and a clinic mini-PC
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")  # where Ollama serves the model
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")  # gemma variant, swap for e4b / 26b on bigger gpus

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")  # large-v3 for prod, small for cpu dev
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")  # int8 for cpu

# iso codes mapped to the human-readable names the gemma prompts reference
LANG_NAME = {
    "en": "English",
    "es": "Spanish",
    "zh": "Mandarin Chinese",
    "ar": "Arabic",
    "hy": "Armenian",
}

# models and clients populated by lifespan(), kept module-level so route handlers can reach them
whisper_model = None
ollama_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # loads heavy models once at startup so requests don't pay the load cost
    global whisper_model
    global ollama_client

    # Whisper handles speech-to-text, large-v3 on cuda is the prod target
    whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    # async client kept alive across requests, 120s timeout covers slow first-token from a cold Gemma load
    ollama_client = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=120.0)

    yield

    # clean shutdown so the underlying http connections are released
    await ollama_client.aclose()


app = FastAPI(title="medtranslate", lifespan=lifespan)

# the whole frontend folder is served under /static so index.html can reference assets like /static/logo.png
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# unicode ranges per target script, used to strip characters from other writing systems
# that Gemma sometimes injects when its training is thin (e.g. Hebrew "בדיקות" appearing
# in Armenian medical translations because Gemma knows the Hebrew word better)
SCRIPT_RANGES = {
    "hy": [(0x0530, 0x058F), (0xFB13, 0xFB17)],  # Armenian
    "zh": [(0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF)],  # CJK + punctuation
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFE70, 0xFEFF)],  # Arabic
}
# ASCII punctuation, common shared punctuation, digits, whitespace are kept across all scripts
_ALWAYS_KEEP = set(" \t\n\r.,;:!?¿¡«»\"'()[]{}…-—–/0123456789")


def strip_foreign_script(text: str, target_lang: str) -> str:
    ranges = SCRIPT_RANGES.get(target_lang)
    if not ranges:
        return text
    out = []
    for char in text:
        if char in _ALWAYS_KEEP:
            out.append(char)
            continue
        cp = ord(char)
        if any(lo <= cp <= hi for lo, hi in ranges):
            out.append(char)
    cleaned = "".join(out)
    # collapse multiple spaces left behind after stripping
    return " ".join(cleaned.split())


async def translate(text: str, direction: str) -> str:
    # all translation goes through Gemma 4 via Ollama
    # uses /api/chat (not /api/generate) because Gemma 4 has a thinking mode that
    # otherwise eats the response into a "thinking" field and returns empty content
    # think=False skips the chain-of-thought for lower latency, temperature 0 for determinism
    source_lang, target_lang = direction.split("-")
    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": GEMMA_TRANSLATE_SYSTEM.format(
                src=LANG_NAME.get(source_lang, source_lang),
                tgt=LANG_NAME.get(target_lang, target_lang))},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    response = await ollama_client.post("/api/chat", json=body)
    response.raise_for_status()
    output = response.json()["message"]["content"].strip()
    # post-process to strip foreign-script characters Gemma sometimes injects
    return strip_foreign_script(output, target_lang)


@app.get("/health")
async def health():
    # readiness probe, reports which models loaded and whether Ollama is actually reachable
    # useful for debugging deploy issues (e.g. backend up but container cannot reach Ollama service)
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
        # nvidia-smi missing on a cpu-only box, treat as no gpu
        pass

    # short timeout because /health should not hang if Ollama is down
    ollama_reachable = False
    if ollama_client is not None:
        try:
            response = await ollama_client.get("/api/tags", timeout=2.0)
            ollama_reachable = response.status_code == 200
        except Exception:
            pass

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "whisper_loaded": whisper_model is not None,
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "ollama_reachable": ollama_reachable,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    direction: str = Form(...)
):
    # the full pipeline, browser uploads a webm audio blob, we transcribe with Whisper
    # then translate the result, returns both source text and translation so the frontend
    # can render the conversation pair and pass the turn to /soap and /extract later
    # delete=False is required on Windows so we can re-open the file for Whisper after closing it
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # direction is "src-tgt", e.g. "en-es" or "zh-en", passed straight from the frontend
        src_lang = direction.split("-")[0]
        tgt_lang = direction.split("-")[1]
        # giving Whisper the language explicitly skips its auto-detect step and improves accuracy
        segments, info = whisper_model.transcribe(tmp_path, language=src_lang)
        source_text = " ".join([s.text.strip() for s in segments])
    finally:
        # always clean up the temp file, even if Whisper or translate raised
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
    # called when the user edits a source bubble and submits, re-runs translation
    # on the edited text without going through Whisper again
    translated_text = await translate(req.text, req.direction)
    return {"translated_text": translated_text}


class ConversationTurn(BaseModel):
    # one back-and-forth turn captured from the frontend, the conversation array sent
    # to /soap and /extract is a list of these in chronological order
    speaker: str  # "doctor" or "patient"
    source_text: str
    source_lang: str
    translated_text: str
    target_lang: str


class SoapRequest(BaseModel):
    # shared request shape for /soap and /extract, both consume the same visit transcript
    # chart_lang picks which side of each turn to use as the canonical clinical record
    conversation: list[ConversationTurn]
    chart_lang: str = "en"


@app.post("/soap")
async def soap_note(req: SoapRequest):
    # generates a clinical SOAP note from the visit transcript using Gemma
    # the doctor's typical pain point is typing the chart, this turns the conversation into a draft note
    # fallback to "en" if the frontend sent something we don't have a language name for
    chart_lang = req.chart_lang if req.chart_lang in LANG_NAME else "en"

    # build a flat transcript Gemma can read, preferring whichever side of the turn
    # is already in chart_lang so we are not asking Gemma to translate while it summarizes
    lines = []
    for turn in req.conversation:
        if turn.source_lang == chart_lang:
            text = turn.source_text
        elif turn.target_lang == chart_lang:
            text = turn.translated_text
        else:
            # neither side matches chart_lang (e.g. es<->zh visit, chart in en)
            # send both so Gemma has the original utterance plus the translation
            text = f"{turn.source_text} (translated: {turn.translated_text})"
        lines.append(f"{turn.speaker.upper()}: {text}")
    transcript = "\n".join(lines) if lines else "(empty visit)"

    # temperature 0.2 leaves a little room for varied phrasing in the prose sections
    # num_predict 1024 gives enough room for all four soap sections
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
    response = await ollama_client.post("/api/chat", json=body)
    response.raise_for_status()
    return {"soap_note": response.json()["message"]["content"].strip(), "chart_lang": chart_lang}


@app.post("/extract")
async def extract(req: SoapRequest):
    # pulls structured fields (chief complaint, symptoms, meds, allergies, vitals) out of the
    # visit transcript, returned as JSON so the frontend can render a live chart side panel
    # always normalized to english because the chart panel uses english labels
    chart_lang = "en"

    # same transcript-flattening logic as /soap (could be factored out, kept inline for clarity)
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

    # format="json" forces Ollama to constrain output to valid JSON, no markdown fences or prose
    # temperature 0 because the extraction should be deterministic, no creativity wanted
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
    response = await ollama_client.post("/api/chat", json=body)
    response.raise_for_status()
    body_text = response.json()["message"]["content"].strip()

    # Ollama returns the JSON as a string, parse it here so the frontend gets a real object
    # if Gemma somehow produces invalid JSON we surface the raw text rather than 500ing
    import json
    try:
        chart = json.loads(body_text)
    except json.JSONDecodeError:
        chart = {"raw": body_text, "parse_error": True}
    return chart


class TtsRequest(BaseModel):
    text: str
    lang: str  # ISO 639-1 code, e.g. "hy" for Armenian


@app.post("/tts")
async def tts(req: TtsRequest):
    # fallback text-to-speech for languages the browser's SpeechSynthesis has no voice for
    # uses eSpeak NG which supports ~100 languages including ones Microsoft does not ship
    # the voice is robotic but intelligible, runs locally with no internet required
    # writes the text to a utf-8 temp file and passes via -f instead of argv, otherwise
    # Windows mangles non-ASCII characters through cp1252 and eSpeak produces silence
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp_txt:
        tmp_txt.write(req.text)
        txt_path = tmp_txt.name
    try:
        subprocess.run(
            [ESPEAK_BIN, "-v", req.lang, "-w", wav_path, "-f", txt_path],
            check=True, capture_output=True
        )
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(wav_path)
        os.unlink(txt_path)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/")
async def root():
    # serves the single-page frontend, all other assets are picked up via /static
    return FileResponse("frontend/index.html")