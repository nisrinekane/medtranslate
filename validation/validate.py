"""
MedTranslate Validation Script
Runs the 40-sentence medical test set and measures:
  1. STT accuracy  — Word Error Rate (WER) per language and overall
  2. Translation    — BLEU score per direction and overall
  3. End-to-end     — BLEU on audio→transcribe→translate vs reference translation
"""

import json
import os
import sys
import time
from pathlib import Path

import jiwer
import sacrebleu
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset.json"
AUDIO_DIR = SCRIPT_DIR / "audio"

# Change to "medium" or "large-v3" for better accuracy
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float32")


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def audio_path_for(entry):
    lang = "en" if entry["direction"] == "en-es" else "es"
    return AUDIO_DIR / f"{entry['id']:02d}_{lang}.wav"


def load_models():
    print(f"Loading Whisper ({WHISPER_SIZE}) on {DEVICE}...")
    whisper = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print("Loading MarianMT en→es...")
    tok_en_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model_en_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    print("Loading MarianMT es→en...")
    tok_es_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model_es_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    return whisper, tok_en_es, model_en_es, tok_es_en, model_es_en


def transcribe(whisper, audio_file, language):
    segments, _ = whisper.transcribe(str(audio_file), language=language)
    return " ".join(s.text.strip() for s in segments)


def translate(text, direction, tok_en_es, model_en_es, tok_es_en, model_es_en):
    if direction == "en-es":
        tokens = tok_en_es(text, return_tensors="pt", padding=True)
        out = model_en_es.generate(**tokens)
        return tok_en_es.decode(out[0], skip_special_tokens=True)
    else:
        tokens = tok_es_en(text, return_tensors="pt", padding=True)
        out = model_es_en.generate(**tokens)
        return tok_es_en.decode(out[0], skip_special_tokens=True)


def compute_wer(references, hypotheses):
    return jiwer.wer(references, hypotheses)


def compute_bleu(references, hypotheses):
    return sacrebleu.corpus_bleu(hypotheses, [references])


def print_separator():
    print("=" * 70)


def run():
    dataset = load_dataset()

    # Check audio files exist
    missing = [e for e in dataset if not audio_path_for(e).exists()]
    if missing:
        print(f"ERROR: Missing audio for {len(missing)} sentences.")
        print("Run generate_audio.sh first.")
        for e in missing:
            print(f"  - {audio_path_for(e)}")
        sys.exit(1)

    whisper, tok_en_es, model_en_es, tok_es_en, model_es_en = load_models()
    print_separator()
    print("Running validation on 40 sentences...\n")

    # Accumulators
    stt_refs_en, stt_hyps_en = [], []
    stt_refs_es, stt_hyps_es = [], []
    trans_refs_en_es, trans_hyps_en_es = [], []
    trans_refs_es_en, trans_hyps_es_en = [], []
    e2e_refs_en_es, e2e_hyps_en_es = [], []
    e2e_refs_es_en, e2e_hyps_es_en = [], []

    results = []

    for entry in dataset:
        eid = entry["id"]
        direction = entry["direction"]
        source_text = entry["source"]
        ref_translation = entry["reference"]
        audio_file = audio_path_for(entry)
        src_lang = "en" if direction == "en-es" else "es"

        # --- STT ---
        transcription = transcribe(whisper, audio_file, src_lang)

        # --- Translation (text-only, no STT) ---
        translation_from_text = translate(
            source_text, direction,
            tok_en_es, model_en_es, tok_es_en, model_es_en
        )

        # --- End-to-end (STT + translation) ---
        translation_from_audio = translate(
            transcription, direction,
            tok_en_es, model_en_es, tok_es_en, model_es_en
        )

        # Collect STT metrics
        if src_lang == "en":
            stt_refs_en.append(source_text)
            stt_hyps_en.append(transcription)
        else:
            stt_refs_es.append(source_text)
            stt_hyps_es.append(transcription)

        # Collect translation metrics
        if direction == "en-es":
            trans_refs_en_es.append(ref_translation)
            trans_hyps_en_es.append(translation_from_text)
            e2e_refs_en_es.append(ref_translation)
            e2e_hyps_en_es.append(translation_from_audio)
        else:
            trans_refs_es_en.append(ref_translation)
            trans_hyps_es_en.append(translation_from_text)
            e2e_refs_es_en.append(ref_translation)
            e2e_hyps_es_en.append(translation_from_audio)

        results.append({
            "id": eid,
            "direction": direction,
            "source": source_text,
            "transcription": transcription,
            "translation_from_text": translation_from_text,
            "translation_from_audio": translation_from_audio,
            "reference": ref_translation,
        })

        # Progress
        status = "OK" if transcription.strip() else "EMPTY"
        print(f"  [{eid:02d}/{len(dataset)}] {direction} — STT: {status}")

    # --- Compute metrics ---
    print_separator()
    print("\n📊 RESULTS\n")

    # STT WER
    print("── Speech-to-Text (WER) ─────────────────────────────────")
    wer_en = compute_wer(stt_refs_en, stt_hyps_en)
    wer_es = compute_wer(stt_refs_es, stt_hyps_es)
    wer_all = compute_wer(stt_refs_en + stt_refs_es, stt_hyps_en + stt_hyps_es)
    print(f"  English WER:  {wer_en:.1%}  ({len(stt_refs_en)} sentences)")
    print(f"  Spanish WER:  {wer_es:.1%}  ({len(stt_refs_es)} sentences)")
    print(f"  Overall WER:  {wer_all:.1%}  (40 sentences)")
    print()

    # Translation BLEU (text-only)
    print("── Translation BLEU (text → text, no STT) ───────────────")
    bleu_en_es = compute_bleu(trans_refs_en_es, trans_hyps_en_es)
    bleu_es_en = compute_bleu(trans_refs_es_en, trans_hyps_es_en)
    bleu_trans_all = compute_bleu(
        trans_refs_en_es + trans_refs_es_en,
        trans_hyps_en_es + trans_hyps_es_en
    )
    print(f"  EN→ES BLEU:   {bleu_en_es.score:.1f}  ({len(trans_refs_en_es)} sentences)")
    print(f"  ES→EN BLEU:   {bleu_es_en.score:.1f}  ({len(trans_refs_es_en)} sentences)")
    print(f"  Overall BLEU: {bleu_trans_all.score:.1f}  (40 sentences)")
    print()

    # End-to-end BLEU (audio → STT → translation)
    print("── End-to-End BLEU (audio → transcribe → translate) ─────")
    e2e_bleu_en_es = compute_bleu(e2e_refs_en_es, e2e_hyps_en_es)
    e2e_bleu_es_en = compute_bleu(e2e_refs_es_en, e2e_hyps_es_en)
    e2e_bleu_all = compute_bleu(
        e2e_refs_en_es + e2e_refs_es_en,
        e2e_hyps_en_es + e2e_hyps_es_en
    )
    print(f"  EN→ES BLEU:   {e2e_bleu_en_es.score:.1f}  ({len(e2e_refs_en_es)} sentences)")
    print(f"  ES→EN BLEU:   {e2e_bleu_es_en.score:.1f}  ({len(e2e_refs_es_en)} sentences)")
    print(f"  Overall BLEU: {e2e_bleu_all.score:.1f}  (40 sentences)")
    print()

    # Summary
    print_separator()
    print("\n📋 SUMMARY")
    print(f"  STT WER:           {wer_all:.1%}   {'✓' if wer_all < 0.10 else '⚠ high'}")
    print(f"  Translation BLEU:  {bleu_trans_all.score:.1f}    {'✓' if bleu_trans_all.score >= 40 else '⚠ below 40'}")
    print(f"  End-to-End BLEU:   {e2e_bleu_all.score:.1f}    {'✓' if e2e_bleu_all.score >= 30 else '⚠ below 30'}")
    print()

    # Save detailed results
    output_path = SCRIPT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "whisper_model": WHISPER_SIZE,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
            },
            "metrics": {
                "stt_wer_en": round(wer_en, 4),
                "stt_wer_es": round(wer_es, 4),
                "stt_wer_overall": round(wer_all, 4),
                "bleu_en_es": round(bleu_en_es.score, 2),
                "bleu_es_en": round(bleu_es_en.score, 2),
                "bleu_overall": round(bleu_trans_all.score, 2),
                "e2e_bleu_en_es": round(e2e_bleu_en_es.score, 2),
                "e2e_bleu_es_en": round(e2e_bleu_es_en.score, 2),
                "e2e_bleu_overall": round(e2e_bleu_all.score, 2),
            },
            "details": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Detailed results saved to {output_path}\n")


if __name__ == "__main__":
    run()
