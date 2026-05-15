"""
Head-to-head benchmark: Gemma 4 (via Ollama) vs MarianMT on 40 medical phrases

Reads validation/dataset-40.csv (en-es and es-en directions with reference translations),
runs both models, prints per-row diff, and reports BLEU per direction per model

Usage:
    python validation/score_compare.py [csv_path]
"""
import csv
import sys
import time
from pathlib import Path

import httpx
from sacrebleu.metrics import BLEU
from transformers import MarianMTModel, MarianTokenizer

sys.stdout.reconfigure(encoding="utf-8")  # Mandarin / accented chars on Windows

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
GEMMA_MODEL = "gemma4:e2b"

LANG_NAME = {"en": "English", "es": "Spanish", "zh": "Mandarin Chinese"}

GEMMA_SYSTEM = (
    "You are a professional medical interpreter for a clinical setting. "
    "Translate the user's {src} into {tgt}. "
    "Use formal address (e.g. 'usted' / 'le' for Spanish). "
    "Output ONLY the translated sentence — no commentary, no quotes, no labels, no explanation."
)


def gemma_translate(text: str, direction: str) -> str:
    src_lang, tgt_lang = direction.split("-")
    body = {
        "model": GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": GEMMA_SYSTEM.format(src=LANG_NAME[src_lang], tgt=LANG_NAME[tgt_lang])},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    r = httpx.post(OLLAMA_URL, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def marian_translate_factory():
    print("Loading MarianMT en->es...")
    tok_en_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model_en_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    print("Loading MarianMT es->en...")
    tok_es_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model_es_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    def translate(text: str, direction: str) -> str:
        if direction == "en-es":
            tok, model = tok_en_es, model_en_es
        else:
            tok, model = tok_es_en, model_es_en
        tokens = tok(text, return_tensors="pt", padding=True)
        out = model.generate(**tokens)
        return tok.decode(out[0], skip_special_tokens=True)

    return translate


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation/dataset-40.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    print(f"Loaded {len(rows)} phrases from {csv_path}\n")

    marian = marian_translate_factory()
    print()

    # accumulators per direction
    refs = {"en-es": [], "es-en": []}
    marian_hyps = {"en-es": [], "es-en": []}
    gemma_hyps = {"en-es": [], "es-en": []}
    marian_total_s = 0.0
    gemma_total_s = 0.0

    for i, row in enumerate(rows, 1):
        direction = row["direction"]
        source = row["source"]
        ref = row["reference_translation"]

        t0 = time.time()
        m_out = marian(source, direction)
        marian_total_s += time.time() - t0

        t0 = time.time()
        try:
            g_out = gemma_translate(source, direction)
        except Exception as e:
            g_out = f"<ERROR: {e}>"
        gemma_total_s += time.time() - t0

        refs[direction].append(ref)
        marian_hyps[direction].append(m_out)
        gemma_hyps[direction].append(g_out)

        print(f"[{i:02d}] {direction}  src: {source}")
        print(f"     ref:    {ref}")
        print(f"     marian: {m_out}")
        print(f"     gemma:  {g_out}")
        print()

    bleu = BLEU()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    for direction in ("en-es", "es-en"):
        if not refs[direction]:
            continue
        m_score = bleu.corpus_score(marian_hyps[direction], [refs[direction]])
        g_score = bleu.corpus_score(gemma_hyps[direction], [refs[direction]])
        print(f"\n{direction}  ({len(refs[direction])} phrases)")
        print(f"  MarianMT:  {m_score}")
        print(f"  Gemma 4:   {g_score}")

    all_refs = refs["en-es"] + refs["es-en"]
    all_marian = marian_hyps["en-es"] + marian_hyps["es-en"]
    all_gemma = gemma_hyps["en-es"] + gemma_hyps["es-en"]
    m_overall = bleu.corpus_score(all_marian, [all_refs])
    g_overall = bleu.corpus_score(all_gemma, [all_refs])
    print(f"\nOVERALL  ({len(all_refs)} phrases)")
    print(f"  MarianMT:  {m_overall}")
    print(f"  Gemma 4:   {g_overall}")

    print(f"\nLatency (sum of all calls):")
    print(f"  MarianMT: {marian_total_s:6.1f}s  ({marian_total_s/len(rows)*1000:.0f} ms/phrase)")
    print(f"  Gemma 4:  {gemma_total_s:6.1f}s  ({gemma_total_s/len(rows)*1000:.0f} ms/phrase)")


if __name__ == "__main__":
    main()
