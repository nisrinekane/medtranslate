#!/bin/bash
# Generate WAV audio files for each sentence in the dataset using macOS say
# English voice: Samantha, Spanish voice: Paulina (Mexican Spanish)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIO_DIR="$SCRIPT_DIR/audio"
mkdir -p "$AUDIO_DIR"

echo "Generating audio files..."

# English sentences (en-es direction) — source is English
say -v Samantha -o "$AUDIO_DIR/01_en.aiff" "I have pain here."
say -v Paulina  -o "$AUDIO_DIR/02_es.aiff" "Me siento mareado."
say -v Samantha -o "$AUDIO_DIR/03_en.aiff" "I cannot breathe well."
say -v Paulina  -o "$AUDIO_DIR/04_es.aiff" "Tengo náuseas."
say -v Samantha -o "$AUDIO_DIR/05_en.aiff" "I have a fever."
say -v Paulina  -o "$AUDIO_DIR/06_es.aiff" "Estoy muy cansado."
say -v Samantha -o "$AUDIO_DIR/07_en.aiff" "I am pregnant."
say -v Paulina  -o "$AUDIO_DIR/08_es.aiff" "Me caí."
say -v Samantha -o "$AUDIO_DIR/09_en.aiff" "I have diabetes."
say -v Paulina  -o "$AUDIO_DIR/10_es.aiff" "Tengo la presión alta."
say -v Samantha -o "$AUDIO_DIR/11_en.aiff" "I have asthma."
say -v Paulina  -o "$AUDIO_DIR/12_es.aiff" "Soy alérgico a la penicilina."
say -v Samantha -o "$AUDIO_DIR/13_en.aiff" "I was vomiting."
say -v Paulina  -o "$AUDIO_DIR/14_es.aiff" "Tengo diarrea."
say -v Samantha -o "$AUDIO_DIR/15_en.aiff" "I cannot sleep."
say -v Paulina  -o "$AUDIO_DIR/16_es.aiff" "No comí hoy."
say -v Samantha -o "$AUDIO_DIR/17_en.aiff" "I need my medication."
say -v Paulina  -o "$AUDIO_DIR/18_es.aiff" "Esto empezó hoy."
say -v Samantha -o "$AUDIO_DIR/19_en.aiff" "It is getting worse."
say -v Paulina  -o "$AUDIO_DIR/20_es.aiff" "No entiendo."
say -v Samantha -o "$AUDIO_DIR/21_en.aiff" "The pain started three days ago and it has not stopped since then."
say -v Paulina  -o "$AUDIO_DIR/22_es.aiff" "He estado tomando medicamento para la presión durante cinco años pero se me acabó la semana pasada."
say -v Samantha -o "$AUDIO_DIR/23_en.aiff" "I feel a sharp pain on my right side every time I take a deep breath."
say -v Paulina  -o "$AUDIO_DIR/24_es.aiff" "Me operaron la rodilla hace dos años y ahora está hinchada otra vez."
say -v Samantha -o "$AUDIO_DIR/25_en.aiff" "I have not been able to eat or drink anything since yesterday morning."
say -v Paulina  -o "$AUDIO_DIR/26_es.aiff" "Me siento muy mareado cuando me levanto y casi me caigo esta mañana."
say -v Samantha -o "$AUDIO_DIR/27_en.aiff" "My chest feels tight and I have been short of breath since last night."
say -v Paulina  -o "$AUDIO_DIR/28_es.aiff" "Tomo tres medicamentos diferentes cada día pero no sé cómo se llaman."
say -v Samantha -o "$AUDIO_DIR/29_en.aiff" "The pain comes and goes but in the last two days it has been constant."
say -v Paulina  -o "$AUDIO_DIR/30_es.aiff" "Me he sentido débil y cansado durante dos semanas y está empeorando."
say -v Samantha -o "$AUDIO_DIR/31_en.aiff" "I am allergic to penicillin and the last time I took it I had a bad reaction."
say -v Paulina  -o "$AUDIO_DIR/32_es.aiff" "He tenido este dolor antes pero esta vez se siente diferente y más fuerte."
say -v Samantha -o "$AUDIO_DIR/33_en.aiff" "I did not take my medication this morning because I ran out three days ago."
say -v Paulina  -o "$AUDIO_DIR/34_es.aiff" "Me ha dolido el estómago desde que comí anoche y vomité dos veces."
say -v Samantha -o "$AUDIO_DIR/35_en.aiff" "I have diabetes and I think my sugar is very high right now because I feel strange."
say -v Paulina  -o "$AUDIO_DIR/36_es.aiff" "Me caí en casa esta mañana y desde entonces me ha dolido y se me ha hinchado la muñeca."
say -v Samantha -o "$AUDIO_DIR/37_en.aiff" "I have been having headaches every day for two weeks and the pain is very strong."
say -v Paulina  -o "$AUDIO_DIR/38_es.aiff" "No puedo caminar bien porque mi pierna izquierda ha estado dormida desde que me desperté esta mañana."
say -v Samantha -o "$AUDIO_DIR/39_en.aiff" "I have been coughing for ten days and last night I coughed up some blood."
say -v Paulina  -o "$AUDIO_DIR/40_es.aiff" "Necesito a alguien que hable español porque no puedo explicar todo en inglés."

# Convert all AIFF to WAV (16-bit PCM, 16kHz mono — ideal for Whisper)
echo "Converting to WAV (16kHz mono)..."
for f in "$AUDIO_DIR"/*.aiff; do
  wav="${f%.aiff}.wav"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$f" "$wav"
  rm "$f"
done

echo "Done. Generated $(ls "$AUDIO_DIR"/*.wav | wc -l | tr -d ' ') audio files in $AUDIO_DIR/"
