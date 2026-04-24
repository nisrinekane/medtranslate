# medtranslate

Translation system(speech-to-text + EN↔ES translation).

## Prerequisites

For Linux machines using Google Chrome browser, you need to install the following voice packs to get speech-to-text to work, otherwise it will silently fail:

```bash
# Option 1: basic voices (smaller download)
sudo apt install speech-dispatcher espeak-ng espeak-ng-data

# Option 2: better quality voices (larger download):
sudo apt install libespeak-ng1 festival festvox-kallpc16k
```

## Run with Docker

Requires Docker and an NVIDIA GPU with drivers installed

```bash
docker compose up
```

Open **http://localhost:8080** in your browser
