# Sports English-to-Bangla Dubbing Demo API

A minimal FastAPI app that accepts a video upload, randomly selects up to 30 seconds, transcribes English speech with Whisper, translates it to Bangla using NLLB, generates Bangla speech with gTTS, and shows the dubbed video in the browser.

## What the app does

1. User uploads one video.
2. The app randomly selects a 30-second section.
3. Audio is extracted at 16 kHz mono.
4. Whisper transcribes English speech with timestamps.
5. NLLB translates each segment to Bangla.
6. gTTS generates Bangla speech chunks.
7. The speech chunks are placed back on the timeline.
8. The dubbed audio is muxed with the video.
9. The browser displays the final dubbed video.

## Demo limitation

This demo processes only 30 seconds per uploaded video.

If the uploaded video is shorter than 30 seconds, the full video is processed.

## System requirement

`ffmpeg` must be installed.

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

The first request is slow because Whisper and NLLB models are downloaded and loaded.

For a smoother demo, run one test upload before the actual presentation.
