"""
Sports Bangla Dubbing API

A minimal FastAPI application that:
1. accepts a video upload,
2. randomly selects up to 30 seconds from that video,
3. extracts English speech using Whisper,
4. translates the English transcript to Bangla using NLLB,
5. generates Bangla speech using gTTS,
6. overlays the generated speech on the selected video timeline, and
7. shows the dubbed video in the browser.

The app intentionally has no manual time-selection UI.
For every uploaded video, it randomly chooses one 30-second section.
If the uploaded video is shorter than 30 seconds, it processes the full video.
"""

from __future__ import annotations

import html
import json
import math
import random
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from pydub import AudioSegment
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------
# Application folders
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
WORK_DIR = BASE_DIR / "work"
OUTPUT_DIR = BASE_DIR / "outputs"

for directory in [UPLOAD_DIR, WORK_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(title="Sports English-to-Bangla Dubbing Demo")

# Expose final MP4 files through /outputs/<filename>.
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ---------------------------------------------------------------------
# Lazy-loaded ML models
# ---------------------------------------------------------------------
# Loading models globally at import time makes the server slow to start.
# These globals are filled the first time a video is processed.

_WHISPER_MODEL = None
_NLLB_TOKENIZER = None
_NLLB_MODEL = None


def get_whisper_model():
    """Load Whisper once and reuse it for later requests."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


def get_nllb_model():
    """Load NLLB once and reuse tokenizer/model for later requests."""
    global _NLLB_TOKENIZER, _NLLB_MODEL
    if _NLLB_TOKENIZER is None or _NLLB_MODEL is None:
        model_name = "facebook/nllb-200-distilled-600M"
        _NLLB_TOKENIZER = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _NLLB_MODEL = _NLLB_MODEL.to(device)
    return _NLLB_TOKENIZER, _NLLB_MODEL


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Run a shell command safely without invoking a shell.

    stdout and stderr are captured together so that errors are readable
    in FastAPI logs.
    """
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ffprobe_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    completed = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    return float(completed.stdout.strip())


def save_upload(upload: UploadFile, target: Path) -> None:
    """Save uploaded file to disk."""
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def randomly_select_30s_clip(input_video: Path, selected_video: Path) -> dict[str, float]:
    """
    Randomly select a 30-second region from the uploaded video.

    If the video is shorter than 30 seconds, the entire video is used.
    """
    duration = ffprobe_duration(input_video)
    clip_len = min(30.0, duration)
    start = random.uniform(0.0, max(0.0, duration - clip_len))
    end = start + clip_len

    run([
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-t", str(clip_len),
        "-i", str(input_video),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(selected_video),
    ])

    return {"duration": duration, "start": start, "end": end, "clip_len": clip_len}


def extract_audio(selected_video: Path, audio_wav: Path) -> None:
    """Extract 16 kHz mono WAV audio from the selected video clip."""
    run([
        "ffmpeg",
        "-y",
        "-i", str(selected_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_wav),
    ])


def transcribe_english(audio_wav: Path) -> list[dict[str, Any]]:
    """Use Whisper to transcribe English speech with timestamps."""
    model = get_whisper_model()
    result = model.transcribe(str(audio_wav), language="en", fp16=False)
    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("No speech detected by Whisper.")
    return segments


def translate_segments_to_bangla(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate each Whisper segment from English to Bangla using NLLB."""
    tokenizer, model = get_nllb_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bangla_segments = []

    for seg in segments:
        english = seg["text"].strip()
        if not english:
            bangla = "নীরবতা"
        else:
            inputs = tokenizer(english, return_tensors="pt", truncation=True).to(device)
            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("ben_Beng"),
                max_length=128,
            )
            bangla = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
            if not bangla:
                bangla = "নীরবতা"

        bangla_segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "english": english,
            "bangla": bangla,
        })

    return bangla_segments


def generate_tts_chunks(bangla_segments: list[dict[str, Any]], tts_dir: Path) -> list[Path]:
    """Generate one Bangla WAV TTS file for each translated segment."""
    tts_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    for i, seg in enumerate(bangla_segments):
        text = seg.get("bangla", "").strip() or "নীরবতা"

        mp3_path = tts_dir / f"chunk_{i:03d}.mp3"
        wav_path = tts_dir / f"chunk_{i:03d}.wav"

        gTTS(text=text, lang="bn").save(str(mp3_path))

        run([
            "ffmpeg",
            "-y",
            "-i", str(mp3_path),
            "-ac", "1",
            "-ar", "16000",
            str(wav_path),
        ])

        chunk_paths.append(wav_path)

    return chunk_paths


def overlay_tts_on_timeline(
    selected_video: Path,
    bangla_segments: list[dict[str, Any]],
    chunk_paths: list[Path],
    dub_wav: Path,
) -> None:
    """
    Place each TTS chunk at its original Whisper timestamp.

    If a generated TTS chunk is too long, it is clipped.
    If it is too short, silence is appended.
    """
    clip_duration = ffprobe_duration(selected_video)

    base = AudioSegment.silent(
        duration=int(math.ceil(clip_duration * 1000)),
        frame_rate=16000,
    ).set_channels(1)

    for seg, chunk_path in zip(bangla_segments, chunk_paths):
        audio = AudioSegment.from_wav(chunk_path).set_frame_rate(16000).set_channels(1)

        start_ms = int(float(seg["start"]) * 1000)
        target_ms = max(300, int((float(seg["end"]) - float(seg["start"])) * 1000))

        if len(audio) > target_ms:
            audio = audio[:target_ms]
        else:
            audio += AudioSegment.silent(duration=target_ms - len(audio), frame_rate=16000)

        base = base.overlay(audio, position=start_ms)

    base.export(dub_wav, format="wav")


def mux_video_and_dubbed_audio(selected_video: Path, dub_wav: Path, final_video: Path) -> None:
    """Combine the selected video stream with the generated Bangla audio."""
    run([
        "ffmpeg",
        "-y",
        "-i", str(selected_video),
        "-i", str(dub_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_video),
    ])


def process_video(upload_path: Path, job_dir: Path, final_video: Path) -> dict[str, Any]:
    """Run the complete video-to-Bangla-dubbed-video pipeline."""
    selected_video = job_dir / "selected_30s_clip.mp4"
    audio_wav = job_dir / "selected_audio_16k_mono.wav"
    dub_wav = job_dir / "bangla_dub.wav"
    tts_dir = job_dir / "tts_chunks"

    clip_info = randomly_select_30s_clip(upload_path, selected_video)
    extract_audio(selected_video, audio_wav)

    segments = transcribe_english(audio_wav)
    bangla_segments = translate_segments_to_bangla(segments)

    # Save intermediate text for debugging and reproducibility.
    (job_dir / "segments.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")
    (job_dir / "translated_segments.json").write_text(
        json.dumps(bangla_segments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    chunk_paths = generate_tts_chunks(bangla_segments, tts_dir)
    overlay_tts_on_timeline(selected_video, bangla_segments, chunk_paths, dub_wav)
    mux_video_and_dubbed_audio(selected_video, dub_wav, final_video)

    return {
        **clip_info,
        "segments": bangla_segments,
        "final_video": str(final_video),
    }


# ---------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Simple upload page."""
    return """
    <!doctype html>
    <html>
    <head>
        <title>Bangla Sports Dubbing Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; }
            .box { border: 1px solid #ddd; padding: 24px; border-radius: 10px; }
            button { padding: 10px 18px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h2>Sports English-to-Bangla Dubbing Demo</h2>
        <div class="box">
            <p><b>Limit:</b> this demo randomly processes only 30 seconds of the uploaded video.</p>
            <form action="/dub" method="post" enctype="multipart/form-data">
                <p><input type="file" name="video" accept="video/*" required></p>
                <p><button type="submit">Upload and Dub</button></p>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/dub", response_class=HTMLResponse)
def dub(video: UploadFile = File(...)) -> str:
    """
    Receive one uploaded video, process a random 30-second clip,
    and show the dubbed output in the browser.
    """
    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(video.filename or "input_video.mp4").name
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    final_video = OUTPUT_DIR / f"{job_id}_bangla_dubbed_30s_demo.mp4"

    save_upload(video, upload_path)
    info = process_video(upload_path, job_dir, final_video)

    video_url = f"/outputs/{final_video.name}"
    transcript_rows = "\n".join(
        f"<tr><td>{s['start']:.2f}-{s['end']:.2f}</td>"
        f"<td>{html.escape(s['english'])}</td>"
        f"<td>{html.escape(s['bangla'])}</td></tr>"
        for s in info["segments"]
    )

    return f"""
    <!doctype html>
    <html>
    <head>
        <title>Dubbed Output</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 30px auto; }}
            video {{ width: 100%; max-width: 860px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            td, th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
            th {{ background: #f3f3f3; }}
        </style>
    </head>
    <body>
        <h2>Bangla Dubbed Video</h2>
        <p>
            Random section processed:
            <b>{info['start']:.2f}s</b> to <b>{info['end']:.2f}s</b>
            from uploaded video.
        </p>

        <video controls src="{video_url}"></video>

        <p><a href="{video_url}" download>Download video</a></p>
        <p><a href="/">Process another video</a></p>

        <h3>Transcript</h3>
        <table>
            <tr><th>Time</th><th>English</th><th>Bangla</th></tr>
            {transcript_rows}
        </table>
    </body>
    </html>
    """


@app.get("/download/{filename}")
def download(filename: str) -> FileResponse:
    """Optional direct download endpoint."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    return FileResponse(path, media_type="video/mp4", filename=filename)
