"""Transcribe an audio file into word-level timestamps.

    python attention_insights/transcribe.py talk.wav [words.json] [--model small.en]

Uses faster-whisper (no ffmpeg needed). Output JSON:
    {"duration": float, "segments": [{"start", "end", "text", "words": [{"start", "end", "word"}]}]}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def load_audio(path: str, rate: int = 16000) -> np.ndarray:
    """Read wav/aiff/flac/ogg with soundfile, downmix to mono, resample to 16 kHz."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != rate:
        n = int(round(len(x) * rate / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def transcribe(audio: str, model_name: str = "small.en") -> dict:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(load_audio(audio), word_timestamps=True, vad_filter=True)
    out = []
    for seg in segments:
        words = [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                 for w in (seg.words or [])]
        out.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                    "text": seg.text.strip(), "words": words})
        print(f"[{seg.start:6.1f} – {seg.end:6.1f}] {seg.text.strip()}")
    return {"duration": round(info.duration, 2), "segments": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()
    out = Path(a.out or Path(a.audio).with_suffix(".words.json"))
    data = transcribe(a.audio, a.model)
    out.write_text(json.dumps(data, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
