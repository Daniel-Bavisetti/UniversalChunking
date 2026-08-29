"""Generate a deterministic fixture video with KNOWN ground-truth boundaries.

Engineered to test the fusion claim (plan sec.18). The fixture contains:

  t=24  VISUAL-ONLY boundary   - scene colour changes, topic/vocabulary continues.
                                 Audio-only chunking (config B) MUST miss this.
  t=48  SEMANTIC-ONLY boundary - topic/vocabulary changes, scene stays identical.
                                 Visual-only detection MUST miss this.
  t=72  BOTH change            - the easy case; every config should find it.

Speech is real (Windows SAPI text-to-speech) so faster-whisper produces genuine
words and timestamps rather than us faking the ASR stage.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BUILD = DATA / "_fixture_build"

FPS = 12
W, H = 854, 480
# Pause AFTER each segment. Deliberately uneven: a pause at the visual-only or
# semantic-only boundary would hand it to the audio-only config for free, so only
# the "both" boundary gets one.
PAUSES = [0.0, 0.0, 1.5, 1.5]
SAMPLE_RATE = 16000

# BGR (OpenCV channel order)
BLUE = (120, 60, 20)
RED = (40, 40, 140)
GREEN = (60, 120, 40)

SEGMENTS = [
    dict(
        idx=0,
        topic="auth",
        colour=BLUE,
        heading="AUTHENTICATION SETUP",
        bullets=["Username and password", "Session tokens", "Sign in flow"],
        text=(
            "Let's start by configuring authentication for the application. "
            "The user signs in with a username and a password. "
            "Once those credentials are verified we issue a session token. "
            "That token identifies the user on every later request. "
            "The password is never stored directly, only a hash of the password. "
            "A user account can also require a second authentication factor."
        ),
    ),
    dict(
        # VISUAL changes here (t=24). Topic and vocabulary deliberately continue.
        idx=1,
        topic="auth",
        colour=RED,
        heading="AUTHENTICATION SETUP",
        bullets=["Token expiry", "Refresh credentials", "Password reset"],
        text=(
            "The session token expires after a while, so the user has to sign in again. "
            "We can refresh the credentials silently before that token expires. "
            "If the user forgets the password, the reset flow sends a new login link. "
            "That keeps authentication simple for the user. "
            "A refresh token lets the user stay signed in without a new password. "
            "When the user signs out we revoke the session token immediately."
        ),
    ),
    dict(
        # TOPIC changes here (t=48). Visual is deliberately identical to segment 1.
        idx=2,
        topic="db",
        colour=RED,
        heading="AUTHENTICATION SETUP",
        bullets=["Token expiry", "Refresh credentials", "Password reset"],
        text=(
            "Now we move on to the database migration. "
            "The migration adds a new table and a new column to the existing schema. "
            "We create an index on that column so the query runs quickly. "
            "Every migration is applied in order against the database. "
            "A rollback script reverses the schema change if a migration fails. "
            "The query planner uses that index to avoid a full table scan."
        ),
    ),
    dict(
        # BOTH change here (t=72).
        idx=3,
        topic="deploy",
        colour=GREEN,
        heading="DEPLOYMENT",
        bullets=["Build container", "Rollout to cluster", "Health checks"],
        text=(
            "Finally we can talk about deployment. "
            "The build produces a container image which we push to the registry. "
            "The cluster performs a rolling release across every server. "
            "Health checks confirm the deployment succeeded before we finish the rollout. "
            "If a health check fails the cluster stops the rollout and rolls back. "
            "Each server pulls the new container image from that registry."
        ),
    ),
]

KIND_BY_INDEX = {0: "visual_only", 1: "semantic_only", 2: "both"}
NOTE_BY_KIND = {
    "visual_only": "scene colour changes; vocabulary continues -> audio-only must miss it",
    "semantic_only": "vocabulary changes; scene identical -> visual-only must miss it",
    "both": "both modalities change -> every config should find it",
}


def ground_truth(durations: list[float]) -> dict:
    edges, acc = [], 0.0
    for d in durations[:-1]:
        acc += d
        edges.append(round(acc, 2))
    return {
        "video": "fixture.mp4",
        "duration": round(sum(durations), 2),
        "boundaries": edges,
        "kinds": {str(b): KIND_BY_INDEX[i] for i, b in enumerate(edges)},
        "notes": {str(b): NOTE_BY_KIND[KIND_BY_INDEX[i]] for i, b in enumerate(edges)},
    }


def synth_speech(text: str, out_wav: Path, txt_path: Path) -> None:
    """Windows SAPI text-to-speech -> 16 kHz mono 16-bit WAV.

    The text goes through a file rather than being interpolated into the
    PowerShell source, which keeps quoting out of the picture entirely.
    """
    txt_path.write_text(text, encoding="utf-8")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$t = Get-Content -Raw -Encoding UTF8 -LiteralPath $env:VKE_TEXT; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        "$s.Rate = 0; "
        "$s.SetOutputToWaveFile($env:VKE_WAV, $fmt); "
        "$s.Speak($t); "
        "$s.Dispose()"
    )
    env = {"VKE_TEXT": str(txt_path), "VKE_WAV": str(out_wav)}
    import os

    proc_env = {**os.environ, **env}
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True,
        text=True,
        env=proc_env,
    )
    if result.returncode != 0 or not out_wav.exists():
        raise RuntimeError(
            f"SAPI synthesis failed (rc={result.returncode}):\n{result.stderr[:500]}"
        )


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit PCM, got {w.getsampwidth()*8}-bit")
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return data, rate


def build_audio(out_wav: Path) -> list[float]:
    """Synthesize each segment; return the per-segment durations.

    Segment length follows the speech rather than the speech being truncated to
    fit a fixed window. Truncation put a hard audio discontinuity exactly at each
    ground-truth boundary, which handed the boundary to the audio-only config for
    free and invalidated the comparison the fixture exists to make.
    """
    chunks = []
    durations: list[float] = []
    for seg in SEGMENTS:
        part = BUILD / f"seg{seg['idx']}.wav"
        txt = BUILD / f"seg{seg['idx']}.txt"
        synth_speech(seg["text"], part, txt)
        data, rate = read_wav(part)

        if rate != SAMPLE_RATE:  # safety net; SAPI should honour our format
            idx = (np.arange(int(len(data) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE)
            data = data[np.clip(idx.astype(int), 0, len(data) - 1)]

        pause = PAUSES[seg["idx"]]
        pad = int(pause * SAMPLE_RATE)
        chunks.append(np.concatenate([data, np.zeros(pad, dtype=np.int16)]))
        seconds = (len(data) + pad) / SAMPLE_RATE
        durations.append(seconds)
        print(f"  segment {seg['idx']} ({seg['topic']:6s}): "
              f"{len(data)/SAMPLE_RATE:5.1f}s speech + {pause:.1f}s pause "
              f"= {seconds:5.1f}s")

    full = np.concatenate(chunks)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(full.tobytes())
    return durations


def draw_frame(seg: dict, t_in_seg: float, seg_seconds: float) -> np.ndarray:
    img = np.full((H, W, 3), seg["colour"], dtype=np.uint8)
    # Vertical gradient so the histogram is not perfectly flat.
    grad = np.linspace(0, 28, H, dtype=np.uint8)[:, None, None]
    img = np.clip(img.astype(np.int16) + grad, 0, 255).astype(np.uint8)

    cv2.putText(img, seg["heading"], (48, 96), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (48, 118), (W - 48, 118), (235, 235, 235), 2)
    for i, bullet in enumerate(seg["bullets"]):
        cv2.putText(img, f"- {bullet}", (72, 180 + i * 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (240, 240, 240), 2, cv2.LINE_AA)

    # A moving marker keeps motion non-zero without changing the scene.
    x = int(72 + (W - 200) * (t_in_seg / max(seg_seconds, 1e-6)))
    cv2.circle(img, (x, H - 60), 13, (255, 255, 255), -1)
    return img


def build_video(silent_mp4: Path, durations: list[float]) -> None:
    writer = cv2.VideoWriter(
        str(silent_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    try:
        for seg, seconds in zip(SEGMENTS, durations):
            for frame_i in range(int(round(seconds * FPS))):
                writer.write(draw_frame(seg, frame_i / FPS, seconds))
    finally:
        writer.release()


def mux(silent_mp4: Path, wav: Path, out: Path) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-i", str(silent_mp4), "-i", str(wav),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "96k", "-shortest", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[:800]}")


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    out = DATA / "fixture.mp4"

    print("[1/4] synthesizing speech (Windows SAPI)...")
    wav = BUILD / "audio.wav"
    durations = build_audio(wav)

    print("[2/4] rendering frames...")
    silent = BUILD / "silent.mp4"
    build_video(silent, durations)

    print("[3/4] muxing audio + video...")
    mux(silent, wav, out)

    print("[4/4] writing ground truth...")
    gt = ground_truth(durations)
    gt_path = DATA / "fixture_ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"\nOK  {out}  ({out.stat().st_size/1e6:.1f} MB, "
          f"{gt['duration']:.0f}s)")
    print(f"OK  {gt_path}")
    print("\nground-truth boundaries:")
    for b in gt["boundaries"]:
        print(f"  {b:6.1f}s  {gt['kinds'][str(b)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
