"""
Record the fixed STT test set (Phase 3 step 1) ONCE, in sir's own voice.

Every STT candidate is benchmarked on exactly these files, so the results are reproducible.
Output: data/testset/NN.wav (16 kHz, mono, 16-bit) + data/testset/testset.json (expected text).
data/ is not in git: the recordings never leave this Mac.

Usage:
    python record_testset.py               record every phrase that has no WAV yet
    python record_testset.py --redo 08 12  record phrases 08 and 12 again
    python record_testset.py --list        show the phrases and which ones are recorded

Per phrase: Enter = start, Enter = stop, then Enter = keep, p = play back, r = record again.
Control+C stops; files already saved are kept, so just run the script again to continue.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000            # what every STT model expects
OUT_DIR = Path("data/testset")
INDEX_FILE = OUT_DIR / "testset.json"

MIN_SECONDS = 0.3               # shorter than this = probably pressed Enter twice by accident
QUIET_PEAK = 0.02               # loudest sample below this = (almost) silence
CLIP_PEAK = 0.99                # loudest sample above this = too loud, distorted

# (id, expected text, category, confirm label). The confirm label is used in steps 5 and 6.
PHRASES = [
    ("01", "Open Spotify.", "tool", None),
    ("02", "Pause the music.", "tool", None),
    ("03", "Next song, please.", "tool", None),
    ("04", "Set the volume to forty percent.", "tool", None),
    ("05", "Turn it up a bit.", "tool", None),
    ("06", "Quit Notes.", "tool", None),
    ("07", "What time is it?", "tool", None),
    ("08", "What's the weather in Buizingen tomorrow?", "tool", None),
    ("09", "What is eighteen percent of fifty?", "tool", None),
    ("10", "Search my files for thesis.", "tool", None),
    ("11", "Open the BBC website.", "tool", None),
    ("12", "Hello Jarvis, how are you today?", "chat", None),
    ("13", "Thank you, that's all.", "chat", None),
    ("14", "Yes.", "confirm", "yes"),
    ("15", "Yeah, go ahead.", "confirm", "yes"),
    ("16", "No.", "confirm", "no"),
    ("17", "No, don't do that.", "confirm", "no"),
    ("18", "Hmm, I'm not sure.", "confirm", "unclear"),
]

PERMISSION_HINT = (
    "  The recording is pure silence: macOS probably blocked the microphone.\n"
    "  System Settings > Privacy & Security > Microphone > switch on Terminal (or Visual Studio Code),\n"
    "  then quit that app completely (Cmd+Q), reopen it and run this script again."
)


def wav_path(phrase_id: str) -> Path:
    return OUT_DIR / f"{phrase_id}.wav"


def write_index() -> None:
    """testset.json: the expected text of every phrase. The benchmark reads this file."""
    items = [
        {"id": pid, "file": wav_path(pid).name, "text": text, "category": category, "confirm": confirm}
        for pid, text, category, confirm in PHRASES
    ]
    INDEX_FILE.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")


def record_once() -> np.ndarray:
    """Record from the default microphone between two Enter presses. Returns mono float32 audio."""
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())  # runs in a separate audio thread, ~every few milliseconds

    input("  Press Enter to START speaking...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("  Recording... press Enter to STOP.")

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:, 0]  # list of (n, 1) blocks -> one flat vector


def check_audio(audio: np.ndarray) -> list[str]:
    """Simple quality warnings: too short, too quiet, too loud."""
    warnings = []
    seconds = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if seconds < MIN_SECONDS:
        warnings.append(f"very short ({seconds:.2f} s): did you press Enter twice?")
    if 0.0 < peak < QUIET_PEAK:
        warnings.append(f"very quiet (peak {peak:.3f}): speak up or move closer")
    if peak >= CLIP_PEAK:
        warnings.append(f"too loud (peak {peak:.2f}): the audio is distorted, move a bit back")
    return warnings


def play(audio: np.ndarray) -> None:
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def record_phrase(phrase_id: str, text: str) -> None:
    """Record one phrase until sir keeps it, then save it as a 16-bit WAV."""
    while True:
        print(f'\n[{phrase_id}] Say: "{text}"')
        audio = record_once()
        seconds = len(audio) / SAMPLE_RATE
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        print(f"  {seconds:.1f} s, peak {peak:.2f}")
        if len(audio) and peak == 0.0:
            print(PERMISSION_HINT)
        for warning in check_audio(audio):
            print(f"  ! {warning}")

        while True:
            choice = input("  Enter = keep, p = play back, r = record again: ").strip().lower()
            if choice == "p":
                play(audio)
            elif choice == "r":
                break  # back to the outer loop: record again
            elif choice == "":
                sf.write(wav_path(phrase_id), audio, SAMPLE_RATE, subtype="PCM_16")
                print(f"  Saved {wav_path(phrase_id)}")
                return
            else:
                print("  Type nothing (Enter), p or r.")


def print_list() -> None:
    total = 0.0
    for pid, text, category, _ in PHRASES:
        path = wav_path(pid)
        if path.exists():
            seconds = sf.info(path).duration
            total += seconds
            status = f"{seconds:4.1f} s"
        else:
            status = "  --  "
        print(f"  {pid}  {status}  [{category:<7}]  {text}")
    done = sum(wav_path(pid).exists() for pid, *_ in PHRASES)
    print(f"  {done} of {len(PHRASES)} recorded, {total:.1f} s of audio in total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the fixed STT test set.")
    parser.add_argument("--redo", nargs="+", metavar="ID", help="record these phrase ids again, e.g. --redo 08 12")
    parser.add_argument("--list", action="store_true", help="show the phrases and which ones are recorded")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_index()

    if args.list:
        print_list()
        return

    known = {pid: text for pid, text, *_ in PHRASES}
    if args.redo:
        unknown = [pid for pid in args.redo if pid not in known]
        if unknown:
            print(f"Unknown phrase id(s): {', '.join(unknown)}. Valid: 01 to {PHRASES[-1][0]}.")
            return
        todo = args.redo
    else:
        todo = [pid for pid in known if not wav_path(pid).exists()]

    if not todo:
        print(f"All {len(PHRASES)} phrases are recorded. Use --list to check, --redo ID to record one again.")
        return

    print(f"{len(todo)} phrase(s) to record. Speak normally, at the distance you will use Jarvis, in a quiet room.")
    try:
        for pid in todo:
            record_phrase(pid, known[pid])
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Saved files are kept; run the script again to continue.")
        return
    except sd.PortAudioError as error:
        print(f"\nMicrophone error: {error}")
        print(PERMISSION_HINT)
        return

    print()
    print_list()


if __name__ == "__main__":
    main()
