# Jarvis

A local, privacy-first voice assistant for macOS, inspired by Jarvis from Iron Man.
You talk to Jarvis in English, he understands you and actually does things on your Mac.

The end goal: Jarvis as an **orchestrator** of a "village" of specialist agents
(market, research, agenda/mail, ...). You only talk to Jarvis; he delegates.

## Principles

- **€0 budget**: local and open-source first (Ollama, faster-whisper, Piper/Kokoro, openWakeWord).
- **Light on RAM**: built for an M1 MacBook with 8 GB. LLM, STT and TTS must fit together.
- **Swappable brain**: the LLM sits behind one interface. Switching between a local
  Ollama model and a free-tier cloud API is a config change, not a code change.
- **Step by step**: every phase ends with a clear "done" test.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Foundation: environment, structure, local LLM, test script | done |
| 1 | Text Jarvis: terminal chat loop with memory and personality | done |
| 2 | Tools: open apps, search files, Spotify, ... + confirmation + logging | |
| 3 | Voice: speech-to-text + text-to-speech (push-to-talk) | |
| 4 | Always on: wake word "Hey Jarvis" + streaming | |
| 5 | Persistent memory + agenda/mail integration | |
| 6 | The village: Jarvis as orchestrator of specialist agents | |
| 7 | Polish: HUD, autostart, monitoring | |

## Project structure

```
jarvis/
  core/      chat loop, conversation memory, personality, session log
  brain/     LLM interface and providers (Ollama, cloud)
  tools/     actions on the Mac
  agents/    specialist agents (the village)
  voice/     speech-to-text, text-to-speech, wake word
  memory/    persistent memory
config/      settings (config.toml) and Jarvis' personality (personality.md)
data/        local data (not in git)
logs/        session transcripts (not in git)
```

## Setup

Requirements: macOS on Apple Silicon, Homebrew, Python 3.11, Ollama.

```bash
cd ~/Desktop/Jarvis
conda deactivate            # only if Anaconda is installed
source .venv/bin/activate
pip install -r requirements.txt
brew services start ollama
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## Usage

Start Text Jarvis:

```bash
python -m jarvis
```

Type a message and Jarvis streams his answer back, word by word. He remembers the
conversation within a session (older turns are trimmed automatically to fit the context window).

| Command | What it does |
|---|---|
| `/help` | show the commands |
| `/reset` | forget the conversation (the personality stays) |
| `/stats` | timing and tokens of the last answer, current history size |
| `/exit`, `/quit` | quit (also Ctrl+D, or Ctrl+C at the prompt) |

Ctrl+C while Jarvis is answering stops the answer. Every session is saved as a JSONL
transcript in `logs/` (readable in R with `jsonlite::stream_in(file("logs/<session>.jsonl"))`).

The personality lives in `config/personality.md`; all other settings in `config/config.toml`.

## Tests

```bash
python test_brain.py          # Phase 0: is the brain fast enough for voice?
python test_conversation.py   # Phase 1: memory, style, honesty, no sycophancy, trimming, speed
```

## Known limitations (local 4B model)

- Invents facts with confidence (dates, scores, results). Fix later: web search tool or cloud brain.
- Unreliable at arithmetic. Fix in Phase 2: a calculator tool.
- Occasionally drops a space between two words ("can'tdo").
- Writes larger numbers as digits despite the prompt. Fix in Phase 3: number-to-words before TTS.
