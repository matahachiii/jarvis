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
| 1 | Text Jarvis: terminal chat loop with memory and personality | |
| 2 | Tools: open apps, search files, Spotify, ... + confirmation + logging | |
| 3 | Voice: speech-to-text + text-to-speech (push-to-talk) | |
| 4 | Always on: wake word "Hey Jarvis" + streaming | |
| 5 | Persistent memory + agenda/mail integration | |
| 6 | The village: Jarvis as orchestrator of specialist agents | |
| 7 | Polish: HUD, autostart, monitoring | |

## Project structure

```
jarvis/
  core/      conversation loop, orchestrator
  brain/     LLM interface and providers (Ollama, cloud)
  tools/     actions on the Mac
  agents/    specialist agents (the village)
  voice/     speech-to-text, text-to-speech, wake word
  memory/    persistent memory
config/      settings (config.toml)
data/        local data (not in git)
logs/        log files (not in git)
```

## Setup

Requirements: macOS on Apple Silicon, Homebrew, Python 3.11, Ollama.

```bash
cd ~/Desktop/Jarvis
conda deactivate            # only if Anaconda is installed
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:4b
python test_brain.py
```
