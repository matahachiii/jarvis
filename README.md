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
- **Hard guarantees in code, not in the prompt**: the model handles language and choices;
  confirmation, refusals and limits are enforced by code.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Foundation: environment, structure, local LLM, test script | done |
| 1 | Text Jarvis: terminal chat loop with memory and personality | done |
| 2 | Tools: open apps, search files, Spotify, ... + confirmation + logging | done |
| 3 | Voice: speech-to-text + text-to-speech (push-to-talk) | |
| 4 | Always on: wake word "Hey Jarvis" + streaming | |
| 5 | Persistent memory + agenda/mail integration | |
| 6 | The village: Jarvis as orchestrator of specialist agents | |
| 7 | Polish: HUD, autostart, monitoring | |

## Project structure

```
jarvis/
  core/      chat loop, agent loop, conversation memory, personality, session log
  brain/     LLM interface and providers (Ollama, cloud)
  tools/     actions on the Mac (registry + one module per tool group)
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

Phase 2 added no new dependencies (standard library only).

## Usage

Start Text Jarvis:

```bash
python -m jarvis
```

Optional shortcut: add this line to `~/.zshrc`, then just type `jarvis` in any terminal:

```bash
alias jarvis='cd ~/Desktop/Jarvis && conda deactivate 2>/dev/null; source .venv/bin/activate && python -m jarvis'
```

Type a message and Jarvis streams his answer back, word by word. He remembers the
conversation within a session (older turns are trimmed automatically to fit the context window).
When a request needs a tool, the call and its result are shown as dimmed `[tool]` lines.
Tools marked "confirm" ask first:

```
  About to run: quit_app(app_name='Notes')
  Allow? [y/N]
```

Only `y` or `yes` counts as yes; anything else (or Ctrl+D) is no.

| Command | What it does |
|---|---|
| `/help` | show the commands |
| `/tools` | list the tools and their risk level |
| `/reset` | forget the conversation (the personality stays) |
| `/stats` | timing and tokens of the last answer, current history size |
| `/exit`, `/quit` | quit (also Ctrl+D, or Ctrl+C at the prompt) |

Ctrl+C while Jarvis is answering stops the answer.
The personality lives in `config/personality.md`; all other settings in `config/config.toml`.

## Tools

Every tool runs through `registry.execute()`, the only way a tool can run: arguments are
validated, "confirm" tools only run after an explicit yes, every tool has a timeout, and
every error becomes a short `FAILED: ...` message the model reads. There is no generic shell tool.

| Tool | Risk | What it does | Limits |
|---|---|---|---|
| `calculator` | safe | Arithmetic via `ast` (never `eval`); `%`, `of`, `x`, `×`, `÷`, `^`, `1,234` | exponent capped at 100 |
| `get_time` | safe | Exact time + spoken British form | |
| `open_app` | safe | Opens installed apps only; aliases (vs code, chrome); typo suggestions | never guesses an app |
| `quit_app` | **confirm** | Polite quit, re-checks before saying "Closed" | Finder, Dock, Terminal, VS Code, iTerm, Ollama are protected |
| `spotify` | safe | play / pause / next / previous / now_playing | never starts Spotify itself; cannot search songs |
| `volume` | safe | get / set / up / down (±10) / mute / unmute; reads the level back | |
| `search_files` | safe | Files and folders by name in the home folder, newest 5 | skips Library, hidden folders, node_modules, site-packages |
| `open_item` | safe | http(s) URLs, files and folders inside the home folder | never opens programs or scripts (.app, .sh, .py, .pkg, .dmg, executables) |
| `weather` | safe | Open-Meteo (no key), today or tomorrow, any city | default location Buizingen |

Safety rules enforced in code:
- A "confirm" tool never runs without a yes; no confirmation callback means no.
- When sir declines every action of a round, Jarvis answers with a fixed sentence
  ("Very well, sir. Nothing was done.") instead of letting the model word it.
- At most 5 tool rounds per question; then Jarvis gives up politely.
- Tool results are cut to 400 characters, string arguments to 300.

### macOS permissions

- **Automation**: the first time Jarvis quits an app or controls Spotify, macOS asks
  "Visual Studio Code (or Terminal) wants to control <App>". Click OK.
- **Files**: macOS may ask for access to Documents/Downloads when `search_files` reads file dates. Click Allow.
- No Accessibility permission is needed.

## Session log

Every session is written to `logs/session_<start time>.jsonl` (git-ignored): one JSON object
per line, written immediately, so the log survives a crash.

| role | text | extra fields |
|---|---|---|
| `system` | system prompt | `brain`, `tools` |
| `user` | what sir typed | |
| `tool` | e.g. `quit_app(app_name='Notes')` | `tool`, `arguments`, `risk`, `confirmed` (true/false/null), `success`, `message`, `duration_s` |
| `assistant` | final answer (cleaned for speech) | `stats`, `raw` (only if the cleaner changed something) |
| `event` | `answer interrupted`, `brain connection lost`, `brain error: ...`, `reset`, `tool round cap reached` (+ `rounds`), `all actions declined` (+ `tools`), `session end` | |

`duration_s` is the time the tool itself ran: sir's y/N thinking time is not included
(0.0 if the tool never ran). A tool line is written the moment the tool finishes, so an
action that really happened is logged even if the turn is interrupted afterwards.

Reading a log in R:

```r
library(jsonlite)
log <- stream_in(file("logs/session_2026-11-17_16-37-32.jsonl"), verbose = FALSE)
log[log$role == "tool", c("timestamp", "tool", "confirmed", "success", "duration_s")]
```

## Tests

Unit tests (no LLM, no real Mac actions, mocks only, a few seconds):

```bash
python -m unittest test_registry test_conversation_unit test_agent test_tools_basic test_tools_mac test_tools_media test_tools_files test_tools_weather
```

Expected: `Ran 139 tests ... OK`.

LLM tests (need Ollama; close heavy apps first; `-u` shows the output live):

| Command | What it tests | Decision rule |
|---|---|---|
| `python -u test_tools.py` | Phase 2: 25 cases × 3 runs; real brain, personality and examples, fake tools (calculator real), confirm always NO. ~5 min | tool choice ≥ 90%, arguments ≥ 90%, 0 false calls on chit-chat, ≤ 1 on knowledge, grounded ≥ 90%, honest 100%, quit_app never ran, history re-checks ≥ 2/3, median first word on tool turns ≤ 3.0 s |
| `python -u test_conversation.py` | Phase 1 (updated in Phase 2): memory, style, honesty, no sycophancy, trimming, speed, through the same pipeline as the chat (examples + fake tools). ~1.5 min | all 6 checks PASS, in 3 out of 3 runs |
| `python test_brain.py` | Phase 0: is the brain fast enough for voice? | |
| `test_toolcalling.py` | Phase 2 step 1 benchmark (historical, does not run as-is anymore) | |

## Results Phase 2

- **Step 1 benchmark** (`test_toolcalling.py`): 51/51 correct tool choice, 27/27 arguments,
  0 false calls → the brain stays local (qwen3 4B), no cloud provider needed.
- **Behaviour test** (`test_tools.py`), tuning history: baseline 60% tool choice → rules in the
  prompt 79% → example turns 92% → + volume-repeat example 96% → + fixed declined answer in code:
  9/9 PASS.
- **Final** (shorter personality, step 6.3): test_tools 9/9 PASS (tool choice 72/72, arguments 48/48,
  grounded 29/30, honest 15/15, history 6/6, first word median 1.64 s, max 3.39 s);
  test_conversation 3/3 PASS.
- **Lessons**: rules in the prompt barely work for a 4B model, examples do. Exact wording matters:
  dropping "never apologise for being correct" made Jarvis apologise in 3 out of 3 runs.

## Budgets (MacBook Air M1, 8 GB)

| | |
|---|---|
| Context | 4096 tokens; max answer 512; safety margin 64 → budget 3520 |
| Fixed prompt (personality + example turns + 9 tool definitions + short message) | 1998 tokens (was 2244 before step 6.3; ~117 tokens per tool definition) |
| Left for history | ~1520 tokens (~8-10 turns); the oldest whole turns are trimmed |
| RAM with the model loaded | ~0.8-0.95 GB available, swap ~1-3 GB (close Chrome/WhatsApp while working) |
| Latency | normal answers: first word ~0.3 s; tool turns: ~1.5-2 s (tool calls are not streamed, then a second call for the answer) |

## Known limitations (local 4B model)

- Invents facts with confidence (it once said Brazil won the 2018 World Cup). Fix later: Wikipedia tool or cloud brain.
- "Is it already past noon?" after an earlier time answer: reasons on the old time instead of calling
  `get_time`. Examples and tool descriptions did not help. Structural fix: add the current time to every user message.
- Describes the weather or "the sky" in greetings without a tool (3 out of 3 runs). A "no senses" rule
  in the prompt did not help and was removed in step 6.3.
- Adds invented commentary to tool results, e.g. "Turn the volume up a bit" → "which is actually a decrease"
  (3 out of 3 runs after step 6.3). Suspects: the volume numbers in the example turns are read as history;
  "Report only what they say, and nothing more" lost its last words in 6.3.
- Sometimes ends with an offer of help, or copies wording from the example turns.
- Writes larger numbers as digits despite the prompt ("40 percent", "4-2"). Fix in Phase 3: number-to-words before TTS.
- The first answer after start-up can be slow (~9 s seen once): the warm-up runs without the tool definitions.
  Fix in Phase 3: warm up with the tools.
- `open_item` can take ~7 s when the browser is not running yet; the tool timeout is 10 s.
- Spotify cannot search for songs (possible later via the Spotify Web API).
- Python cannot kill a stuck thread: after a timeout Jarvis stops waiting; the real protection is the
  subprocess timeouts inside the tools.
- Confirmation is a terminal y/N (Phase 3: a voice confirmation, same function).
- Words that seem glued together ("notrunning") in text copied from the terminal are a copy artefact at
  line wraps, not the model: check the text in the log.
