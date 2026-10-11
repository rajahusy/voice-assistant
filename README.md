# Lightweight AIT English assistant

Implemented in `/home/aitcs/Documents/ait_project`. This reuses the heading, Q&A and source-metadata parsers from the nearby `voice-assistant` project. Its original project and environment are untouched. The college document was copied from that project; it is a dated source snapshot, not a live verification of college facts. Other files in `knowledge/` are also discovered automatically.

## Start with text

Python 3.13.5 with SQLite FTS5 is sufficient. No pip dependencies, embeddings or model download are needed for text mode.

```bash
cd /home/aitcs/Documents/ait_project
python3 assistant.py --check
python3 assistant.py --mode text
python3 assistant.py --text "Who is the principal of AIT?" --metrics
python3 assistant.py --reindex
python3 assistant.py --evaluate evaluation.json
python3 -m unittest discover -s tests -v
```

On Windows use `python` instead of `python3`. Copy the project source and documents, then create a fresh virtual environment on each machine. Do not copy a Windows virtual environment to the Pi.

In text mode `/metrics` reports the last turn's timings and current process memory. `stop` clears the current interaction without ending text mode; `exit` ends it. Voice `stop` returns to listening and `exit` ends the session. Ctrl+C interrupts playback and exits voice mode. Spoken interruption during playback is intentionally unavailable in this half-duplex version: the microphone is closed while speaking to prevent self-triggering.

## Documents and evidence

Put TXT, Markdown, CSV, JSON, JSONL, PDF or DOCX documents in `knowledge/`. Q&A records use `question` and `answer`, optionally `source`, `as_of`, `category` and `keywords`. Other JSON objects/CSV rows are indexed as named fields. JSONL streams one record at a time; JSON arrays have a 16 MiB limit. PDF requires the optional `requirements-documents.txt`; scanned images need OCR beforehand. DOCX uses ZIP/XML from Python's standard library. Encrypted PDFs, malformed documents and unsupported files get an indexing diagnostic, while other documents continue working.

Startup indexes only changed files, using modification time/size and then SHA-256 when needed. Each file is indexed transactionally, deleted files are removed, and a malformed update retains its previous index. An unchanged startup does not parse all documents or load them into memory. Search uses SQLite FTS5/Porter tokenization and BM25, requested-term coverage, a heading tie-breaker, and duplicate-passage suppression. Identical chunks within a file are stored once; duplicates across sources retain provenance but are collapsed in search results.

Source references appear separately from spoken answers. Offline replies use source sentences or stored Q&A answers. The conversational resolver keeps six recent turns and uses the previous topic for pronouns and incomplete questions; this is a lightweight heuristic, not full language understanding. Missing requested terms are rejected even when context terms match. Broad or unusually phrased questions may need better source wording or keyword aliases. The supplied four-case retrieval evaluation checks source recall, not answer truth; review your documents and expand the evaluation for real use.

Date/day/time questions use the system clock in Asia/Kolkata (India Standard Time). Keep the device clock synchronized. Questions with explicit current/year-specific wording use the OpenAI fallback with required web search when cloud processing is enabled. College questions missing from local documents also use this fallback; it requests official evidence and reports when reliable information is unavailable. Dates and qualifications in stored records describe those records, not guarantees that the facts are still current.

## Configuration and cloud consent

Copy `config.example.json` to `config.local.json`, edit it, and pass `--config config.local.json`. Relative document/model/database paths resolve against the project folder. Unknown keys and invalid provider/timing values are rejected. `--check` reports package availability and lists audio devices when sounddevice is installed; it never sends an API request or prints your key.

Cloud processing is disabled by default. To permit general-knowledge answers and optional speech transcription, set `cloud_enabled` to `true` and supply `OPENAI_API_KEY` in the process environment. The program loads `OPENAI_API_KEY` and `OPENAI_MODEL` from the local, git-ignored `.env` file. Existing process environment values take precedence. If present, `config.local.json` is loaded automatically unless `--config` specifies a different file. The optional systemd unit can also load its environment file. Do not put secrets in source control.

```bash
export OPENAI_API_KEY='your-key'
python3 assistant.py --config config.local.json --mode text
```

PowerShell: `$env:OPENAI_API_KEY = 'your-key'`. General answers use `model` (default `gpt-4o-mini`); current and college fact fallbacks use `search_model` (default `gpt-4.1-mini`) with the Responses API `web_search` tool. See the [official OpenAI web search documentation](https://developers.openai.com/api/docs/guides/tools-web-search). Search incurs additional API charges. Transcription uses `transcription_model` (default `gpt-4o-transcribe`). Requests use standard-library HTTPS with a finite timeout and at most two attempts for transient failures. Authentication errors do not retry. Missing local answers and retrieval errors fall back to OpenAI when enabled. Cloud conversations send up to six prior cloud-answer turns. Local document replies are kept out of that cloud history; your current question and resolved topic still go to the API when routed online. If OpenAI fails, the app reports the error and continues accepting questions.

`cloud_documents: true` separately allows sending up to four relevant passages to the API for passage selection. The API returns an index; the final local answer remains extractive so newly generated institutional facts cannot enter the answer. It adds a request only when multiple useful passages exist. It is not unrestricted neural rewriting. Cloud STT sends the recorded WAV to OpenAI and requires both cloud permission and a key.

API references: [Responses](https://developers.openai.com/api/docs/guides/text), [speech transcription](https://developers.openai.com/api/docs/guides/speech-to-text).

## English voice on Raspberry Pi

1. Use Raspberry Pi OS 64-bit and your Python 3.13.5 installation. Verify `python3.13 --version`; do not replace the OS Python.
2. Install OS audio prerequisites and make a fresh virtual environment:

```bash
sudo apt update
sudo apt install libportaudio2 espeak-ng
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-voice.txt
.venv/bin/python assistant.py --check
```

3. Choose STT. For offline input, explicitly install the optional Vosk requirement, obtain an English model yourself, and configure `vosk_model` to its directory. No models are automatically downloaded. First confirm the wheel resolves for your target:

```bash
.venv/bin/python -m pip download --only-binary=:all: --no-deps -r requirements-vosk.txt -d /tmp/ait-wheel-check
.venv/bin/python -m pip install -r requirements-vosk.txt
```

Vosk's published wheel is `py3-none-manylinux2014_aarch64`, and its metadata allows Python >=3. This is packaging evidence, not proof its native runtime works with every Python 3.13.5 installation. Test loading your actual model. `stt: auto` tries a configured Vosk model and uses OpenAI transcription if local loading fails and cloud/key are configured. `stt: vosk` requires working Vosk and gives one diagnostic if unavailable. `stt: openai` uses the configured cloud transcription provider.

4. Start automatic voice listening:

```bash
.venv/bin/python assistant.py --config config.local.json --mode voice --metrics
```

After the welcome message, speak without pressing any keys. Recording ends at the configured silence duration or maximum recording duration, and the assistant returns to listening after each reply. Say "exit" or "quit" to finish; Ctrl+C also exits. Entirely quiet recordings produce no speech or API request. Tune `energy_threshold`, `silence_seconds`, `listen_timeout`, `max_recording`, and device indexes from `--check`. Automatic listening uses RMS energy detection, not a neural VAD; fans/noise can trigger it. The input stream is closed during transcription and speech, a cooldown follows speech, and near-identical transcripts shortly after playback are discarded. This guard can also suppress an immediate intentional repetition of the assistant's words. `continuous` and `--continuous` remain accepted for old configurations; voice mode always listens automatically.

5. `espeak-ng` is the minimal English fallback; it is intelligible robotic speech, not human-quality speech. For improved naturalness, obtain a compatible Piper binary and English voice model yourself, configure `tts: piper`, `piper_binary`, and `piper_model`, then test playback. Piper synthesis uses a temporary WAV and sounddevice playback. On Windows, `tts: auto` selects installed English SAPI via PowerShell; install the sounddevice requirement for input. `speech_rate` configures pacing; Piper uses length scale. The espeak/SAPI adapters use the OS default output device; `output_device` applies to Piper.

6. Optional service: edit **all** user/home paths in `deployment/ait-assistant.service`, ensure `config.local.json` provides working STT/TTS and continuous listening, then install it with your system administrator's permission. `Restart=no` avoids infinite hardware/authentication retry loops. A desktop user service may be more appropriate if audio devices belong to a logged-in session. Consult `journalctl -u ait-assistant.service` for one-time diagnostics.

Package metadata checked on 2026-10-11: [sounddevice 0.5.6](https://pypi.org/project/sounddevice/0.5.6/) supports Python >=3.7 with a pure-Python wheel; PortAudio is still required. [Vosk 0.3.45](https://pypi.org/project/vosk/0.3.45/) publishes an ARM64 manylinux wheel. [pypdf 6.20.0](https://pypi.org/project/pypdf/6.20.0/) requires Python >=3.9 with a pure-Python wheel. Native audio/model performance needs target testing. No optional packages or models were installed automatically.

## Validation and hardware checklist

`VALIDATION.json` records measured local indexing, retrieval, total response time and current RSS. STT, AI and synthesis times are zero when those stages are unused; zero is not a voice benchmark. `--metrics` collects each voice stage after a turn. The synthesis measurement includes playback. There are no invented speed claims for Raspberry Pi voice.

Automated tests cover the requested A–L behavior scenarios with temporary documents and injected fake audio/cloud adapters, plus indexing rollback, formats, bounded history and bounded network retries. The real `--mode voice` missing-dependency path was exercised. Live microphone, speaker, Vosk model, Piper, Windows SAPI, PDF extraction and authenticated OpenAI requests remain untested here. Before deployment:

- Record normal English, silence, background noise and a maximum-length utterance; confirm transcription quality and timing.
- Verify English speech, volume, voice speed, Ctrl+C interruption, `stop`, `exit`, and return to listening.
- Let speaker output reach the microphone; check that playback never records and residual echoes are suppressed.
- Unplug input/output hardware; confirm a clear diagnostic and bounded recovery, then confirm text mode works.
- Disconnect internet and test offline questions; test API authentication/timeout recovery with your configured provider.
- Restart without document changes and confirm skipped indexes; change/delete a file and confirm only the affected index changes.
- Run real admission, principal, qualification, fees and ambiguous HOD conversations against your reviewed documents. No heuristic retrieval system establishes source truth by itself.

Voice mode speaks `startup_greeting` once before opening the microphone. Edit that setting in `config.local.json` to change the welcome message; set it to an empty string to disable it. Text mode does not play audio.

Transcription uses `gpt-4o-transcribe`. The default and local `transcription_prompt` are empty to avoid steering ambiguous audio toward college vocabulary. `transcription_aliases` maps explicitly configured mishearings ("AAT Golan" and "AAT college") to "AIT college". Change/remove these entries when discussing other institutions. Corrected transcripts display both `Heard:` and `You:`. `Listening...` appears on screen at startup and after answers, never through TTS; silent capture loops do not repeat the status.

Voice mode requires `webrtcvad-wheels` from `requirements-voice.txt`. Before transcription, local WebRTC VAD checks 30 ms audio frames at 16 kHz, requiring at least `min_speech_seconds` (default 0.3) of detected speech and a 15% speech fraction. DC offset is removed and other input rates are resampled using linear interpolation. Noise-only captures are discarded without an API call. VAD reduces false triggers but can still mistake some noises for speech or reject short/quiet utterances.

OpenAI transcription requests token log probabilities where supported. Mean confidence below `transcription_min_logprob` (default -0.7) prompts the user to repeat rather than answering. Empty transcripts stay quiet. An unexpected named college (such as AD College instead of the configured AIT domain) prompts the user to repeat the college name before answering. `voice_correction: true` (enabled locally, disabled in the example) sends recognized text to the configured LLM for a conservative clarity/spelling check. It gets no conversation history and may not add missing subjects or intent. A local token check rejects added content and changed numbers. Ambiguity, malformed corrections and correction API failures produce a clarification instead of an answer. Explicit requested department clarifications and exit/stop commands bypass correction. This adds an API request to most spoken questions; it does not establish that a transcript matches the original audio.

After `idle_prompt_seconds` (default 120) without accepted speech, the assistant says "How can I help you?" and keeps listening. The timer resets after the reminder or a response, so it does not repeat on every capture loop. Clarifications and reminders do not enter answer history. Audio is not saved by these checks.

### Concise answers with LLM assistance

`cloud_compose: true` (requires `cloud_documents` and `cloud_enabled`) sends at most four relevant retrieved passages to the configured text model. For principal/HOD questions the model chooses only the name; other questions use a short directly relevant source span. Its JSON result must contain an exact quote present in the selected passage, or the app falls back to local extraction. This permits evidence selection and shortening without accepting newly generated document facts. It can add one API request per locally answered question. The current local configuration enables this at the user's request; the example configuration keeps it off. CSE abbreviations are expanded for retrieval, and offline HOD answers favor leadership sentences over department descriptions. Source references remain separate from spoken answers. Stored source dates still describe the snapshot; brief replies do not establish current incumbency.

### Smoother neural speech

The local configuration now selects `tts: openai`, `tts_model: gpt-4o-mini-tts`, `tts_voice: marin`, and `tts_speed: 0.95`. `tts_instructions` requests warm, calm English with natural pauses and AIT pronounced letter by letter. This sends the spoken answer text to OpenAI and uses paid API requests and internet connectivity. The terminal identifies the voice as AI-generated. The API returns 24 kHz 16-bit mono PCM, which is wrapped in a finalized WAV in memory and played through sounddevice. The local configuration uses `output_device: null` so Linux playback follows the PipeWire default speaker through `pw-play`, including Bluetooth. Explicit `output_device` settings use sounddevice and bypass the desktop default; use them only for a specific hardware connection. On a cloud or playback failure the session switches once to espeak-ng (Windows: SAPI) instead of repeatedly retrying neural speech. Set `tts` to `espeak` to return to local robotic speech, or configure Piper for an optional natural offline voice. No additional Python packages or voice models were downloaded for this change.

### Desktop autostart on this Pi

`deployment/ait-voice-assistant.desktop` opens LXTerminal and runs `deployment/start-desktop-voice.sh` after the user logs into the desktop. The launcher waits eight seconds for audio/Bluetooth startup, checks the default speaker with a bounded wait, and starts the existing virtual-environment Python in voice mode. A file lock prevents duplicate autostart launches. It uses the desktop audio session and current default speaker, including OD DISCO, and does not require root or change system boot settings. The launcher paths are specific to `/home/aitcs/Documents/ait_project`. The installed entry is `~/.config/autostart/ait-voice-assistant.desktop`; move it out of that directory to disable autostart. Desktop auto-login is a separate Raspberry Pi setting; without it, the assistant starts after you log in. Log out and back in to verify startup; no reboot has been performed by the agent.
