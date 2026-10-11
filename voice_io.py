"""Half-duplex hardware adapters, independent from document/conversation logic."""
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from enum import Enum
from difflib import SequenceMatcher
from collections import deque
from openai_fallback import OpenAIClient, CloudError
from answer_service import Reply

class State(str, Enum):
    IDLE = 'IDLE'
    LISTENING = 'LISTENING'
    PROCESSING = 'PROCESSING'
    SPEAKING = 'SPEAKING'
    ERROR_RECOVERY = 'ERROR_RECOVERY'

class SpeechError(RuntimeError):
    pass


def normalize_transcript(text, config):
    """Correct configured institution mishearings without fuzzy name guessing."""
    text = text.strip()
    # Remove spoken greetings/request prefixes, not the actual question.
    text = re.sub(r'^(?:hi|hello|hey)\b[\s,.!]*', '', text, flags=re.I)
    text = re.sub(r'^(?:i asked|i ask|my question is|can you tell me|could you tell me)[\s,:]*', '', text, flags=re.I)
    text = re.sub(r'\ba[ .-]+i[ .-]+t\b\.?', 'AIT', text, flags=re.I)
    for heard, intended in config.transcription_aliases.items():
        text = re.sub(r'(?<!\w)' + re.escape(heard) + r'(?!\w)', lambda _: intended, text, flags=re.I)
    if re.search(r'\bait\b', text, re.I):
        text = re.sub(r'\b(?:principle|princilple)\b', 'principal', text, flags=re.I)
    return text.strip()

class Microphone:
    def __init__(self, config):
        self.config = config
        try:
            import sounddevice
        except ImportError as error:
            raise SpeechError('Python audio package is missing. Install requirements-voice.txt in .venv and run .venv/bin/python assistant.py --mode voice.') from error
        try:
            self.sd = sounddevice
            sounddevice.check_input_settings(device=config.input_device, channels=1, dtype='int16', samplerate=config.sample_rate)
        except Exception as error:
            raise SpeechError(f'Microphone unavailable at {config.sample_rate} Hz on {config.input_device!r}: {error}. Check the device, supported sample rate and PortAudio; text mode still works.') from error

    def record(self):
        cfg = self.config
        frames, voiced, silent = [], False, 0
        preroll = deque(maxlen=6)
        overflows = 0
        block = max(1, int(cfg.sample_rate * 0.05))
        try:
            # Stream is closed before transcription and playback: no speaker capture.
            # Let PortAudio choose native buffer sizes; use a larger capture
            # latency so ordinary Pi scheduling delays do not lose samples.
            with self.sd.RawInputStream(samplerate=cfg.sample_rate, channels=1, dtype='int16', device=cfg.input_device, blocksize=0, latency=0.2) as stream:
                deadline = time.monotonic() + cfg.max_recording
                for index in range(math.ceil(cfg.max_recording / 0.05)):
                    if time.monotonic() >= deadline:
                        break
                    data, overflow = stream.read(block)
                    if overflow:
                        overflows += 1
                        if overflows == 1:
                            logging.warning('Microphone dropped some samples; continuing capture with buffered audio')
                        if overflows > 3:
                            raise SpeechError('Microphone repeatedly dropped audio samples. Close other audio apps and retry; check USB power and system load.')
                    data = bytes(data)
                    samples = struct.unpack('<' + 'h' * (len(data) // 2), data)
                    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1))
                    active = rms >= cfg.energy_threshold
                    if active:
                        if not voiced:
                            frames.extend(preroll)
                        voiced, silent = True, 0
                    elif voiced:
                        silent += 1
                    if voiced:
                        frames.append(data)
                    else:
                        preroll.append(data)
                    if voiced and silent * 0.05 >= cfg.silence_seconds:
                        break
                    if not voiced and (index + 1) * 0.05 >= cfg.listen_timeout:
                        return b''
        except KeyboardInterrupt:
            raise
        except SpeechError:
            raise
        except Exception as error:
            raise SpeechError('Recording failed. Check the microphone connection.') from error
        if not voiced:
            return b''
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(cfg.sample_rate)
            wav.writeframes(b''.join(frames))
        return output.getvalue()

class SpeechDetector:
    """Reject non-speech locally before any transcription request."""
    def __init__(self, config):
        try:
            import webrtcvad
        except ImportError as error:
            raise SpeechError('Install requirements-voice.txt for speech detection before using voice mode.') from error
        self.vad = webrtcvad.Vad(3)
        self.config = config

    def has_speech(self, audio):
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise SpeechError('Speech detection requires mono 16-bit WAV audio')
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
        samples = struct.unpack('<' + 'h' * (len(raw) // 2), raw)
        if not samples:
            return False
        # Remove DC offset, then resample to WebRTC's supported 16 kHz rate.
        mean = sum(samples) / len(samples)
        pcm = []
        for index in range(int(len(samples) * 16000 / rate)):
            position = index * rate / 16000
            left = int(position)
            fraction = position - left
            value = samples[left] * (1 - fraction) + samples[min(left + 1, len(samples) - 1)] * fraction - mean
            pcm.append(max(-32768, min(32767, round(value))))
        detector = type(self.vad)(3)  # Reset hangover state for every recording.
        speech = 0
        frames = len(pcm) // 480
        for index in range(frames):
            chunk = struct.pack('<480h', *pcm[index * 480:(index + 1) * 480])
            speech += detector.is_speech(chunk, 16000)
        return speech * 0.03 >= self.config.min_speech_seconds and speech / max(frames, 1) >= 0.15


class Transcriber:
    def __init__(self, config, cloud):
        self.config, self.cloud, self.model = config, cloud, None
        self.detector = SpeechDetector(config)
        self.unclear = False
        self.provider = ''
        if config.stt in ('auto', 'vosk') and config.vosk_model:
            try:
                import vosk
                model_path = config.path(config.vosk_model)
                if not model_path.is_dir():
                    raise SpeechError('Vosk model directory is missing')
                vosk.SetLogLevel(-1)
                self.vosk = vosk
                self.model = vosk.Model(str(model_path))
                self.provider = 'vosk'
            except Exception as error:
                if config.stt == 'vosk':
                    raise SpeechError('Vosk could not load. Install a compatible package and English model, or configure cloud STT.') from error
        if not self.model:
            if config.stt in ('auto', 'openai') and config.cloud_enabled and os.environ.get('OPENAI_API_KEY'):
                self.provider = 'openai'
            else:
                raise SpeechError('No working STT provider. Configure an English Vosk model, or enable cloud processing and set OPENAI_API_KEY.')

    def transcribe(self, audio):
        self.unclear = False
        if not audio or not self.detector.has_speech(audio):
            return ''
        if self.provider == 'openai':
            text = self.cloud.transcribe(audio)
            self.unclear = not text.strip() or getattr(self.cloud, 'transcription_unclear', False) is True
            return text
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            recognizer = self.vosk.KaldiRecognizer(self.model, wav.getframerate())
            texts = []
            while data := wav.readframes(4000):
                if recognizer.AcceptWaveform(data):
                    texts.append(json.loads(recognizer.Result()).get('text', ''))
            texts.append(json.loads(recognizer.FinalResult()).get('text', ''))
        return ' '.join(texts).strip()

class Speaker:
    def __init__(self, config):
        self.config, self.process = config, None
        self.provider = config.tts
        if self.provider == 'auto':
            self.provider = 'piper' if config.piper_model and shutil.which(config.piper_binary) else ('windows' if os.name == 'nt' else 'espeak')
        if self.provider == 'piper':
            if not shutil.which(config.piper_binary) or not config.path(config.piper_model).is_file():
                raise SpeechError('Piper binary or voice model is missing')
        elif self.provider == 'espeak':
            self.binary = shutil.which('espeak-ng') or shutil.which('espeak')
            if not self.binary:
                raise SpeechError('Install espeak-ng or configure Piper for speech output')
        elif self.provider == 'windows' and os.name != 'nt':
            raise SpeechError('Windows SAPI speech output requires Windows')
        elif self.provider == 'openai':
            if not config.cloud_enabled or not os.environ.get('OPENAI_API_KEY'):
                raise SpeechError('Neural speech requires cloud_enabled and OPENAI_API_KEY')
            self.cloud = OpenAIClient(config)

    def play_wav(self, audio):
        # Follow the desktop's selected speaker, including Bluetooth. Direct
        # ALSA/PortAudio hardware devices bypass PipeWire's default routing.
        if self.config.output_device is None and shutil.which('pw-play'):
            with tempfile.TemporaryDirectory(prefix='ait-neural-voice-') as folder:
                path = os.path.join(folder, 'reply.wav')
                with open(path, 'wb') as stream:
                    stream.write(audio)
                self.run(['pw-play', path])
            return
        import sounddevice
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            if wav.getsampwidth() != 2:
                raise SpeechError('Voice playback expects 16-bit WAV audio')
            with sounddevice.RawOutputStream(samplerate=wav.getframerate(), channels=wav.getnchannels(),
                    dtype='int16', device=self.config.output_device) as stream:
                while data := wav.readframes(4096):
                    stream.write(data)

    def speak_neural(self, text):
        try:
            self.play_wav(self.cloud.synthesize(text))
        except KeyboardInterrupt:
            raise
        except Exception as error:
            # One downgrade per session avoids repeated failing cloud calls.
            if os.name == 'nt':
                self.provider = 'windows'
            else:
                self.binary = shutil.which('espeak-ng') or shutil.which('espeak')
                if not self.binary:
                    raise SpeechError('Neural speech failed and no local voice is installed') from error
                self.provider = 'espeak'
            print('Neural speech unavailable. Using the local voice for this session.')
            self.speak(text)

    def run(self, args, text=None):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE if text is not None else subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.process.communicate(text.encode('utf-8') if text is not None else None, timeout=60)
            if self.process.returncode:
                raise SpeechError('Voice playback or synthesis failed; check the output device')
        except subprocess.TimeoutExpired as error:
            self.stop()
            raise SpeechError('Voice synthesis timed out') from error
        finally:
            if self.process and self.process.poll() is not None:
                self.process = None

    def speak(self, text):
        if not text:
            return
        text = ' '.join(text.split())
        if not text:
            return
        text = re.sub(r'\bDr\.', 'Doctor', text)
        if text[-1] not in '.!?':
            text += '.'
        try:
            if self.provider == 'openai':
                self.speak_neural(text)
            elif self.provider == 'espeak':
                self.run([self.binary, '-v', self.config.voice, '-s', str(self.config.speech_rate), '--stdin'], text)
            elif self.provider == 'windows':
                # Text is data on stdin, never interpolated into PowerShell code.
                script = "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet,[System.Speech.Synthesis.VoiceAge]::NotSet,0,[System.Globalization.CultureInfo]::GetCultureInfo('en-US')); $s.Rate=" + str(max(-10, min(10, (self.config.speech_rate - 175) // 20))) + "; $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
                self.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script], text)
            else:
                with tempfile.TemporaryDirectory(prefix='ait-voice-') as directory:
                    path = str(os.path.join(directory, 'reply.wav'))
                    self.run([self.config.piper_binary, '--model', str(self.config.path(self.config.piper_model)), '--output_file', path,
                              '--length_scale', str(175 / self.config.speech_rate)], text)
                    import sounddevice
                    with wave.open(path, 'rb') as wav:
                        with sounddevice.RawOutputStream(samplerate=wav.getframerate(), channels=wav.getnchannels(), dtype='int16', device=self.config.output_device) as stream:
                            while data := wav.readframes(4096):
                                stream.write(data)
        except KeyboardInterrupt:
            self.stop()
            raise
        except SpeechError:
            raise
        except Exception as error:
            raise SpeechError('Voice output failed; check the speaker and voice provider') from error

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.process = None

class VoiceSession:
    def __init__(self, manager, microphone, transcriber, speaker, config):
        self.manager, self.microphone, self.transcriber, self.speaker, self.config = manager, microphone, transcriber, speaker, config
        self.state, self.last_answer, self.spoken_at = State.IDLE, '', 0
        self.last_activity = time.monotonic()

    def speak_notice(self, text, route):
        self.state = State.SPEAKING
        print('Assistant:', text)
        try:
            self.speaker.speak(text)
        finally:
            self.last_answer, self.spoken_at = text, time.monotonic()
            if route == 'idle':
                self.last_activity = self.spoken_at
            time.sleep(self.config.cooldown_seconds)
            self.state = State.IDLE
        print('Listening...', flush=True)
        return Reply(text, route)

    def quiet(self):
        self.state = State.IDLE
        if time.monotonic() - self.last_activity >= self.config.idle_prompt_seconds:
            return self.speak_notice('How can I help you?', 'idle')
        return None

    def is_echo(self, text):
        return bool(self.last_answer and time.monotonic() - self.spoken_at < 4
                    and SequenceMatcher(None, text.lower().strip(), self.last_answer.lower().strip()).ratio() >= 0.82)

    def step(self):
        started = time.perf_counter()
        self.state = State.LISTENING
        audio = self.microphone.record()
        if not audio:
            return self.quiet()
        self.state = State.PROCESSING
        t = time.perf_counter()
        question = self.transcriber.transcribe(audio).strip()
        transcription = time.perf_counter() - t
        if not question or self.is_echo(question):
            return self.quiet()
        if getattr(self.transcriber, 'unclear', False) is True:
            return self.speak_notice("I didn't hear that clearly. Could you repeat your question?", 'clarify')
        self.last_activity = time.monotonic()
        raw_question = question
        question = normalize_transcript(question, self.config)
        if not question:
            self.state = State.IDLE
            return None
        command = question.lower().strip(' .!?')
        commands = ('stop', 'exit', 'quit', 'stop assistant', 'goodbye')
        if command not in commands:
            named_college = re.search(r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+[Cc]ollege\b', question)
            if named_college:
                name = named_college.group(1)
                if name.lower() not in ('the', 'this', 'your', 'our', 'my') and not any(
                        re.search(r'\b' + re.escape(d) + r'\b', name, re.I) for d in self.config.domains):
                    return self.speak_notice(f"I heard {name} College. Could you repeat the college's name clearly?", 'clarify')
            # A lone topic is not a question unless we explicitly requested it.
            words = re.findall(r'\w+', question)
            if len(words) < 3 and not self.manager.pending_question and not self.manager.topic:
                return self.speak_notice('Could you say your full question?', 'clarify')
            if self.config.voice_correction and self.config.cloud_enabled and not self.manager.pending_question:
                try:
                    question = self.transcriber.cloud.correct_question(question)
                except (CloudError, ValueError, TypeError):
                    question = None
                if not question:
                    return self.speak_notice("I didn't understand the question clearly. Could you say it again?", 'clarify')
        if question != raw_question:
            print('Heard:', raw_question)
        print('You:', question)
        reply = self.manager.respond(question)
        self.manager.metrics['transcription_seconds'] = transcription
        if reply.command:
            self.manager.metrics['total_seconds'] = time.perf_counter() - started
            self.speaker.stop()
            self.state = State.IDLE
            if reply.command == 'stop':
                print('Listening...', flush=True)
            return reply
        print('Assistant:', reply.text)
        self.state = State.SPEAKING
        t = time.perf_counter()
        try:
            self.speaker.speak(reply.text)
        finally:
            self.manager.metrics['synthesis_seconds'] = time.perf_counter() - t
            self.manager.metrics['total_seconds'] = time.perf_counter() - started
        self.last_answer, self.spoken_at = reply.text, time.monotonic()
        self.last_activity = self.spoken_at
        time.sleep(self.config.cooldown_seconds)
        self.state = State.IDLE
        print('Listening...', flush=True)
        return reply

    def run(self):
        if getattr(self.speaker, 'provider', '') == 'openai':
            print('Speech voice: OpenAI neural TTS (AI-generated).')
        if self.config.startup_greeting:
            try:
                self.state = State.SPEAKING
                print('Assistant:', self.config.startup_greeting)
                self.speaker.speak(self.config.startup_greeting)
                self.last_answer = self.config.startup_greeting
                self.spoken_at = time.monotonic()
                self.last_activity = self.spoken_at
                time.sleep(self.config.cooldown_seconds)
            except KeyboardInterrupt:
                self.speaker.stop()
            except Exception as error:
                self.state = State.ERROR_RECOVERY
                print(f'Diagnostic: Startup greeting failed: {error}')
                self.speaker.stop()
                self.state = State.IDLE
                return
            self.state = State.IDLE
        print('Voice ready. Speak when ready. Say exit to finish; Ctrl+C also exits.')
        print('Listening...', flush=True)
        failures = 0
        while True:
            try:
                reply = self.step()
                failures = 0
                if reply and reply.command in ('exit', 'quit', 'goodbye', 'stop assistant'):
                    break
            except KeyboardInterrupt:
                self.speaker.stop()
                self.state = State.IDLE
                break
            except EOFError:
                break
            except Exception as error:
                self.state = State.ERROR_RECOVERY
                failures += 1
                if failures == 1:
                    print(f'Diagnostic: {error}')
                if failures >= 2:
                    print('Voice session ended. Use --mode text while correcting the diagnostic.')
                    break
                time.sleep(0.5)
        self.speaker.stop()
        self.state = State.IDLE
