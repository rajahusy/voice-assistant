"""Validated, dependency-free configuration. Secrets come from the environment."""
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    knowledge_dir: str = 'knowledge'
    database: str = 'data/knowledge.sqlite3'
    domains: tuple = ('ait', 'college', 'institution', 'principal', 'admission', 'fees', 'hod', 'department', 'qualification', 'campus', 'scholarship', 'placement')
    cloud_enabled: bool = False
    cloud_documents: bool = False
    cloud_compose: bool = False
    model: str = 'gpt-4o-mini'
    transcription_model: str = 'gpt-4o-transcribe'
    transcription_prompt: str = ('English college enquiries at AIT College, Adichunchanagiri Institute of Technology, '
        'Chikkamagaluru. Vocabulary: AIT, A I T, college, principal, HOD, admission, fees, qualification, '
        'Computer Science, CSE, ECE. Transcribe what is spoken; do not answer the question.')
    transcription_aliases: dict = field(default_factory=lambda: {'aat golan': 'AIT college', 'aat college': 'AIT college'})
    timeout: float = 15
    history_turns: int = 6
    stt: str = 'auto'
    vosk_model: str = ''
    tts: str = 'auto'
    tts_model: str = 'gpt-4o-mini-tts'
    tts_voice: str = 'marin'
    tts_speed: float = 0.95
    tts_instructions: str = ('Speak in clear, warm, natural conversational English. Use a calm tone, smooth pacing, '
        'and short natural pauses. Pronounce AIT as the letters A I T. Read names carefully. Do not add or change words.')
    voice: str = 'en-us'
    speech_rate: int = 165
    startup_greeting: str = 'Welcome to the college assistant. How can I help you?'
    piper_binary: str = 'piper'
    piper_model: str = ''
    continuous: bool = True
    sample_rate: int = 16000
    max_recording: float = 15
    silence_seconds: float = 0.9
    listen_timeout: float = 6
    energy_threshold: int = 350
    cooldown_seconds: float = 0.7
    input_device: object = None
    output_device: object = None

    def path(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    def validate(self):
        if not 5 <= self.history_turns <= 8:
            raise ValueError('history_turns must be between 5 and 8')
        for name in ('timeout', 'sample_rate', 'max_recording', 'silence_seconds', 'listen_timeout', 'speech_rate'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.energy_threshold < 0 or self.cooldown_seconds < 0:
            raise ValueError('energy_threshold and cooldown_seconds cannot be negative')
        if self.stt not in ('auto', 'vosk', 'openai') or self.tts not in ('auto', 'espeak', 'piper', 'windows', 'openai'):
            raise ValueError('Unsupported speech provider')
        if not 0.25 <= self.tts_speed <= 4:
            raise ValueError('tts_speed must be between 0.25 and 4')
        if self.tts == 'openai' and not self.cloud_enabled:
            raise ValueError('OpenAI speech output requires cloud_enabled')
        if self.cloud_documents and not self.cloud_enabled:
            raise ValueError('cloud_documents requires cloud_enabled')
        if self.cloud_compose and not self.cloud_documents:
            raise ValueError('cloud_compose requires cloud_documents')
        return self


def load_env(path):
    """Load supported settings as literal data; existing environment wins."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        name, value = name.strip(), value.strip()
        if name not in ('OPENAI_API_KEY', 'OPENAI_MODEL'):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(name, value)


def load_config(path=None):
    load_env(ROOT / '.env')
    if path is None and (ROOT / 'config.local.json').exists():
        path = ROOT / 'config.local.json'
    data = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    if os.environ.get('OPENAI_MODEL'):
        data['model'] = os.environ['OPENAI_MODEL']
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    return Config(**data).validate()
