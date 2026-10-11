import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import sqlite3
import sys
from config import load_config
from knowledge_base import OfflineKnowledgeBase
from answer_service import ConversationManager
from openai_fallback import OpenAIClient
from metrics import memory_usage


def check(config):
    connection = sqlite3.connect(':memory:')
    try:
        connection.execute('CREATE VIRTUAL TABLE probe USING fts5(text)')
        fts = True
    except sqlite3.Error:
        fts = False
    finally:
        connection.close()
    report = {'python': platform.python_version(), 'platform': platform.platform(), 'architecture': platform.machine(),
        'fts5': fts, 'cloud_enabled': config.cloud_enabled, 'api_key_present': bool(os.environ.get('OPENAI_API_KEY')),
        'sounddevice_installed': importlib.util.find_spec('sounddevice') is not None,
        'vosk_installed': importlib.util.find_spec('vosk') is not None,
        'vosk_model_exists': bool(config.vosk_model and config.path(config.vosk_model).is_dir()),
        'piper_available': bool(shutil.which(config.piper_binary)),
        'piper_model_exists': bool(config.piper_model and config.path(config.piper_model).is_file()),
        'espeak_available': bool(shutil.which('espeak-ng') or shutil.which('espeak')),
        'knowledge_dir': str(config.path(config.knowledge_dir)), **memory_usage()}
    if report['sounddevice_installed']:
        try:
            import sounddevice
            report['audio_devices'] = [dict(device) for device in sounddevice.query_devices()]
        except Exception as error:
            report['audio_diagnostic'] = str(error)
    print(json.dumps(report, indent=2))
    return 0 if fts else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='Lightweight English document voice assistant')
    parser.add_argument('--mode', choices=('text', 'voice'), default='text')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--text', help='Answer one typed question')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--reindex', action='store_true', help='Rebuild all document passages')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--evaluate', help='JSON retrieval evaluation cases')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    kb = None
    try:
        config = load_config(args.config)
        if args.continuous:
            config.continuous = True
        if args.check:
            return check(config)
        kb = OfflineKnowledgeBase(config.path(config.knowledge_dir), config.path(config.database))
        stats = kb.index(force=args.reindex)
        for diagnostic in stats['errors']:
            print('Index diagnostic:', diagnostic, file=sys.stderr)
        if args.reindex:
            print(json.dumps(stats, indent=2))
            return 0 if not stats['errors'] else 1
        if args.evaluate:
            print(json.dumps(kb.evaluate(json.loads(config.path(args.evaluate).read_text(encoding='utf-8'))), indent=2))
            return 0
        manager = ConversationManager(kb, config, OpenAIClient(config))
        if args.mode == 'voice' and not args.text:
            from voice_io import Microphone, Transcriber, Speaker, VoiceSession
            cloud = OpenAIClient(config)
            session = VoiceSession(manager, Microphone(config), Transcriber(config, cloud), Speaker(config), config)
            session.run()
        else:
            while True:
                try:
                    question = args.text if args.text is not None else input('You: ')
                except (EOFError, KeyboardInterrupt):
                    break
                if question.strip() == '/metrics':
                    print(json.dumps({**manager.metrics, **memory_usage()}, indent=2))
                    continue
                reply = manager.respond(question)
                if reply.text:
                    print('Assistant:', reply.text)
                if reply.sources:
                    print('Source:', ', '.join(reply.sources))
                if args.metrics:
                    print(json.dumps({**manager.metrics, **memory_usage()}, indent=2))
                if args.text is not None or reply.command in ('exit', 'quit', 'stop assistant', 'goodbye'):
                    break
        if args.metrics and args.mode == 'voice':
            print(json.dumps({**manager.metrics, **memory_usage()}, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as error:
        print(f'Diagnostic: {error}', file=sys.stderr)
        return 1
    finally:
        if kb:
            kb.close()

if __name__ == '__main__':
    raise SystemExit(main())
