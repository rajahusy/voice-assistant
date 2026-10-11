import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from config import Config
from knowledge_base import OfflineKnowledgeBase
from answer_service import ConversationManager
from openai_fallback import CloudError, OpenAIClient
from voice_io import VoiceSession, State, Transcriber, SpeechError, Microphone

class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / 'docs'; self.docs.mkdir()
        (self.docs / 'college.json').write_text(json.dumps([
            {'question':'Who is the principal of AIT?', 'answer':'The principal is Dr. Example.'},
            {'question':'What is the qualification of the principal of AIT?', 'answer':'The principal holds a PhD.'},
            {'question':'What is admission at AIT?', 'answer':'Admission requires mathematics.'},
            {'question':'What are AIT fees?', 'answer':'Fees are not specified in the documents.'}
        ]))
        self.kb = OfflineKnowledgeBase(self.docs, self.root / 'db.sqlite')
        self.kb.index()
        self.cfg = Config(cooldown_seconds=0)
        self.cloud = Mock()
        self.manager = ConversationManager(self.kb, self.cfg, self.cloud)

    def tearDown(self):
        self.kb.close(); self.temp.cleanup()

    def test_A_clear_direct_answer(self):
        r = self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(r.text, 'The principal is Dr. Example.')
        self.assertEqual(r.route, 'local'); self.cloud.answer.assert_not_called()

    def test_B_pronoun_followup(self):
        self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(self.manager.respond('What is his qualification?').text, 'The principal holds a PhD.')

    def test_principle_spelling_and_redundant_college(self):
        for question in ('who is the principle of AIT college',
                         'Who is the principal of AIT college?',
                         'Who is the principle of AIT?'):
            with self.subTest(question=question):
                reply = self.manager.respond(question)
                self.assertEqual(reply.route, 'local')
                self.assertEqual(reply.text, 'The principal is Dr. Example.')
        self.cloud.answer.assert_not_called()

    def test_general_principle_is_preserved(self):
        from knowledge_base import tokens
        self.assertIn('principle', tokens('Explain the principle of relativity'))
        self.assertNotIn('principal', tokens('Explain the principle of relativity'))

    def test_C_ambiguous_followup_and_hod(self):
        self.assertEqual(self.manager.respond('What about his qualification?').route, 'clarify')
        self.assertEqual(self.manager.respond('Who is the HOD?').text, 'Which department do you mean?')

    def test_D_silence_no_output(self):
        self.assertEqual(self.manager.respond(' ').text, '')
        session, mic, stt, tts = self.session()
        mic.record.return_value = b''
        self.assertIsNone(session.step()); stt.transcribe.assert_not_called(); tts.speak.assert_not_called()

    def test_E_repeat_without_greeting(self):
        first = self.manager.respond('Who is the principal of AIT?')
        second = self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(first.text, second.text); self.assertEqual(len(self.manager.history), 2)

    def test_F_missing_fact_does_not_go_online(self):
        self.cfg.cloud_enabled = True
        self.assertEqual(self.manager.respond('What is the principal salary at AIT?').route, 'unverified')
        self.cloud.answer.assert_not_called()

    def test_G_disconnect_offline_available(self):
        self.cfg.cloud_enabled = True
        self.cloud.answer.side_effect = CloudError('offline')
        self.assertEqual(self.manager.respond('Explain quantum entanglement').route, 'offline')
        self.assertEqual(self.manager.respond('Who is the principal of AIT?').route, 'local')

    def test_H_missing_stt_and_configured_fallback(self):
        cfg = Config(vosk_model='not-a-real-model')
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(SpeechError):
                Transcriber(cfg, self.cloud)
        cfg.cloud_enabled = True
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-test-key'}):
            stt = Transcriber(cfg, self.cloud)
        self.assertEqual(stt.provider, 'openai')

    def session(self):
        mic, stt, tts = Mock(), Mock(), Mock()
        mic.record.return_value = b'audio'
        stt.transcribe.return_value = 'Who is the principal of AIT?'
        return VoiceSession(self.manager, mic, stt, tts, self.cfg), mic, stt, tts

    def test_I_stop_and_exit(self):
        session, _, stt, tts = self.session()
        stt.transcribe.return_value = 'stop'
        self.assertEqual(session.step().command, 'stop'); tts.stop.assert_called_once(); tts.speak.assert_not_called()
        self.assertEqual(self.manager.respond('exit').command, 'exit')

    def test_J_feedback_suppressed_and_state_order(self):
        session, mic, stt, tts = self.session()
        mic.record.side_effect = lambda: (self.assertEqual(session.state, State.LISTENING) or b'audio')
        tts.speak.side_effect = lambda _: self.assertEqual(session.state, State.SPEAKING)
        session.step()
        stt.transcribe.return_value = session.last_answer
        self.assertIsNone(session.step()); self.assertEqual(tts.speak.call_count, 1)
        self.assertEqual(session.state, State.IDLE)

    def test_K_restart_incremental_and_modified_deleted(self):
        self.kb.close()
        self.kb = OfflineKnowledgeBase(self.docs, self.root / 'db.sqlite')
        self.assertEqual(self.kb.index()['updated'], 0)
        (self.docs / 'new.txt').write_text('The robotics laboratory opens at nine.')
        self.assertEqual(self.kb.index()['updated'], 1)
        (self.docs / 'new.txt').unlink()
        self.assertEqual(self.kb.index()['removed'], 1)

    def test_L_text_without_audio_import(self):
        with patch.dict('sys.modules', {'sounddevice': None, 'vosk': None}):
            self.assertEqual(self.manager.respond('Who is the principal of AIT?').route, 'local')
            with self.assertRaises(SpeechError):
                Microphone(self.cfg)

    def test_incomplete_fees_and_admission(self):
        self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(self.manager.respond('What about admission?').text, 'Admission requires mathematics.')
        self.assertEqual(self.manager.respond('And the fees?').text, 'Fees are not specified in the documents.')

    def test_department_clarification_is_consumed(self):
        (self.docs / 'hod.json').write_text(json.dumps({'question':'Who is the CSE HOD?', 'answer':'The CSE HOD is Dr. Sample.'}))
        self.kb.index()
        self.assertEqual(self.manager.respond('Who is the HOD?').route, 'clarify')
        self.assertEqual(self.manager.respond('CSE').text, 'The CSE HOD is Dr. Sample.')

    def test_cloud_passages_require_separate_consent(self):
        self.cfg.cloud_enabled = True
        self.manager.respond('Who is the principal of AIT?')
        self.cloud.select_evidence.assert_not_called()
        self.cfg.cloud_documents = True
        self.cloud.select_evidence.return_value = 0
        self.assertEqual(self.manager.respond('Who is the principal of AIT?').route, 'local')
        self.cloud.select_evidence.assert_called_once()
        self.cloud.select_evidence.side_effect = CloudError('offline')
        self.assertEqual(self.manager.respond('Who is the principal of AIT?').route, 'local')

    def test_voice_failure_bounded(self):
        session, mic, _, tts = self.session()
        self.cfg.continuous = True
        mic.record.side_effect = SpeechError('hardware disconnected')
        with patch('time.sleep'), patch('builtins.print') as output:
            session.run()
        self.assertEqual(mic.record.call_count, 2)
        self.assertEqual(sum('Diagnostic:' in str(c) for c in output.call_args_list), 1)
        tts.stop.assert_called_once()
        self.assertEqual(session.state, State.IDLE)

    def test_broad_courses_question_prefers_complete_overview(self):
        (self.docs / 'courses.json').write_text(json.dumps([
            {'question':'What courses does AIT offer?', 'answer':'AIT offers Bachelor of Engineering, Master of Technology, MBA, and PhD programs.', 'keywords':'courses offered programs overview'},
            {'question':'What MBA does AIT offer?', 'answer':'AIT offers an MBA in Finance. Course eligibility requires a degree.'}
        ]))
        self.kb.index()
        for question in ('What are the courses AIT College offers?', 'What courses are offered at AIT?', 'What programs does AIT offer?'):
            with self.subTest(question=question):
                answer = self.manager.respond(question).text
                self.assertIn('Bachelor of Engineering', answer)
                self.assertIn('MBA', answer)

    def test_hod_question_extracts_person_instead_of_department_overview(self):
        text = ("AIT's Department of Computer Science and Engineering was established in 1986. "
                "The department supports research and conferences.\n"
                "The faculty-achievements page identifies Dr. Pushpa Ravikumar as the CSE Head and lists intake figures.")
        from knowledge_base import tokens
        question = 'Who is the HOD of the CSE department?'
        self.assertEqual(self.manager.extract(text, tokens(question), question), 'Dr. Pushpa Ravikumar')

    def test_principal_response_omits_source_boilerplate(self):
        (self.docs / 'college.json').write_text(json.dumps({'question':'Who is the principal of AIT?',
            'answer':"Dr. Example, per AIT's public self-disclosure accessed 2026-10-10."}))
        self.kb.index()
        self.assertEqual(self.manager.respond('Who is the principal of AIT college?').text, 'Dr. Example')

    def test_failed_llm_composition_falls_back_to_documents(self):
        self.cfg.cloud_enabled = self.cfg.cloud_documents = self.cfg.cloud_compose = True
        self.cloud.compose_from_evidence.side_effect = CloudError('offline')
        reply = self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(reply.text, 'The principal is Dr. Example.')
        self.cloud.answer.assert_not_called()

    def test_reported_mishearing_is_corrected_and_answered_locally(self):
        session, _, stt, tts = self.session()
        stt.transcribe.return_value = 'Hi, I asked, who is the principal of AAT Golan?'
        with patch('builtins.print'):
            reply = session.step()
        self.assertEqual(reply.route, 'local')
        self.assertEqual(reply.text, 'The principal is Dr. Example.')
        tts.speak.assert_called_once_with('The principal is Dr. Example.')
        self.cloud.answer.assert_not_called()

    def test_listening_display_is_not_spoken_or_repeated_on_silence(self):
        session, mic, stt, tts = self.session()
        mic.record.side_effect = [b'', b'', b'question', b'exit']
        stt.transcribe.side_effect = ['Who is the principal of AIT?', 'exit']
        with patch('builtins.print') as display:
            session.run()
        status_calls = [c for c in display.call_args_list if c.args == ('Listening...',)]
        self.assertEqual(len(status_calls), 2)
        self.assertFalse(any('Listening' in str(c) for c in tts.speak.call_args_list))

    def test_transcript_normalization_preserves_other_colleges_and_general_terms(self):
        from voice_io import normalize_transcript
        for text in ('Who is the principal of MIT college?', 'Explain the principle of relativity'):
            self.assertEqual(normalize_transcript(text, self.cfg), text)
        self.assertEqual(normalize_transcript('Who is the princilple of A I T college?', self.cfg),
                         'Who is the principal of AIT college?')

    def test_empty_transcript_no_answer(self):
        session, _, stt, tts = self.session()
        stt.transcribe.return_value = ''
        self.assertIsNone(session.step()); tts.speak.assert_not_called()

    def test_voice_listens_without_keyboard_and_returns_after_reply(self):
        session, mic, stt, tts = self.session()
        # Old configuration values must not bring back the keyboard prompt.
        self.cfg.continuous = False
        mic.record.side_effect = [b'', b'question-audio', b'exit-audio']
        stt.transcribe.side_effect = ['Who is the principal of AIT?', 'exit']
        with patch('builtins.input', side_effect=AssertionError('Keyboard input is forbidden')) as keyboard:
            session.run()
        keyboard.assert_not_called()
        self.assertEqual(mic.record.call_count, 3)
        self.assertEqual(tts.speak.call_count, 2)  # One greeting and one answer.
        tts.speak.assert_any_call(self.cfg.startup_greeting)
        tts.speak.assert_any_call('The principal is Dr. Example.')
        self.assertEqual(session.state, State.IDLE)

    def test_memory_bounded(self):
        for _ in range(20): self.manager.respond('Who is the principal of AIT?')
        self.assertEqual(len(self.manager.history), 6)

    def test_local_history_not_sent_to_general_cloud(self):
        self.cfg.cloud_enabled = True
        self.manager.respond('Who is the principal of AIT?')
        self.cloud.answer.return_value = 'A quantum answer.'
        self.manager.respond('Explain quantum entanglement')
        self.assertNotIn('Dr. Example', str(self.cloud.answer.call_args))
        self.assertEqual(len(self.manager.cloud_history), 1)

    def test_current_information_is_not_invented(self):
        self.cfg.cloud_enabled = True
        self.assertEqual(self.manager.respond('What are the latest fees at AIT?').route, 'unverified')
        self.cloud.answer.assert_not_called()

    def test_formats_duplicates_and_rollback(self):
        (self.docs / 'notes.md').write_text('Robotics uses sensors.')
        (self.docs / 'notes.csv').write_text('name,value\nweather,sunny\n')
        (self.docs / 'notes.jsonl').write_text('{"question":"What is a sensor?","answer":"A sensor detects changes."}\n')
        (self.docs / 'copy.md').write_text('Robotics uses sensors.')
        with zipfile.ZipFile(self.docs / 'notes.docx', 'w') as doc:
            doc.writestr('word/document.xml', '<w:document xmlns:w="urn:word"><w:p><w:r><w:t>A robot can move.</w:t></w:r></w:p></w:document>')
        self.assertFalse(self.kb.index()['errors'])
        self.assertEqual(sum(entry.text == 'Robotics uses sensors.' for entry, _, _ in self.kb.search('Robotics sensors')), 1)
        self.assertTrue(self.kb.search('robot move'))
        (self.docs / 'college.json').write_text('{invalid')
        self.assertTrue(self.kb.index()['errors'])
        self.assertEqual(self.manager.respond('Who is the principal of AIT?').route, 'local')

    def test_retrieval_evaluation(self):
        report = self.kb.evaluate([{'question':'Who is the principal of AIT?', 'source':'college.json'}])
        self.assertEqual(report['recall_at_4'], 1)

class MicrophoneBufferTests(unittest.TestCase):
    def microphone(self, overflow_flags, sample=1000):
        import struct
        cfg = Config(sample_rate=1000, max_recording=0.25, listen_timeout=0.25)
        mic = Microphone.__new__(Microphone)
        mic.config = cfg
        stream = Mock()
        stream.read.side_effect = [(struct.pack('<50h', *([sample] * 50)), flag) for flag in overflow_flags]
        context = Mock()
        context.__enter__ = Mock(return_value=stream)
        context.__exit__ = Mock(return_value=False)
        mic.sd = Mock()
        mic.sd.RawInputStream.return_value = context
        return mic

    def test_transient_overflow_keeps_recording(self):
        import wave
        mic = self.microphone([True, False, False, False, False])
        with self.assertLogs(level='WARNING'):
            audio = mic.record()
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            self.assertEqual(wav.getnframes(), 250)
        self.assertEqual(mic.sd.RawInputStream.call_args.kwargs['blocksize'], 0)
        self.assertEqual(mic.sd.RawInputStream.call_args.kwargs['latency'], 0.2)

    def test_repeated_overflow_is_bounded(self):
        mic = self.microphone([True] * 5)
        with self.assertLogs(level='WARNING'), self.assertRaisesRegex(SpeechError, 'repeatedly'):
            mic.record()

    def test_buffered_silence_remains_silent(self):
        mic = self.microphone([False] * 5, sample=0)
        self.assertEqual(mic.record(), b'')


class NeuralSpeechTests(unittest.TestCase):
    def audio(self):
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(bytes(480))
        return buffer.getvalue()

    def test_cloud_speech_uses_binary_response_and_configured_voice(self):
        client = OpenAIClient(Config(cloud_enabled=True))
        client.request = Mock(return_value=bytes(480))
        self.assertEqual(client.synthesize('Welcome.'), self.audio())
        payload = json.loads(client.request.call_args.args[1])
        self.assertEqual(payload['voice'], 'marin')
        self.assertEqual(payload['response_format'], 'pcm')
        self.assertTrue(client.request.call_args.kwargs['raw'])
        client.request.return_value = b'not audio'
        with self.assertRaises(CloudError): client.synthesize('Welcome.')

    def test_neural_failure_falls_back_once_per_session(self):
        from voice_io import Speaker
        cfg = Config(tts='openai', cloud_enabled=True)
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake-test'}):
            speaker = Speaker(cfg)
        speaker.cloud = Mock()
        speaker.cloud.synthesize.side_effect = CloudError('offline')
        speaker.run = Mock()
        with patch('shutil.which', return_value='/fake/espeak'), patch('builtins.print') as output:
            speaker.speak('Dr. Example')
            speaker.speak('Welcome')
        self.assertEqual(speaker.provider, 'espeak')
        self.assertEqual(speaker.cloud.synthesize.call_count, 1)
        self.assertEqual(speaker.run.call_count, 2)
        self.assertEqual(output.call_count, 1)
        self.assertEqual(speaker.run.call_args_list[0].args[1], 'Doctor Example.')

    def test_neural_playback_and_empty_text_silent(self):
        from voice_io import Speaker
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake-test'}):
            speaker = Speaker(Config(tts='openai', cloud_enabled=True))
        speaker.cloud = Mock(); speaker.cloud.synthesize.return_value = self.audio()
        speaker.play_wav = Mock()
        speaker.speak(' ')
        speaker.cloud.synthesize.assert_not_called()
        speaker.speak('Hello')
        speaker.play_wav.assert_called_once_with(self.audio())

    def test_default_output_uses_pipewire_for_bluetooth_routing(self):
        from voice_io import Speaker
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake-test'}):
            speaker = Speaker(Config(tts='openai', cloud_enabled=True, output_device=None))
        speaker.run = Mock()
        with patch('shutil.which', return_value='/usr/bin/pw-play'):
            speaker.play_wav(self.audio())
        self.assertEqual(speaker.run.call_args.args[0][0], 'pw-play')

    def test_request_can_return_wav_without_json_decoding(self):
        wav = self.audio()
        response = Mock()
        response.read.return_value = wav
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake-test'}), patch('urllib.request.urlopen', return_value=response):
            self.assertEqual(OpenAIClient(Config(cloud_enabled=True)).request('audio/speech', b'{}', raw=True), wav)


class EnvironmentConfigTests(unittest.TestCase):
    def test_env_literal_values_and_existing_environment_precedence(self):
        from config import load_env
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('# comment\nOPENAI_API_KEY="fake-local"\nOPENAI_MODEL=gpt-4o-mini\nIGNORED=value\n')
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-existing'}, clear=True):
                load_env(path)
                import os
                self.assertEqual(os.environ['OPENAI_API_KEY'], 'fake-existing')
                self.assertEqual(os.environ['OPENAI_MODEL'], 'gpt-4o-mini')
                self.assertNotIn('IGNORED', os.environ)

    def test_default_local_config_and_explicit_offline_override(self):
        from config import load_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('OPENAI_API_KEY=fake-test\nOPENAI_MODEL=test-model\n')
            (root / 'config.local.json').write_text('{"cloud_enabled": true}')
            explicit = root / 'offline.json'; explicit.write_text('{"cloud_enabled": false}')
            with patch('config.ROOT', root), patch.dict('os.environ', {}, clear=True):
                self.assertTrue(load_config().cloud_enabled)
                self.assertEqual(load_config().model, 'test-model')
                self.assertFalse(load_config(explicit).cloud_enabled)


class CloudTests(unittest.TestCase):
    def test_disabled_never_sends(self):
        with patch('urllib.request.urlopen') as send:
            with self.assertRaises(CloudError): OpenAIClient(Config()).request('responses', b'{}')
            send.assert_not_called()

    def test_invalid_key_no_retry(self):
        import urllib.error
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake'}):
            with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError('url', 401, 'unauthorized', {}, None)) as send:
                with self.assertRaisesRegex(CloudError, 'Invalid'): OpenAIClient(Config(cloud_enabled=True)).request('responses', b'{}')
                self.assertEqual(send.call_count, 1)

    def test_connectivity_retry_bounded(self):
        import urllib.error
        with patch.dict('os.environ', {'OPENAI_API_KEY':'fake'}), patch('time.sleep'):
            with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('offline')) as send:
                with self.assertRaises(CloudError): OpenAIClient(Config(cloud_enabled=True)).request('responses', b'{}')
                self.assertEqual(send.call_count, 2)

    def test_llm_compose_accepts_only_a_valid_source_span(self):
        client = OpenAIClient(Config(cloud_enabled=True, cloud_documents=True, cloud_compose=True))
        def response(value):
            return {'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
        client.request = Mock(return_value=response({'passage_index':0,'quote':'Dr. Example'}))
        self.assertEqual(client.compose_from_evidence('Who is the principal?', ['The principal is Dr. Example.']), ('Dr. Example',0))
        for invalid in ({'passage_index':0,'quote':'Dr. Invented'}, {'passage_index':9,'quote':'Dr. Example'},
                        {'passage_index':-1,'quote':''}, {'passage_index':0,'quote':'The principal is Dr. Example. Invented fact.'}):
            client.request.return_value = response(invalid)
            with self.assertRaises(CloudError): client.compose_from_evidence('Who?', ['The principal is Dr. Example.'])

    def test_cloud_evidence_index_validated(self):
        client = OpenAIClient(Config(cloud_enabled=True))
        client.request = Mock(return_value={'output':[{'content':[{'type':'output_text','text':'0'}]}]})
        self.assertEqual(client.select_evidence('Question', ['Passage']), 0)
        for invalid in ('invented answer', '-1', '100'):
            client.request.return_value = {'output':[{'content':[{'type':'output_text','text':invalid}]}]}
            with self.assertRaises(CloudError): client.select_evidence('Question', ['Passage'])

    def test_response_parse_and_transcription(self):
        client = OpenAIClient(Config(cloud_enabled=True))
        client.request = Mock(return_value={'output':[{'content':[{'type':'output_text','text':'A short answer.'}]}]})
        self.assertEqual(client.answer('Question', []), 'A short answer.')
        client.request.return_value = {'text':'English words'}
        self.assertEqual(client.transcribe(b'RIFF-test'), 'English words')
        self.assertIn(b'RIFF-test', client.request.call_args.args[1])
        self.assertIn(b'name="prompt"', client.request.call_args.args[1])
        self.assertIn(b'AIT College', client.request.call_args.args[1])

if __name__ == '__main__': unittest.main()
