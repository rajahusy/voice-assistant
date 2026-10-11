"""Conservative evidence routing and short-term conversation memory."""
import re
import time
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
from dataclasses import dataclass
from knowledge_base import tokens
from openai_fallback import CloudError

@dataclass
class Reply:
    text: str
    route: str
    sources: tuple = ()
    command: str = ''

class ConversationManager:
    def __init__(self, knowledge_base, config, cloud=None):
        self.kb, self.config, self.cloud = knowledge_base, config, cloud
        self.history = deque(maxlen=config.history_turns)
        self.cloud_history = deque(maxlen=config.history_turns)
        self.topic = ''
        self.institutional = False
        self.pending_question = ''
        self.metrics = {}

    def resolve(self, question):
        followup = bool(re.search(r'\b(his|her|their|its|it|that|more)\b|^(and\b|what about\b)', question.lower()))
        if followup and not self.topic:
            return None
        return f'{question} {self.topic}' if followup else question

    def respond(self, question):
        started = time.perf_counter()
        self.metrics = dict(retrieval_seconds=0.0, transcription_seconds=0.0, ai_seconds=0.0, synthesis_seconds=0.0)
        question = question.strip()
        if not question:
            return self.finish(question, Reply('', 'silence'), started, remember=False)
        command = question.lower().strip(' .!?')
        if command in ('stop', 'exit', 'quit', 'stop assistant', 'goodbye'):
            return self.finish(question, Reply('Goodbye.' if command != 'stop' else '', 'command', command=command), started, remember=False)
        if self.pending_question:
            question = f'{self.pending_question} {question}'
            self.pending_question = ''
        clock_answer = self.clock_answer(question)
        if clock_answer:
            return self.finish(question, Reply(clock_answer, 'clock'), started)
        resolved = self.resolve(question)
        if resolved is None:
            return self.finish(question, Reply('Which person or topic do you mean?', 'clarify'), started, remember=False)
        followup = resolved != question
        institutional = any(re.search(r'\b' + re.escape(d) + r'\b', resolved.lower()) for d in self.config.domains) or (followup and self.institutional)
        if re.search(r'\b(hod|head of department)\b', question.lower()) and not any(d in resolved.lower() for d in ('computer', 'civil', 'mechanical', 'electrical', 'electronics', 'cse', 'ece', 'ise')):
            self.pending_question = question
            return self.finish(question, Reply('Which department do you mean?', 'clarify'), started, remember=False)
        if re.search(r'\b(current|today|latest|now|this year|last year)\b', question.lower()):
            return self.finish(question, self.online_reply(resolved, institutional), started)
        t = time.perf_counter()
        try:
            results = self.kb.search(resolved)
        except (sqlite3.Error, OSError, ValueError):
            self.metrics['retrieval_seconds'] = time.perf_counter() - t
            return self.finish(question, self.online_reply(resolved, institutional), started)
        self.metrics['retrieval_seconds'] = time.perf_counter() - t
        required = tokens(question)
        useful = []
        for entry, coverage, rank in results:
            available = set(tokens(entry.text))
            # Context tokens must not make a missing requested fact appear answered.
            covered = all(any(a.startswith(r) or r.startswith(a) for a in available) for r in required)
            if covered and coverage >= 0.65:
                useful.append(entry)
        if useful:
            entry = useful[0]
            if self.config.cloud_documents and not self.config.cloud_compose and self.cloud and len(useful) > 1:
                t = time.perf_counter()
                try:
                    entry = useful[self.cloud.select_evidence(resolved, [e.text for e in useful[:4]])]
                except (CloudError, ValueError, IndexError):
                    pass  # Offline extractive answer remains available.
                finally:
                    self.metrics['ai_seconds'] = time.perf_counter() - t
            answer = entry.answer or self.extract(entry.text, required, question)
            composed = False
            if self.config.cloud_compose and self.cloud:
                t = time.perf_counter()
                try:
                    answer, index = self.cloud.compose_from_evidence(resolved, [e.text for e in useful[:4]])
                    entry = useful[index]
                    composed = True
                except (CloudError, ValueError, TypeError, IndexError):
                    pass
                finally:
                    self.metrics['ai_seconds'] += time.perf_counter() - t
            if re.search(r'\bwho\b', question, re.I) and 'principal' in required:
                answer = re.split(r',\s*per\b', answer, maxsplit=1, flags=re.I)[0].strip()
            if entry.as_of and entry.as_of.lower() != 'evergreen' and not composed:
                answer = f'As of {entry.as_of}, {answer}'
            reply = Reply(answer, 'local', (entry.source,) + tuple(entry.source_urls))
            self.topic = ' '.join(tokens(question if not followup else self.topic))
            self.institutional = institutional
        else:
            reply = self.online_reply(resolved, institutional)
        return self.finish(question, reply, started)

    def online_reply(self, question, institutional=False):
        if not self.config.cloud_enabled or not self.cloud:
            return Reply("I couldn't verify that in my documents. Online answers are disabled.", 'unverified')
        started = time.perf_counter()
        try:
            text = self.cloud.answer(question, self.cloud_history)
            if not isinstance(text, str) or not text.strip():
                raise CloudError('OpenAI returned an empty answer')
            self.topic = ' '.join(tokens(question))
            self.institutional = institutional
            sources = getattr(self.cloud, 'answer_sources', ())
            return Reply(text, 'cloud', tuple(sources) if isinstance(sources, (tuple, list)) else ())
        except (CloudError, ValueError) as error:
            return Reply(f"Online answers are unavailable: {error}. Local document questions still work.", 'offline')
        finally:
            self.metrics['ai_seconds'] += time.perf_counter() - started

    @staticmethod
    def clock_answer(question):
        # Handle clock questions before pronoun resolution and freshness routing.
        q = question.lower().replace('\u2019', "'")
        date_query = re.search(r"\b(?:what(?:'s| is)?|tell me)\b.*\b(?:date|day)\b", q)
        time_query = re.search(r"\b(?:what(?:'s| is)?|tell me)\b.*\btime\b", q)
        if not (date_query or time_query):
            return ''
        # Avoid treating admissions dates, historical dates or time zones as clock requests.
        words = set(re.findall(r'[a-z]+', q))
        allowed = set('hi hello hey good morning afternoon evening please can could you tell me what s is the date day time today todays it now current of week and'.split())
        if words - allowed:
            return ''
        now = datetime.now(ZoneInfo('Asia/Kolkata'))
        parts = []
        if date_query:
            parts.append(f"Today is {now:%A}, {now.day} {now:%B %Y}.")
        if time_query:
            parts.append(f"The time is {now:%I:%M %p} India Standard Time.")
        return ' '.join(parts)

    def finish(self, question, reply, started, remember=True):
        if remember:
            self.history.append((question, reply.text))
            if reply.route == 'cloud':
                self.cloud_history.append((question, reply.text))
        self.metrics['total_seconds'] = time.perf_counter() - started
        return reply

    @staticmethod
    def extract(text, terms, question=''):
        # Remove source/index boilerplate; retain exact source sentences.
        qa = list(re.finditer(r'(?m)^Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)', text, re.S))
        if qa:
            best = max(qa, key=lambda m: sum(t in m.group(1).lower() for t in terms))
            if any(t in best.group(1).lower() for t in terms):
                return best.group(2).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()
                 and not re.match(r'^(=|DOC-|\[DOC-|Source|SOURCE|Keywords|Category|As.of|URL|http|Answer policy|DATA DISCREPANCY)', line.strip())]
        sentences = re.split(r'(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<=[.!?])\s+(?=[A-Z])|\n', '\n'.join(lines))
        def relevance(sentence):
            available = set(tokens(sentence))
            score = sum(t in available for t in terms)
            if 'hod' in terms and re.search(r'\b(?:HOD|Head)\b', sentence, re.I):
                score += 6
            return score
        ranked = sorted(enumerate(sentences), key=lambda pair: -relevance(pair[1]))
        if re.search(r'\bwho\b', question, re.I) and ('hod' in terms or 'principal' in terms) and ranked:
            relevant = ranked[0][1]
            name = re.search(r'\bDr\.\s+[A-Z][\w]*(?:\s+[A-Z][\w]*){0,5}', relevant)
            if name:
                return name.group(0).strip()
        selected = sorted(ranked[:3], key=lambda pair: pair[0])
        answer = ' '.join(s for _, s in selected).strip()
        return answer[:1000] or "I couldn't find a readable answer in that document."
