"""Disk-backed incremental FTS5 retrieval, with streaming simple-format parsing."""
import csv
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path
from document_parsers import KnowledgeEntry, chunk_text, document_entries, _question_answer_entry, _text_question_answer_entries

STOP = set('who what when where why how is are was were the a an of to for in on at by from with and about tell me please do does did can could would will you your his her their its it he she they us we has have had be been being more last year current now today'.split())
ALIASES = {'qualifications': 'qualification', 'fee': 'fees', 'admissions': 'admission', 'programs': 'courses', 'programmes': 'courses', 'course': 'courses', 'program': 'courses', 'programme': 'courses', 'offer': 'offers', 'offered': 'offers'}
DEPARTMENTS = {'cse': ('computer', 'science', 'engineering')}

def tokens(text):
    words = re.findall(r'[a-z0-9]+', text.lower())
    institution = bool(set(words) & {'ait', 'college', 'school', 'institution'})
    result = []
    for term in words:
        if term in STOP or len(term) <= 1:
            continue
        # AIT already identifies the institution; "college" is redundant here.
        if term == 'college' and 'ait' in words:
            continue
        # Common typing/STT confusion, limited to institutional questions.
        if term == 'principle' and institution:
            term = 'principal'
        result.extend(DEPARTMENTS.get(term, (ALIASES.get(term, term),)))
    return list(dict.fromkeys(result))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def read_entries(path, source):
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield from records(json.loads(line), source)
    elif suffix == '.json':
        # One document at a time; JSON arrays require a bounded input size.
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('JSON exceeds 16 MiB; convert it to streaming JSONL')
        yield from records(json.loads(path.read_text(encoding='utf-8')), source)
    elif suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                yield from records(row, source)
    elif suffix == '.pdf':
        from pypdf import PdfReader
        for page in PdfReader(path).pages:
            yield from document_entries(page.extract_text() or '', source)
    elif suffix == '.docx':
        # DOCX is a ZIP of XML. No python-docx/lxml dependency required.
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as archive:
            with archive.open('word/document.xml') as stream:
                for _, element in ET.iterparse(stream, events=('end',)):
                    if element.tag.endswith('}p'):
                        text = ' '.join(e.text or '' for e in element.iter() if e.tag.endswith('}t'))
                        yield from document_entries(text, source)
                        element.clear()
    else:
        # Preserve existing heading/Q&A parser, using bounded blocks.
        with path.open(encoding='utf-8-sig', errors='replace') as stream:
            block = []
            for line in stream:
                if (line.startswith('[DOC-') or line.strip() == '---') and block:
                    text = ''.join(block)
                    yield from (_text_question_answer_entries(text, source) or document_entries(text, source))
                    block = []
                block.append(line)
                if sum(map(len, block)) >= 65536:
                    yield from document_entries(''.join(block), source)
                    block = block[-3:]  # small overlap across oversized sections
            if block:
                text = ''.join(block)
                yield from (_text_question_answer_entries(text, source) or document_entries(text, source))


def records(value, source):
    if isinstance(value, list):
        for item in value:
            yield from records(item, source)
    elif isinstance(value, dict) and 'question' in value and 'answer' in value:
        entry = _question_answer_entry(value, source)
        if entry:
            yield entry
    else:
        text = '\n'.join(f'{k}: {v}' for k, v in value.items()) if isinstance(value, dict) else str(value)
        yield from document_entries(text, source)


class OfflineKnowledgeBase:
    def __init__(self, directory, database):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(database, timeout=5)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS files(source TEXT PRIMARY KEY, stamp TEXT, digest TEXT);
            CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY, source TEXT, text TEXT, metadata TEXT, digest TEXT);
            CREATE INDEX IF NOT EXISTS passage_source ON passages(source);
            CREATE INDEX IF NOT EXISTS passage_digest ON passages(digest);
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text, content='passages', content_rowid='id', tokenize='porter unicode61');
            CREATE TRIGGER IF NOT EXISTS passage_insert AFTER INSERT ON passages BEGIN
              INSERT INTO search(rowid,text) VALUES(new.id,new.text); END;
            CREATE TRIGGER IF NOT EXISTS passage_delete AFTER DELETE ON passages BEGIN
              INSERT INTO search(search,rowid,text) VALUES('delete',old.id,old.text); END;
        ''')

    def close(self):
        self.conn.close()

    def index(self, force=False):
        stats = dict(updated=0, skipped=0, removed=0, duplicates=0, errors=[])
        seen = set()
        supported = {'.txt', '.md', '.json', '.jsonl', '.csv', '.pdf', '.docx'}
        for path in sorted(self.directory.rglob('*')):
            if not path.is_file() or any(p.startswith('.') for p in path.relative_to(self.directory).parts):
                continue
            source = path.relative_to(self.directory).as_posix()
            if path.suffix.lower() not in supported:
                stats['errors'].append(f'{source}: unsupported format (skipped)')
                continue
            seen.add(source)
            stat = path.stat()
            stamp = f'{stat.st_mtime_ns}:{stat.st_size}'
            old = self.conn.execute('SELECT * FROM files WHERE source=?', (source,)).fetchone()
            if old and old['stamp'] == stamp and not force:
                stats['skipped'] += 1
                continue
            try:
                digest = file_hash(path)
                if old and old['digest'] == digest and not force:
                    with self.conn:
                        self.conn.execute('UPDATE files SET stamp=? WHERE source=?', (stamp, source))
                    stats['skipped'] += 1
                    continue
                with self.conn:
                    self.conn.execute('DELETE FROM passages WHERE source=?', (source,))
                    hashes = set()
                    for entry in read_entries(path, source):
                        if not entry.text.strip():
                            continue
                        h = hashlib.sha256(entry.text.strip().encode()).hexdigest()
                        if h in hashes:
                            stats['duplicates'] += 1
                            continue
                        hashes.add(h)
                        metadata = asdict(entry)
                        self.conn.execute('INSERT INTO passages(source,text,metadata,digest) VALUES(?,?,?,?)',
                                          (source, entry.text, json.dumps(metadata), h))
                    self.conn.execute('INSERT OR REPLACE INTO files VALUES(?,?,?)', (source, stamp, digest))
                stats['updated'] += 1
            except Exception as error:
                # Per-file transaction rolls back; other documents remain available.
                logging.warning('Index failed for %s: %s', source, type(error).__name__)
                stats['errors'].append(f'{source}: {error}')
        for row in self.conn.execute('SELECT source FROM files').fetchall():
            if row['source'] not in seen:
                with self.conn:
                    self.conn.execute('DELETE FROM passages WHERE source=?', (row['source'],))
                    self.conn.execute('DELETE FROM files WHERE source=?', (row['source'],))
                stats['removed'] += 1
        return stats

    def search(self, question, limit=4):
        terms = tokens(question)
        if not terms:
            return []
        query = ' OR '.join('"' + term + '"' for term in terms[:24])
        rows = self.conn.execute('''SELECT p.*, bm25(search) AS rank FROM search
            JOIN passages p ON p.id=search.rowid WHERE search MATCH ? ORDER BY rank LIMIT ?''', (query, limit * 4)).fetchall()
        results, seen = [], set()
        for row in rows:
            if row['digest'] in seen:
                continue
            entry = KnowledgeEntry(**json.loads(row['metadata']))
            text_terms = set(tokens(entry.text))
            coverage = sum(t in text_terms or any(w.startswith(t) or t.startswith(w) for w in text_terms) for t in terms) / len(terms)
            heading_bonus = sum(t in tokens(entry.title or '') for t in terms)
            results.append((entry, coverage, row['rank'] - heading_bonus))
            seen.add(row['digest'])
        return sorted(results, key=lambda r: (-r[1], r[2]))[:limit]

    def evaluate(self, cases):
        hits = 0
        details = []
        for case in cases:
            matches = self.search(case['question'])
            good = any(case['source'] == item[0].source for item in matches)
            hits += good
            details.append({'question': case['question'], 'hit': good})
        return {'cases': len(cases), 'recall_at_4': hits / len(cases) if cases else None, 'details': details}
