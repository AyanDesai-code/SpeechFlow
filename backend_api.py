"""Minimal HTTP API to connect the frontend with SpeechFlow analysis.

This is intentionally additive and does not modify terminal workflow in conversation_engine.py.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable


HOST = "0.0.0.0"
PORT = 8787
FILLER_WORDS = {"uh", "um", "erm", "ah", "uhh", "umm"}


def has_prefix_stutter(word: str) -> bool:
    normalized = word.lower().replace("-", "")
    if len(normalized) < 6:
        return False

    for size in (1, 2, 3):
        prefix = normalized[:size]
        if normalized.startswith(prefix * 3):
            return True
    return False


def has_hyphen_stutter(word: str) -> bool:
    parts = word.split("-")
    return len(parts) >= 3 and all(part == parts[0] for part in parts[:-1])


def valid_practice_word(word: str) -> bool:
    return word not in FILLER_WORDS and word.replace("-", "").isalpha() and len(word) >= 4


def _extract_words(lines: Iterable[dict]) -> list[str]:
    words: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        if line.get("role", "").lower() != "user":
            continue
        text = str(line.get("text", "")).lower()
        words.extend(re.findall(r"[a-z]+(?:-[a-z]+)*", text))
    return words


def analyze_transcript(lines: Iterable[dict]) -> dict:
    words = _extract_words(lines)
    repetition_counter: Counter[str] = Counter()

    if not words:
        return {
            "practiceWords": {},
            "summary": "No user transcript was provided, so there are no focus words yet."
        }

    raw_text = " ".join(words)

    matches = re.findall(r"\b(\w+)( \1){1,}", raw_text)
    for match in matches:
        repetition_counter[match[0]] += 2

    for word in words:
        if has_prefix_stutter(word):
            repetition_counter[word] += 3
        if has_hyphen_stutter(word):
            repetition_counter[word] += 3

    idx = 0
    while idx < len(words):
        run_length = 1
        while idx + run_length < len(words) and words[idx] == words[idx + run_length]:
            run_length += 1

        if run_length >= 2:
            repetition_counter[words[idx]] += run_length

        idx += run_length

    for i in range(len(words) - 3):
        if words[i : i + 2] == words[i + 2 : i + 4]:
            phrase = " ".join(words[i : i + 2])
            repetition_counter[phrase] += 2

    practice_words = {
        word: count
        for word, count in repetition_counter.most_common()
        if count >= 2 and valid_practice_word(word)
    }

    if not practice_words:
        summary = "Nice consistency. Keep practicing smooth pacing and clear enunciation."
    else:
        top_word = next(iter(practice_words))
        summary = f"Focus on '{top_word}' first: say it slowly 5 times, then use it in two short sentences."

    return {
        "practiceWords": practice_words,
        "summary": summary,
    }


class SpeechFlowHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "mode": "live"})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze-transcript":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len).decode("utf-8")
            payload = json.loads(raw) if raw else {}
            transcript = payload.get("transcript", [])
            analysis = analyze_transcript(transcript)
            self._send_json(200, analysis)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Body must be valid JSON"})
        except Exception as exc:  # defensive API guard
            self._send_json(500, {"error": f"Unexpected server error: {exc}"})


def run_server(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), SpeechFlowHandler)
    print(f"SpeechFlow API server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
