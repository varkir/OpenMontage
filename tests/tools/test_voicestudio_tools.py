"""VoiceStudio TTS and STT tools.

Runs against a real local HTTP server (no monkeypatching of the network layer).
The server records each request so the tests can check what the tools sent.
"""

import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import BaseTool, ToolStatus, ToolTier  # noqa: E402
from tools.audio.voicestudio_tts import VoiceStudioTTS  # noqa: E402
from tools.analysis.voicestudio_stt import VoiceStudioSTT  # noqa: E402


def _wav_bytes(seconds=0.5, rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class _Server:
    """Serves /v1/audio/speech and /v1/audio/transcriptions with canned replies."""

    def __init__(self):
        self.requests = []
        self.speech_body = _wav_bytes()
        self.speech_status = 200
        self.speech_type = "audio/wav"
        self.stt_reply = {"text": "hello world"}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _record(self, body):
                owner.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                self._record(body)
                if self.path == "/v1/audio/speech":
                    self.send_response(owner.speech_status)
                    self.send_header("Content-Type", owner.speech_type)
                    self.end_headers()
                    self.wfile.write(owner.speech_body)
                elif self.path == "/v1/audio/transcriptions":
                    payload = json.dumps(owner.stt_reply).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


@pytest.fixture
def vs_env(monkeypatch, server):
    monkeypatch.setenv("VOICESTUDIO_BASE_URL", server.url)
    monkeypatch.setenv("VOICESTUDIO_API_KEY", "secret-key")


# ---- TTS: contract ----

class TestTTSContract:
    def test_inherits_base_tool_and_identity(self):
        assert issubclass(VoiceStudioTTS, BaseTool)
        assert VoiceStudioTTS.name == "voicestudio_tts"
        assert VoiceStudioTTS.capability == "tts"
        assert VoiceStudioTTS.provider == "voicestudio"
        assert VoiceStudioTTS.tier == ToolTier.VOICE

    def test_text_is_required(self):
        assert "text" in VoiceStudioTTS.input_schema["required"]

    def test_status_unavailable_without_config(self, monkeypatch):
        monkeypatch.delenv("VOICESTUDIO_BASE_URL", raising=False)
        monkeypatch.delenv("VOICESTUDIO_API_KEY", raising=False)
        assert VoiceStudioTTS().get_status() == ToolStatus.UNAVAILABLE

    def test_status_available_with_base_url_and_key(self, vs_env):
        assert VoiceStudioTTS().get_status() == ToolStatus.AVAILABLE


# ---- TTS: execute ----

class TestTTSExecute:
    def test_writes_wav_and_sends_voice_and_bearer(self, vs_env, server, tmp_path):
        out = tmp_path / "narration" / "line.wav"
        result = VoiceStudioTTS().execute({"text": "The river ran red.", "voice": "en-stone", "language": "en", "output_path": str(out)})

        assert result.success, result.error
        assert out.read_bytes()[:4] == b"RIFF"
        assert result.data["provider"] == "voicestudio"
        assert result.data["voice"] == "en-stone"
        assert str(out) in result.artifacts
        request = server.requests[-1]
        assert request["path"] == "/v1/audio/speech"
        assert request["headers"]["Authorization"] == "Bearer secret-key"
        body = json.loads(request["body"])
        assert body["voice"] == "en-stone"
        assert body["input"] == "The river ran red."
        assert body["language"] == "en"
        assert body["response_format"] == "wav"

    def test_non_wav_reply_is_a_failure(self, vs_env, server, tmp_path):
        server.speech_body = b'{"error": "profile not found"}'
        server.speech_type = "application/json"
        out = tmp_path / "x.wav"
        result = VoiceStudioTTS().execute({"text": "hello", "voice": "missing", "output_path": str(out)})

        assert not result.success
        assert "not a WAV" in result.error
        assert not out.exists()

    def test_http_error_is_a_failure(self, vs_env, server, tmp_path):
        server.speech_status = 401
        server.speech_body = b"unauthorized"
        server.speech_type = "text/plain"
        result = VoiceStudioTTS().execute({"text": "hello", "voice": "en-stone", "output_path": str(tmp_path / "x.wav")})

        assert not result.success
        assert "401" in result.error

    def test_missing_config_fails_without_network(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VOICESTUDIO_BASE_URL", raising=False)
        result = VoiceStudioTTS().execute({"text": "hello", "voice": "en-stone", "output_path": str(tmp_path / "x.wav")})

        assert not result.success
        assert "VOICESTUDIO_BASE_URL" in result.error


# ---- STT ----

class TestSTTContract:
    def test_inherits_base_tool_and_identity(self):
        assert issubclass(VoiceStudioSTT, BaseTool)
        assert VoiceStudioSTT.name == "voicestudio_stt"
        assert VoiceStudioSTT.capability == "analysis"
        assert VoiceStudioSTT.provider == "voicestudio"
        assert "transcribe" in VoiceStudioSTT.capabilities

    def test_input_path_is_required(self):
        assert "input_path" in VoiceStudioSTT.input_schema["required"]

    def test_status_unavailable_without_config(self, monkeypatch):
        monkeypatch.delenv("VOICESTUDIO_BASE_URL", raising=False)
        assert VoiceStudioSTT().get_status() == ToolStatus.UNAVAILABLE


class TestSTTExecute:
    def test_posts_file_and_returns_text_and_segment(self, vs_env, server, tmp_path):
        audio = tmp_path / "line.wav"
        audio.write_bytes(_wav_bytes())
        result = VoiceStudioSTT().execute({"input_path": str(audio), "language": "en", "output_path": str(tmp_path / "t.json")})

        assert result.success, result.error
        assert result.data["text"] == "hello world"
        assert result.data["language"] == "en"
        assert result.data["segments"][0]["text"] == "hello world"
        request = server.requests[-1]
        assert request["path"] == "/v1/audio/transcriptions"
        assert request["headers"]["Authorization"] == "Bearer secret-key"
        assert request["headers"]["Content-Type"].startswith("multipart/form-data")
        assert b'name="model"' in request["body"]
        assert b'name="language"' in request["body"]
        assert b'name="file"' in request["body"]
        assert b"RIFF" in request["body"]

    def test_requests_verbose_json_and_keeps_segment_timings(self, vs_env, server, tmp_path):
        server.stt_reply = {
            "text": "one two",
            "language": "en",
            "segments": [{"start": 0.0, "end": 0.8, "text": "one"}, {"start": 0.8, "end": 1.6, "text": "two"}],
        }
        audio = tmp_path / "line.wav"
        audio.write_bytes(_wav_bytes())
        result = VoiceStudioSTT().execute({"input_path": str(audio), "output_path": str(tmp_path / "t.json")})

        assert result.success, result.error
        assert [s["end"] for s in result.data["segments"]] == [0.8, 1.6]
        assert b"verbose_json" in server.requests[-1]["body"]

    def test_missing_file_fails(self, vs_env, tmp_path):
        result = VoiceStudioSTT().execute({"input_path": str(tmp_path / "nope.wav")})

        assert not result.success
        assert "not found" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class _FakeTool:
    def __init__(self, name, status, text):
        self.name = name
        self._status = status
        self._text = text
        self.calls = []

    def get_status(self):
        from types import SimpleNamespace
        return SimpleNamespace(value=self._status)

    def execute(self, inputs):
        from types import SimpleNamespace
        self.calls.append(inputs)
        return SimpleNamespace(success=True, data={"text": self._text})


class _FakeRegistry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name):
        return self._tools.get(name)


class TestSourceMediaTranscriberOrder:
    def _run(self, registry, tmp_path):
        from lib.source_media_review import _transcribe_if_available
        media = tmp_path / "clip.mp4"
        media.write_bytes(b"x")
        return _transcribe_if_available(media, "video", registry)

    def test_prefers_voicestudio_when_available(self, tmp_path):
        vs = _FakeTool("voicestudio_stt", "available", "from voicestudio")
        local = _FakeTool("transcriber", "available", "from whisper")
        text = self._run(_FakeRegistry([vs, local]), tmp_path)

        assert text == "from voicestudio"
        assert len(vs.calls) == 1
        assert local.calls == []

    def test_falls_back_to_transcriber_when_voicestudio_unavailable(self, tmp_path):
        vs = _FakeTool("voicestudio_stt", "unavailable", "unused")
        local = _FakeTool("transcriber", "available", "from whisper")
        text = self._run(_FakeRegistry([vs, local]), tmp_path)

        assert text == "from whisper"
        assert vs.calls == []
