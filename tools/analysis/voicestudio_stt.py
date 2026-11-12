"""VoiceStudio speech-to-text provider tool (local, OpenAI-style /v1/audio/transcriptions)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

REQUEST_TIMEOUT_SECONDS = 900


class VoiceStudioSTT(BaseTool):
    name = "voicestudio_stt"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "voicestudio"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = []
    install_instructions = (
        "Set the VoiceStudio address and key:\n"
        "  VOICESTUDIO_BASE_URL=http://<host>:3900\n"
        "  VOICESTUDIO_API_KEY=<key>"
    )
    fallback = "transcriber"
    fallback_tools = ["transcriber"]

    capabilities = ["transcribe", "language_detection"]
    supports = {"word_timestamps": False, "diarization": False, "offline": True}
    best_for = [
        "local transcript of a narration file to check the TTS result",
    ]
    not_good_for = [
        "word-level caption timing (use transcriber, which returns word timestamps)",
        "speaker labels",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string", "description": "Path to a WAV or other audio file"},
            "language": {"type": "string", "description": "ISO 639-1 code, or omit to let VoiceStudio detect"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["input_path", "language"]

    def get_status(self) -> ToolStatus:
        if os.environ.get("VOICESTUDIO_BASE_URL") and os.environ.get("VOICESTUDIO_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        base_url = os.environ.get("VOICESTUDIO_BASE_URL", "").rstrip("/")
        if not base_url:
            return ToolResult(success=False, error="VOICESTUDIO_BASE_URL is not set. " + self.install_instructions)

        audio_path = Path(inputs["input_path"])
        if not audio_path.is_file():
            return ToolResult(success=False, error=f"Audio file not found: {audio_path}")

        start = time.time()
        language = inputs.get("language") or ""
        fields = {
            "model": os.environ.get("VOICESTUDIO_STT_MODEL", "whisper-1"),
            "response_format": "verbose_json",
        }
        if language:
            fields["language"] = language
        headers = {"Authorization": f"Bearer {os.environ.get('VOICESTUDIO_API_KEY', '')}"}
        try:
            with audio_path.open("rb") as handle:
                response = requests.post(
                    f"{base_url}/v1/audio/transcriptions",
                    data=fields,
                    files={"file": (audio_path.name, handle, "audio/wav")},
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"VoiceStudio request failed: {exc}")
        if response.status_code != 200:
            return ToolResult(
                success=False,
                error=f"VoiceStudio returned HTTP {response.status_code}: {response.content[:200]!r}",
            )

        payload = response.json()
        text = (payload.get("text") or "").strip()
        segments = payload.get("segments") or [{"start": 0.0, "end": None, "text": text}]
        result = {
            "text": text,
            "language": payload.get("language") or language or None,
            "segments": segments,
            "provider": self.provider,
            "duration_seconds": round(time.time() - start, 2),
        }
        output_path = Path(inputs.get("output_path") or f"{audio_path.stem}_transcript.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return ToolResult(success=True, data=result, artifacts=[str(output_path)], cost_usd=0.0)
