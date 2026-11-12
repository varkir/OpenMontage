"""VoiceStudio text-to-speech provider tool (local, OpenAI-style /v1/audio/speech)."""

from __future__ import annotations

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


class VoiceStudioTTS(BaseTool):
    name = "voicestudio_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "voicestudio"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL

    dependencies = []
    install_instructions = (
        "Set the VoiceStudio address and key:\n"
        "  VOICESTUDIO_BASE_URL=http://<host>:3900\n"
        "  VOICESTUDIO_API_KEY=<key>\n"
        "The voice name must exist as a profile in VoiceStudio (GET /profiles)."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]

    capabilities = ["text_to_speech", "voice_selection"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
    }
    best_for = [
        "local narration on the home server, no per-character cost",
        "Indonesian and English voices already set up in VoiceStudio",
    ]
    not_good_for = [
        "voices that are not created in VoiceStudio",
    ]

    input_schema = {
        "type": "object",
        "required": ["text", "voice"],
        "properties": {
            "text": {"type": "string"},
            "voice": {
                "type": "string",
                "description": "VoiceStudio profile name, for example en-stone or id-dk",
            },
            "language": {
                "type": "string",
                "default": "en",
                "description": "ISO 639-1 language code sent to VoiceStudio",
            },
            "speed": {"type": "number", "default": 1.0, "minimum": 0.25, "maximum": 4.0},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["timeout"])
    idempotency_key_fields = ["text", "voice", "language", "speed"]

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
        if not inputs.get("voice"):
            return ToolResult(success=False, error="voice is required (a VoiceStudio profile name).")

        start = time.time()
        output_path = Path(inputs.get("output_path", "voicestudio_tts.wav"))
        body = {
            "model": os.environ.get("VOICESTUDIO_TTS_MODEL", "omnivoice"),
            "voice": inputs["voice"],
            "input": inputs["text"],
            "language": inputs.get("language", "en"),
            "response_format": "wav",
            "speed": inputs.get("speed", 1.0),
        }
        headers = {"Authorization": f"Bearer {os.environ.get('VOICESTUDIO_API_KEY', '')}"}
        try:
            response = requests.post(
                f"{base_url}/v1/audio/speech", json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"VoiceStudio request failed: {exc}")
        if response.status_code != 200:
            return ToolResult(
                success=False,
                error=f"VoiceStudio returned HTTP {response.status_code}: {response.content[:200]!r}",
            )
        if response.content[:4] != b"RIFF":
            return ToolResult(
                success=False,
                error=f"VoiceStudio response is not a WAV ({response.headers.get('Content-Type')}): {response.content[:200]!r}",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "voice": inputs["voice"],
                "language": body["language"],
                "text_length": len(inputs["text"]),
                "output": str(output_path),
                "duration_seconds": round(time.time() - start, 2),
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
        )
