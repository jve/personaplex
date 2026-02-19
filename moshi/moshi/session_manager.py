# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
SessionManager: per-session ASR + deliberative brain for PersonaPlex.

Soft dependencies (gracefully disabled if absent):
    faster-whisper  — speech-to-text transcription
    anthropic       — Claude brain LLM for proactive responses
"""

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: faster-whisper
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore
    _WHISPER_OK = True
except ImportError:
    _WhisperModel = None  # type: ignore
    _WHISPER_OK = False
    logger.warning("faster-whisper not installed; ASR transcription disabled.")

# ---------------------------------------------------------------------------
# Optional: anthropic
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic_mod  # type: ignore
    _ANTHROPIC_OK = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _ANTHROPIC_OK = False
    logger.warning("anthropic not installed; brain LLM disabled.")


class SessionManager:
    """
    Tracks a single WebSocket session's conversation history.

    On each detected user speech turn (called by the VAD tee in opus_loop):
      1. Transcribes the PCM with Whisper (if available)
      2. Optionally calls Claude to generate a reply
      3. Pushes the reply text to notify_queue so event_loop injects it as
         Odin speech (exactly like POST /api/notify does)

    Args:
        notify_queue:       asyncio.Queue shared with the session's event_loop.
        text_prompt:        The Odin system/role prompt for the session.
        whisper_model_size: faster-whisper model size ("tiny","base","small",…).
        anthropic_api_key:  Anthropic API key; None → brain disabled even if
                            the library is installed.
        device:             "cuda" or "cpu" for Whisper inference.
    """

    def __init__(
        self,
        notify_queue: asyncio.Queue,
        text_prompt: str = "",
        whisper_model_size: str = "base",
        anthropic_api_key: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.notify_queue = notify_queue
        self.text_prompt = text_prompt
        self.conversation_history: list[dict] = []
        self._lock = asyncio.Lock()

        # --- Whisper ---
        self._asr: Optional[object] = None
        if _WHISPER_OK:
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self._asr = _WhisperModel(
                    whisper_model_size, device=device, compute_type=compute_type
                )
                logger.info(
                    f"Whisper model '{whisper_model_size}' loaded on {device} "
                    f"({compute_type})"
                )
            except Exception as exc:
                logger.error(f"Failed to load Whisper: {exc}")

        # --- Anthropic / Claude ---
        self._claude: Optional[object] = None
        if _ANTHROPIC_OK and anthropic_api_key:
            try:
                self._claude = _anthropic_mod.AsyncAnthropic(api_key=anthropic_api_key)
                logger.info("Anthropic AsyncAnthropic client initialised.")
            except Exception as exc:
                logger.error(f"Failed to init Anthropic client: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_sync(self, pcm: np.ndarray) -> str:
        """Blocking Whisper transcription — run via executor."""
        if self._asr is None:
            return ""
        try:
            segments, _ = self._asr.transcribe(  # type: ignore[attr-defined]
                pcm.astype(np.float32), beam_size=5, language="en"
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            logger.error(f"Whisper transcription error: {exc}")
            return ""

    async def _ask_brain(self, user_text: str) -> Optional[str]:
        """Call Claude with the current conversation history. Returns reply or None."""
        if self._claude is None:
            return None

        system = self.text_prompt or "You are a helpful voice assistant."
        messages = self.conversation_history.copy()

        try:
            response = await self._claude.messages.create(  # type: ignore[attr-defined]
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=system,
                messages=messages,
            )
            reply = response.content[0].text.strip() if response.content else ""
            return reply or None
        except Exception as exc:
            logger.error(f"Brain LLM error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_turn(self, pcm: np.ndarray, sample_rate: int) -> None:
        """
        Called by the VAD tee in opus_loop when a user speech turn ends.

        Runs ASR in a thread-pool executor so the event loop is not blocked,
        then optionally invokes the brain and enqueues the reply.
        """
        # Serialise turns so history stays ordered even if turns arrive fast.
        async with self._lock:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._transcribe_sync, pcm)

            if not text:
                logger.debug("VAD turn yielded empty transcript — skipping.")
                return

            logger.info(f"[ASR] user: {text!r}")
            self.conversation_history.append({"role": "user", "content": text})

            reply = await self._ask_brain(text)
            if reply:
                logger.info(f"[Brain] assistant: {reply!r}")
                self.conversation_history.append(
                    {"role": "assistant", "content": reply}
                )
                await self.notify_queue.put(reply)
