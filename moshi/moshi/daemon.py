# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
odin-daemon — headless Python audio client for PersonaPlex.

Runs entirely in the terminal: no browser required.

Modes
-----
  direct   Connect to /api/chat immediately and stay connected.
  wake     Subscribe to /api/events SSE; open /api/chat only when a
           wake event arrives, then close after silence.

Usage examples
--------------
  # Direct mode (always on)
  moshi-daemon --url https://localhost:8998 \
               --voice-prompt NATF2.pt \
               --text-prompt "You are Odin." \
               --mode direct

  # Wake mode (event-driven)
  moshi-daemon --url https://localhost:8998 \
               --voice-prompt NATM1.pt \
               --mode wake
"""

import argparse
import asyncio
import json
import logging
import signal
import ssl
import sys
import time
from collections import deque
from typing import Optional

import aiohttp
import numpy as np
import sounddevice as sd
import sphn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24_000       # Hz — must match the server
CHANNELS = 1
FRAME_SIZE = 1_920         # 80 ms at 24 kHz

# WebSocket message kind bytes (mirrors the server protocol)
KIND_HANDSHAKE  = 0x00
KIND_AUDIO      = 0x01
KIND_TEXT       = 0x02
KIND_CONTROL    = 0x03
KIND_KEEPALIVE  = 0x06

CTRL_SPEAKING   = 0x04    # server → client: mute mic
CTRL_LISTENING  = 0x05    # server → client: unmute mic


# ---------------------------------------------------------------------------
# Audio I/O helpers (sounddevice ↔ asyncio bridge)
# ---------------------------------------------------------------------------

class AudioIO:
    """
    Wraps sounddevice InputStream (mic) and OutputStream (speaker).

    The SD callbacks run in a real-time audio thread; we bridge to asyncio
    using loop.call_soon_threadsafe so that asyncio.Queue stays safe.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        device_in: Optional[str | int],
        device_out: Optional[str | int],
    ) -> None:
        self._loop = loop
        self._device_in = device_in
        self._device_out = device_out
        self.mic_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=128)
        # Thread-safe deque; popleft / append are GIL-atomic in CPython.
        self._playback_buf: deque[np.ndarray] = deque()
        self.muted = False          # True while server is speaking
        self._in_stream: Optional[sd.InputStream] = None
        self._out_stream: Optional[sd.OutputStream] = None

    # -- mic --

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug(f"[mic] {status}")
        if self.muted:
            return
        pcm = indata[:, 0].astype(np.float32).copy()
        self._loop.call_soon_threadsafe(self._mic_enqueue, pcm)

    def _mic_enqueue(self, pcm: np.ndarray):
        try:
            self.mic_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass  # drop oldest is fine — the model is real-time

    # -- speaker --

    def _out_callback(self, outdata, frames, time_info, status):
        if status:
            logger.debug(f"[speaker] {status}")
        out = np.zeros(frames, dtype=np.float32)
        pos = 0
        while pos < frames and self._playback_buf:
            chunk = self._playback_buf[0]
            take = min(frames - pos, len(chunk))
            out[pos: pos + take] = chunk[:take]
            pos += take
            if take == len(chunk):
                self._playback_buf.popleft()
            else:
                self._playback_buf[0] = chunk[take:]
        outdata[:, 0] = out

    def push_pcm(self, pcm: np.ndarray):
        """Thread-safe: append decoded PCM to the playback buffer."""
        self._playback_buf.append(pcm)

    # -- lifecycle --

    def open(self):
        self._in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=self._device_in,
            callback=self._mic_callback,
        )
        self._out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=self._device_out,
            callback=self._out_callback,
        )
        self._in_stream.start()
        self._out_stream.start()

    def close(self):
        if self._in_stream:
            self._in_stream.stop()
            self._in_stream.close()
        if self._out_stream:
            self._out_stream.stop()
            self._out_stream.close()


# ---------------------------------------------------------------------------
# Single WebSocket session
# ---------------------------------------------------------------------------

async def run_session(
    ws_url: str,
    audio: AudioIO,
    ssl_ctx: ssl.SSLContext,
) -> None:
    """
    Run one /api/chat WebSocket session: encode mic → WS, decode WS → speaker.
    Returns when the connection closes.
    """
    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=connector) as http:
        try:
            async with http.ws_connect(ws_url) as ws:
                logger.info(f"WebSocket connected: {ws_url}")

                # Wait for handshake byte 0x00
                msg = await ws.receive()
                if msg.type != aiohttp.WSMsgType.BINARY or not msg.data or msg.data[0] != KIND_HANDSHAKE:
                    logger.error(f"Bad handshake: {msg!r}")
                    return
                logger.info("Handshake OK — session active.")
                print("\n[Odin] Session started. Speak now.\n", flush=True)

                async def send_loop():
                    """Mic → Opus → WebSocket."""
                    while True:
                        try:
                            pcm = await asyncio.wait_for(audio.mic_queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            return
                        opus_writer.append_pcm(pcm)
                        encoded = opus_writer.read_bytes()
                        if len(encoded) > 0:
                            try:
                                await ws.send_bytes(b"\x01" + encoded)
                            except (aiohttp.ClientConnectionError, RuntimeError):
                                return

                async def recv_loop():
                    """WebSocket → Opus → speaker / text / control."""
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            break
                        if msg.type != aiohttp.WSMsgType.BINARY:
                            continue
                        data: bytes = msg.data
                        if not data:
                            continue
                        kind = data[0]

                        if kind == KIND_AUDIO:
                            opus_reader.append_bytes(data[1:])
                            pcm = opus_reader.read_pcm()
                            if pcm.shape[-1] > 0:
                                audio.push_pcm(pcm)

                        elif kind == KIND_TEXT:
                            token = data[1:].decode("utf-8", errors="replace")
                            print(token, end="", flush=True)

                        elif kind == KIND_CONTROL:
                            if len(data) < 2:
                                continue
                            ctrl = data[1]
                            if ctrl == CTRL_SPEAKING:
                                audio.muted = True
                                print("\n[Odin speaking…]", flush=True)
                            elif ctrl == CTRL_LISTENING:
                                audio.muted = False
                                print("\n[Odin listening]", flush=True)

                        elif kind == KIND_KEEPALIVE:
                            pass  # server heartbeat

                done, pending = await asyncio.gather(
                    asyncio.create_task(send_loop()),
                    asyncio.create_task(recv_loop()),
                    return_exceptions=True,
                )
                for task in pending if hasattr(pending, '__iter__') else []:
                    task.cancel()

        except aiohttp.ClientConnectorError as exc:
            logger.error(f"Cannot connect to server: {exc}")
        except Exception as exc:
            logger.error(f"Session error: {exc}", exc_info=True)

    print("\n[Odin] Session closed.", flush=True)


# ---------------------------------------------------------------------------
# SSE wake listener
# ---------------------------------------------------------------------------

async def sse_listener(
    sse_url: str,
    callback,           # async callable(event_json: dict)
    ssl_ctx: ssl.SSLContext,
) -> None:
    """
    Subscribe to /api/events SSE stream and call *callback* on each data event.
    Reconnects automatically with exponential back-off on failure.
    """
    backoff = 2.0
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=connector) as http:
        while True:
            try:
                logger.info(f"Connecting to SSE: {sse_url}")
                async with http.get(sse_url) as resp:
                    if resp.status != 200:
                        logger.error(f"SSE returned HTTP {resp.status}")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                        continue
                    backoff = 2.0
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            try:
                                await callback(json.loads(payload))
                            except Exception as exc:
                                logger.debug(f"SSE callback error: {exc}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(f"SSE disconnected ({exc}); retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    base_url = args.url.rstrip("/")
    ws_scheme = "wss" if base_url.startswith("https") else "ws"
    http_scheme = "https" if base_url.startswith("https") else "http"
    host_port = base_url.split("://", 1)[1]

    ws_url = (
        f"{ws_scheme}://{host_port}/api/chat"
        f"?voice_prompt={args.voice_prompt}"
        f"&text_prompt={args.text_prompt}"
        f"&seed={args.seed}"
    )
    sse_url = f"{http_scheme}://{host_port}/api/events"

    # SSL context: skip verification for self-signed dev certs
    ssl_ctx: ssl.SSLContext | bool
    if base_url.startswith("https"):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
    else:
        ssl_ctx = False  # aiohttp: no SSL

    loop = asyncio.get_event_loop()
    audio = AudioIO(loop, args.device_in or None, args.device_out or None)
    audio.open()

    # Graceful shutdown on SIGINT / SIGTERM
    stop = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown requested.")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        if args.mode == "direct":
            # Connect once and run until stopped
            logger.info("Direct mode: connecting immediately.")
            session_task = asyncio.create_task(run_session(ws_url, audio, ssl_ctx))
            stop_task = asyncio.create_task(stop.wait())
            await asyncio.wait(
                [session_task, stop_task], return_when=asyncio.FIRST_COMPLETED
            )
            session_task.cancel()

        else:  # wake mode
            logger.info("Wake mode: waiting for /api/events wake event.")
            active_session: Optional[asyncio.Task] = None

            async def on_sse_event(event: dict):
                nonlocal active_session
                if event.get("type") != "wake":
                    return
                if active_session and not active_session.done():
                    logger.info("Wake event received but session already active — ignoring.")
                    return
                logger.info(f"Wake event: {event.get('text', '')!r}")
                print(f'\n[Odin] Wake! "{event.get("text", "")}"', flush=True)
                active_session = asyncio.create_task(run_session(ws_url, audio, ssl_ctx))

            sse_task = asyncio.create_task(sse_listener(sse_url, on_sse_event, ssl_ctx))
            stop_task = asyncio.create_task(stop.wait())
            await asyncio.wait([sse_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            sse_task.cancel()
            if active_session:
                active_session.cancel()

    finally:
        audio.close()
        logger.info("Audio streams closed.")


def main():
    parser = argparse.ArgumentParser(
        description="PersonaPlex Python audio daemon — headless client."
    )
    parser.add_argument(
        "--url",
        default="https://localhost:8998",
        help="Base URL of the PersonaPlex server (default: https://localhost:8998).",
    )
    parser.add_argument(
        "--voice-prompt",
        default="NATF2.pt",
        help="Voice prompt filename served by the server (default: NATF2.pt).",
    )
    parser.add_argument(
        "--text-prompt",
        default="",
        help="System/role prompt for this session.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for the session (-1 = random).",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "wake"],
        default="direct",
        help="'direct' = connect immediately; 'wake' = wait for SSE wake event.",
    )
    parser.add_argument(
        "--device-in",
        default=None,
        help="sounddevice input device name or index (default: system default).",
    )
    parser.add_argument(
        "--device-out",
        default=None,
        help="sounddevice output device name or index (default: system default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
