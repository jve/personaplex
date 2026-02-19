# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import json
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
from .session_manager import SessionManager


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False,
                 session_manager_cfg: Optional[dict] = None):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.notify_queue: asyncio.Queue = asyncio.Queue()
        self.sse_subscribers: set = set()
        self.session_manager_cfg: Optional[dict] = session_manager_cfg  # None → disabled
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_sse(self, request):
        """Server-Sent Events endpoint — clients subscribe here for proactive wake events."""
        response = web.StreamResponse(headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        })
        await response.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        self.sse_subscribers.add(q)
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=20.0)
                    await response.write(f"data: {data}\n\n".encode())
                    await response.drain()
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    await response.drain()
        except Exception:
            pass
        finally:
            self.sse_subscribers.discard(q)
        return response

    async def handle_notify(self, request):
        """POST /api/notify — external systems inject an event for Odin to speak."""
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON")
        text = body.get("text", "").strip()
        if not text:
            return web.Response(status=400, text="Missing 'text' field")
        # Queue for the active audio session's event_loop to consume
        await self.notify_queue.put(text)
        # Wake any idle SSE subscribers so the client can start a session
        event_json = json.dumps({"type": "wake", "text": text})
        for q in list(self.sse_subscribers):
            try:
                q.put_nowait(event_json)
            except asyncio.QueueFull:
                pass
        return web.Response(status=200, text="ok")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path
                
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        seed = int(request["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        # Injection state shared between event_loop and opus_loop.
        # asyncio is single-threaded so plain lists are safe as shared mutables.
        pending_text_tokens: list = []   # int token IDs queued by event_loop
        inject_post_silence = [0]        # frames of settling silence after injection

        # --- Per-session SessionManager (None when feature is disabled) ---
        session_mgr: Optional[SessionManager] = None
        if self.session_manager_cfg is not None:
            cfg = self.session_manager_cfg
            text_prompt_for_session = request.query.get("text_prompt", "")
            session_mgr = SessionManager(
                notify_queue=self.notify_queue,
                text_prompt=text_prompt_for_session,
                whisper_model_size=cfg.get("whisper_model", "base"),
                anthropic_api_key=cfg.get("anthropic_key"),
                device=str(self.device),
            )

        async def opus_loop():
            all_pcm_data = None

            # --- VAD state ---
            _vad_threshold = 0.015       # RMS energy threshold (float32 [-1,1])
            _vad_onset_frames = 3        # consecutive speech frames → turn start (~240 ms)
            _vad_offset_frames = 10      # consecutive silent frames → turn end  (~800 ms)
            _vad_max_frames = int(30 * self.mimi.frame_rate)  # 30 s hard cap
            _vad_speech_cnt = 0
            _vad_silence_cnt = 0
            _vad_active = False          # currently collecting a user turn
            _vad_buf: list[np.ndarray] = []

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)

                # --- Injection: Odin speaks forced text, user side silent ---
                if pending_text_tokens:
                    token_id = pending_text_tokens.pop(0)
                    tokens = self.lm_gen.step(
                        input_tokens=self.lm_gen._encode_zero_frame(),
                        text_token=token_id,
                    )
                    if tokens is not None:
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))
                    continue

                # --- Post-injection silence: advance LM context without audio output ---
                if inject_post_silence[0] > 0:
                    inject_post_silence[0] -= 1
                    self.lm_gen.step(
                        input_tokens=self.lm_gen._encode_zero_frame(),
                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                        text_token=self.lm_gen.zero_text_code,
                    )
                    continue

                # --- Normal path: process incoming user audio ---
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    pcm_chunk = all_pcm_data[: self.frame_size]   # numpy float32
                    all_pcm_data = all_pcm_data[self.frame_size:]

                    # --- VAD tee (only when no injection in flight) ---
                    if session_mgr is not None:
                        rms = float(np.sqrt(np.mean(pcm_chunk ** 2)))
                        is_speech = rms > _vad_threshold
                        if is_speech:
                            _vad_speech_cnt += 1
                            _vad_silence_cnt = 0
                        else:
                            _vad_silence_cnt += 1
                            _vad_speech_cnt = 0

                        if not _vad_active and _vad_speech_cnt >= _vad_onset_frames:
                            _vad_active = True
                            _vad_buf = []
                            clog.log("info", "VAD: speech start")

                        if _vad_active:
                            _vad_buf.append(pcm_chunk.copy())
                            turn_ended = (
                                _vad_silence_cnt >= _vad_offset_frames
                                or len(_vad_buf) >= _vad_max_frames
                            )
                            if turn_ended:
                                turn_pcm = np.concatenate(_vad_buf)
                                dur = len(turn_pcm) / self.mimi.sample_rate
                                clog.log("info", f"VAD: turn end ({dur:.1f}s, {len(turn_pcm)} samples)")
                                asyncio.create_task(
                                    session_mgr.process_turn(turn_pcm, self.mimi.sample_rate)
                                )
                                _vad_active = False
                                _vad_buf = []
                                _vad_speech_cnt = 0
                                _vad_silence_cnt = 0

                    chunk = torch.from_numpy(pcm_chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

        async def send_loop():
            last_send_time = time.time()
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)
                    last_send_time = time.time()
                elif time.time() - last_send_time > 5.0:
                    # Keep-alive ping so the client doesn't time out while Odin listens
                    await ws.send_bytes(b"\x06")
                    last_send_time = time.time()

        async def event_loop():
            """Drains notify_queue and injects Odin speech mid-conversation."""
            while True:
                if close:
                    return
                try:
                    text = await asyncio.wait_for(self.notify_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                # Wait for any in-progress injection to finish before starting the next
                while pending_text_tokens or inject_post_silence[0] > 0:
                    await asyncio.sleep(0.02)
                # Tell the client to mute the mic while Odin speaks
                try:
                    await ws.send_bytes(b"\x03\x04")  # control: speaking
                except Exception:
                    return
                token_ids = self.text_tokenizer.encode(text)
                clog.log("info", f"injecting {len(token_ids)} tokens: {text!r}")
                pending_text_tokens.extend(token_ids)
                inject_post_silence[0] = max(1, int(0.3 * self.mimi.frame_rate))
                # Wait for opus_loop to consume all tokens and settle silence
                while pending_text_tokens or inject_post_silence[0] > 0:
                    await asyncio.sleep(0.02)
                # Tell the client the mic can open again
                try:
                    await ws.send_bytes(b"\x03\x05")  # control: listening
                except Exception:
                    return

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                    asyncio.create_task(event_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )
    # Session manager / brain options
    parser.add_argument(
        "--session-manager",
        action="store_true",
        help="Enable per-session ASR (Whisper) + brain LLM (Claude). "
             "Requires faster-whisper and/or anthropic packages.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="faster-whisper model size used for ASR (default: base).",
    )
    parser.add_argument(
        "--anthropic-key",
        type=str,
        default=None,
        help="Anthropic API key for the brain LLM (or set ANTHROPIC_API_KEY env var).",
    )

    args = parser.parse_args()

    # Allow API key from environment
    if args.anthropic_key is None:
        args.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    session_manager_cfg = None
    if args.session_manager:
        session_manager_cfg = {
            "whisper_model": args.whisper_model,
            "anthropic_key": args.anthropic_key,
        }
        logger.info(f"Session manager enabled (whisper={args.whisper_model}, "
                    f"brain={'yes' if args.anthropic_key else 'no (no API key)'})")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        session_manager_cfg=session_manager_cfg,
    )
    logger.info("warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/events", state.handle_sse)
    app.router.add_post("/api/notify", state.handle_notify)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
