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

"""
Offline inference entrypoint for PersonaPlex that mirrors server.py behavior without a WebSocket server.

High-level flow:
- Load Mimi encoders/decoders, Moshi LM, and tokenizer (same as server.py)
- Warmup to initialize CUDA graphs and streaming state
- Prompt phase: load system text tokens and a voice prompt WAV (agent side)
- Streaming-like phase: feed user audio frames from a WAV file into the "input" channels,
  autoregressively sample text + agent audio channels each step, and decode audio frames
- Concatenate generated frames and write an output WAV matching the input duration

This script reuses helpers from lm.py (load_audio, _iterate_audio, encode_from_sphn) to
keep parity with voice-prompt feeding logic in the server.
"""

import argparse
import os
import tarfile
from pathlib import Path
import json
from typing import Optional, List

import numpy as np
import torch
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders, LMGen, MimiModel
from .models.lm import load_audio as lm_load_audio
from .models.lm import _iterate_audio as lm_iterate_audio
from .models.lm import encode_from_sphn as lm_encode_from_sphn


def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed: int):
    """Seed torch, CUDA, numpy, and Python RNG for reproducible runs.

    Matches the seeding strategy in server.py.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
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


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: str, frame_size: int):
    """Run a short warmup loop to initialize CUDA graphs and streaming state.

    Replicates the same warmup behavior as server.py: zeros → encode → LMGen.step → decode.
    """
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            # Decode agent audio channels to ensure decode graphs/states are primed
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_tokens_to_pcm(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, tokens: torch.Tensor) -> np.ndarray:
    """Decode a single step of model tokens to PCM using Mimi.

    tokens is shaped [B, dep_q+1, 1]; channels 1..dep_q are the agent audio codebooks.
    Returns a 1D float32 numpy array (mono) for the current frame.
    """
    pcm = mimi.decode(tokens[:, 1:9])
    _ = other_mimi.decode(tokens[:, 1:9])
    pcm = pcm.detach().cpu().numpy()[0, 0]
    return pcm


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

    log("info", "retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        log("info", f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _load_models(
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool,
    seed: Optional[int],
):
    """Shared model loading + warmup used by both inference modes."""
    if seed is not None and seed != -1:
        seed_all(seed)

    hf_hub_download(hf_repo, "config.json")

    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "moshi loaded")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    return mimi, other_mimi, text_tokenizer, lm_gen, frame_size


def _setup_prompts(mimi, other_mimi, lm_gen, text_tokenizer, voice_prompt_path, text_prompt):
    """Load voice + text prompts and reset streaming state, ready for generation."""
    if voice_prompt_path.endswith(".pt"):
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()


def _write_outputs(
    generated_frames: List[np.ndarray],
    generated_text_tokens: List[str],
    output_wav: str,
    output_text: Optional[str],
    sample_rate: int,
    target_samples: Optional[int] = None,
):
    """Concatenate frames, trim/pad if needed, and write WAV + JSON."""
    output_pcm = np.concatenate(generated_frames, axis=-1)
    if target_samples is not None:
        if output_pcm.shape[-1] > target_samples:
            output_pcm = output_pcm[:target_samples]
        elif output_pcm.shape[-1] < target_samples:
            pad_len = target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate(
                [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
            )
    sphn.write_wav(output_wav, output_pcm, sample_rate)
    log("info", f"Wrote output audio to {output_wav}")
    if output_text is not None:
        with open(output_text, "w") as fh:
            json.dump(generated_text_tokens, fh, ensure_ascii=False)
        log("info", f"Wrote output text to {output_text}")


# ---------------------------------------------------------------------------
# TTS mode
# ---------------------------------------------------------------------------

def run_tts(
    speak_text: str,
    output_wav: str,
    output_text: Optional[str],
    text_prompt: str,
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    trailing_frames: int = 25,
    cpu_offload: bool = False,
):
    """
    Text-to-speech mode: tokenize *speak_text*, inject tokens into the LM with
    silent user audio, and collect the agent's synthesised speech.

    Unlike run_inference(), no input WAV is needed — the user channel is fed
    zeros (silence) for every step so the model only listens to the forced text.

    Args:
        speak_text:      The text Odin should speak aloud.
        trailing_frames: Extra LM steps after all text tokens are consumed, letting
                         the model finish its last word naturally (default 25 ≈ 2 s).
    """
    mimi, other_mimi, text_tokenizer, lm_gen, _ = _load_models(
        tokenizer_path, moshi_weight, mimi_weight, hf_repo, device,
        temp_audio, temp_text, topk_audio, topk_text, greedy,
        save_voice_prompt_embeddings=False, cpu_offload=cpu_offload, seed=seed,
    )
    _setup_prompts(mimi, other_mimi, lm_gen, text_tokenizer, voice_prompt_path, text_prompt)

    # Tokenize the utterance the model should say
    speak_ids: List[int] = text_tokenizer.encode(speak_text)
    log("info", f"TTS: {len(speak_ids)} tokens for: {speak_text!r}")

    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []

    def _step_and_collect(text_token_id):
        tokens = lm_gen.step(
            input_tokens=lm_gen._encode_zero_frame(),
            text_token=text_token_id,
        )
        if tokens is None:
            return
        pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
        generated_frames.append(pcm)
        tok_id = tokens[0, 0, 0].item()
        if tok_id not in (0, 3):
            piece = text_tokenizer.id_to_piece(tok_id).replace("▁", " ")  # type: ignore
            log("info", f"text token '{piece}'")
            generated_text_tokens.append(piece)
        else:
            token_map = ["EPAD", "BOS", "EOS", "PAD"]
            log("info", f"text token '{token_map[tok_id]}'")
            generated_text_tokens.append(token_map[tok_id])

    # Injection phase: force each text token
    for tok_id in speak_ids:
        _step_and_collect(tok_id)

    # Trailing phase: let the model finish the last word naturally
    for _ in range(trailing_frames):
        _step_and_collect(lm_gen.zero_text_code)

    if not generated_frames:
        log("error", "No audio frames generated — check speak_text and configuration.")
        return

    _write_outputs(generated_frames, generated_text_tokens, output_wav, output_text,
                   mimi.sample_rate)


# ---------------------------------------------------------------------------
# WAV-to-WAV inference mode (original)
# ---------------------------------------------------------------------------

def run_inference(
    input_wav: str,
    output_wav: str,
    output_text: Optional[str],
    text_prompt: str,
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
):
    """Run offline inference using an input WAV as the user-side stream.

    - Loads/initializes models and tokenizer
    - Warms up execution
    - Loads system text tokens and voice prompt
    - Runs prompt phases (text + voice + silences) via LMGen.step_system_prompts
    - Streams the user WAV frames into the input channels and samples model outputs
    - Decodes and writes an output WAV of the same duration
    """
    mimi, other_mimi, text_tokenizer, lm_gen, _ = _load_models(
        tokenizer_path, moshi_weight, mimi_weight, hf_repo, device,
        temp_audio, temp_text, topk_audio, topk_text, greedy,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        cpu_offload=cpu_offload, seed=seed,
    )
    _setup_prompts(mimi, other_mimi, lm_gen, text_tokenizer, voice_prompt_path, text_prompt)

    sample_rate = mimi.sample_rate
    user_audio = lm_load_audio(input_wav, sample_rate)  # (C, T) at model SR
    total_target_samples = user_audio.shape[-1]

    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
        max_batch=1,
    ):
        steps = user_encoded.shape[-1]
        for c in range(steps):
            tokens = lm_gen.step(user_encoded[:, :, c : c + 1])
            if tokens is None:
                continue
            pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
            generated_frames.append(pcm)
            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                log("info", f"text token '{_text}'")
                generated_text_tokens.append(_text)
            else:
                text_token_map = ["EPAD", "BOS", "EOS", "PAD"]
                log("info", f"text token '{text_token_map[text_token]}'")
                generated_text_tokens.append(text_token_map[text_token])

    if not generated_frames:
        log("error", "No audio frames were generated. Check input file and configuration.")
        return

    _write_outputs(generated_frames, generated_text_tokens, output_wav, output_text,
                   sample_rate, target_samples=total_target_samples)    


def main():
    """Parse CLI args and run offline inference (WAV-to-WAV or TTS)."""
    parser = argparse.ArgumentParser(
        description=(
            "Offline inference for PersonaPlex.\n\n"
            "Two modes:\n"
            "  WAV mode  (default): feed an input WAV as the user stream.\n"
            "  TTS mode  (--speak-text): synthesise speech directly from text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode selection ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--input-wav", type=str,
        help="[WAV mode] Path to input WAV file (user audio).",
    )
    mode_group.add_argument(
        "--speak-text", type=str,
        help="[TTS mode] Text Odin should speak. No input WAV required.",
    )

    parser.add_argument(
        "--output-wav", required=True, type=str,
        help="Path to write the output WAV file.",
    )
    parser.add_argument(
        "--output-text", type=str, default=None,
        help="Path to write generated text tokens as JSON (optional).",
    )
    parser.add_argument(
        "--trailing-frames", type=int, default=25,
        help="[TTS mode] Extra frames after last token so the model finishes speaking "
             "(default 25 ≈ 2 s at 12.5 fps).",
    )
    parser.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.", type=str, help="Text prompt")

    parser.add_argument(
        "--voice-prompt", required=True, type=str, help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt')."
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from -voice-prompt arg will be joined with this directory path."
        )
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to pre-trained model repo)",
    )

    # Runtime / sampling controls (mirror UI semantics)
    parser.add_argument(
        "--temp-audio", type=float, default=0.8, help="Audio sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7, help="Text sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250, help="Audio top-k sampling (default: 250)"
    )
    parser.add_argument(
        "--topk-text", type=int, default=25, help="Text top-k sampling (default: 25)"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling (greedy decoding)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'."
    )
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables)")

    args = parser.parse_args()

    if args.input_wav is None and args.speak_text is None:
        parser.error("One of --input-wav (WAV mode) or --speak-text (TTS mode) is required.")

    # Resolve voice prompt path
    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    log("info", f"voice_prompt_dir = {voice_prompt_dir}")

    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    greedy = bool(args.greedy)

    common_kwargs = dict(
        output_wav=args.output_wav,
        output_text=args.output_text,
        text_prompt=args.text_prompt,
        voice_prompt_path=voice_prompt_path,
        tokenizer_path=args.tokenizer,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        hf_repo=args.hf_repo,
        device=args.device,
        seed=args.seed,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        topk_audio=args.topk_audio,
        topk_text=args.topk_text,
        greedy=greedy,
        cpu_offload=args.cpu_offload,
    )

    with torch.no_grad():
        if args.speak_text is not None:
            log("info", f"TTS mode — synthesising: {args.speak_text!r}")
            run_tts(
                speak_text=args.speak_text,
                trailing_frames=args.trailing_frames,
                **common_kwargs,
            )
        else:
            log("info", f"WAV mode — input: {args.input_wav}")
            run_inference(
                input_wav=args.input_wav,
                save_voice_prompt_embeddings=False,
                **common_kwargs,
            )


if __name__ == "__main__":
    main()