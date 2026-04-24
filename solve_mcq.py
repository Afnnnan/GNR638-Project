#!/usr/bin/env python3
"""
GNR638 Project: Deep Learning MCQ Answering from Images
========================================================
Uses Qwen3-VL-8B-Instruct (with thinking mode) to read PNG images of
deep learning multiple-choice questions and predict the correct answer.

Usage:
    python solve_mcq.py --data_dir ./sample_test_project_2
    python solve_mcq.py --data_dir ./sample_test_project_2 --model_path /local/path/to/weights
    python solve_mcq.py --data_dir ./sample_test_project_2 --no_think  # disable thinking mode

References:
    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    - Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
"""

import argparse
import logging
import os

# Suppress broken TensorFlow import in transformers (if TF is installed but broken)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Guard against broken TensorFlow installations — transformers may try to import it.
# If TF is broken, block it from being found by the import system.
import sys
import types

try:
    import tensorflow  # noqa: F401
except Exception:
    # TF is installed but broken — remove it from sys.modules and block re-import
    import importlib.abc
    tf_keys = [k for k in sys.modules if k == "tensorflow" or k.startswith("tensorflow.")]
    for k in tf_keys:
        del sys.modules[k]

    class _TFBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Block broken TF from being imported by transformers."""
        def find_module(self, fullname, path=None):
            if fullname == "tensorflow" or fullname.startswith("tensorflow."):
                return self
            return None
        def find_spec(self, fullname, path, target=None):
            if fullname == "tensorflow" or fullname.startswith("tensorflow."):
                return None
            return None
        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            mod = types.ModuleType(fullname)
            mod.__version__ = "0.0.0"
            mod.__path__ = []
            mod.__file__ = ""
            mod.__loader__ = self
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.insert(0, _TFBlocker())

import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VALID_ANSWERS = {1, 2, 3, 4, 5}
ABSTAIN = 5

SYSTEM_PROMPT = """\
You are an expert in deep learning, neural networks, machine learning, \
and computer vision. You answer multiple-choice questions with precision.

Rules:
- Read the question and ALL four options from the image very carefully.
- Think step-by-step through the problem, showing your full reasoning.
- Your final answer MUST be exactly one of: 1, 2, 3, or 4 \
(corresponding to options A, B, C, D respectively).
- If you truly cannot determine the answer, respond with: 5
- On the VERY LAST line of your response, output ONLY this: ANSWER: <number>

Example final line:
ANSWER: 2
"""

RETRY_PROMPT = """\
You are an expert in deep learning. Look at this MCQ image carefully.

Pick the correct option: 1 (A), 2 (B), 3 (C), or 4 (D).
Reply with ONLY a single line: ANSWER: <number>
"""


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device_config():
    """Auto-detect the best available device and return config dict."""
    if torch.cuda.is_available():
        log.info("CUDA detected — using GPU acceleration")
        # Check available VRAM to decide on quantization
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")

        # Use flash_attention_2 if installed, otherwise fall back to sdpa
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info("  FlashAttention2 available — using it")
        except ImportError:
            attn_impl = "sdpa"
            log.info("  FlashAttention2 not installed — using SDPA")

        # 8B model needs ~16GB in float16; if VRAM < 20GB, use 4-bit quantization
        use_4bit = vram_gb < 20
        if use_4bit:
            log.info(f"  VRAM < 20GB — will use 4-bit quantization (bnb)")
        else:
            log.info(f"  VRAM >= 20GB — using float16 (no quantization)")

        return {
            "device": "cuda",
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "attn_implementation": attn_impl,
            "use_4bit": use_4bit,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("MPS detected — using Apple Silicon GPU")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        return {
            "device": "mps",
            "device_map": None,
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
            "use_4bit": False,
        }
    else:
        log.warning("No GPU detected — falling back to CPU (will be slow!)")
        return {
            "device": "cpu",
            "device_map": None,
            "torch_dtype": torch.float32,
            "attn_implementation": "sdpa",
            "use_4bit": False,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, device_cfg: dict):
    """Load the Qwen3-VL model and processor."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    log.info(f"Loading model from: {model_path}")
    t0 = time.time()

    load_kwargs = {
        "attn_implementation": device_cfg["attn_implementation"],
    }
    if device_cfg["device_map"] is not None:
        load_kwargs["device_map"] = device_cfg["device_map"]

    # 4-bit quantization for GPUs with limited VRAM (e.g. T4 16GB)
    if device_cfg.get("use_4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        log.info("  Using 4-bit NF4 quantization (bitsandbytes)")
    else:
        load_kwargs["dtype"] = device_cfg["torch_dtype"]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, **load_kwargs
    )

    # For MPS / CPU: manually move to device
    if device_cfg["device_map"] is None:
        model = model.to(device_cfg["device"])

    model.eval()

    # Limit image resolution to save VRAM
    min_pixels = 256 * 28 * 28    # ~200K pixels
    max_pixels = 1280 * 28 * 28   # ~1M pixels (default is higher)
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )

    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
def parse_answer(text: str) -> int | None:
    """
    Extract the answer from model output.
    Looks for 'ANSWER: X' pattern, then falls back to last digit in text.
    Returns int in {1,2,3,4,5} or None if unparseable.
    """
    # Primary: look for ANSWER: X
    match = re.search(r"ANSWER\s*:\s*([1-5])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: last standalone digit 1-5 in the response
    digits = re.findall(r"\b([1-5])\b", text[-100:])
    if digits:
        return int(digits[-1])

    return None


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------
def answer_question(
    image_path: str,
    model,
    processor,
    device: str,
    enable_thinking: bool = True,
    max_new_tokens: int = 2048,
) -> tuple[int, str]:
    """
    Run inference on a single MCQ image.

    Returns:
        (answer, reasoning) where answer is in {1,2,3,4,5}
    """
    img = Image.open(image_path).convert("RGB")

    # -- First attempt: full reasoning prompt --
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": "Look at this multiple-choice question image. "
                    "Read the question and all options carefully, then "
                    "reason step-by-step and provide your answer.",
                },
            ],
        },
    ]

    # enable_thinking may not be supported in all transformers versions
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        inputs = processor.apply_chat_template(
            messages, **chat_kwargs, enable_thinking=enable_thinking,
        )
    except TypeError:
        log.info("  enable_thinking not supported — using default")
        inputs = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

    # Trim input tokens from output
    trimmed = [
        out[len(inp) :]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    answer = parse_answer(output_text)

    if answer is not None and answer in VALID_ANSWERS:
        return answer, output_text

    # -- Retry with stricter prompt --
    log.warning(f"  First attempt parse failed, retrying with strict prompt...")
    messages_retry = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": RETRY_PROMPT},
            ],
        },
    ]

    try:
        inputs_retry = processor.apply_chat_template(
            messages_retry, **chat_kwargs, enable_thinking=False,
        )
    except TypeError:
        inputs_retry = processor.apply_chat_template(messages_retry, **chat_kwargs)
    inputs_retry = inputs_retry.to(model.device)

    with torch.no_grad():
        gen_retry = model.generate(
            **inputs_retry,
            max_new_tokens=64,
            temperature=0.1,
            top_p=0.9,
        )

    trimmed_retry = [
        out[len(inp) :]
        for inp, out in zip(inputs_retry.input_ids, gen_retry)
    ]
    retry_text = processor.batch_decode(
        trimmed_retry, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    answer_retry = parse_answer(retry_text)

    if answer_retry is not None and answer_retry in VALID_ANSWERS:
        return answer_retry, f"[RETRY] {retry_text}"

    # -- Ultimate fallback: abstain --
    log.warning(f"  Retry also failed. Abstaining with {ABSTAIN}.")
    return ABSTAIN, f"[ABSTAIN] first: {output_text[:200]}... retry: {retry_text}"


# ---------------------------------------------------------------------------
# Clear GPU cache (helps with MPS and limited-VRAM GPUs)
# ---------------------------------------------------------------------------
def clear_cache(device: str):
    """Free GPU memory between questions."""
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GNR638 — Deep Learning MCQ Solver using Qwen3-VL"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the test data directory (contains test.csv and images/)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="Disable thinking mode (faster but less accurate)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (default: 2048)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_csv = data_dir / "test.csv"
    images_dir = data_dir / "images"

    # Resolve images directory — handle multiple folder structures:
    #   1. data_dir/images/*.png  (standard)
    #   2. data_dir/*.png         (flat — images uploaded directly)
    if not images_dir.exists():
        # Check if data_dir itself contains PNG files
        pngs_in_root = sorted(data_dir.glob("*.png"))
        if pngs_in_root:
            log.info(f"No images/ subdirectory found — using {data_dir} directly")
            images_dir = data_dir
        else:
            log.error(f"No images found. Checked {images_dir} and {data_dir}")
            sys.exit(1)

    # Load test set — auto-generate from image files if test.csv is missing
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        log.info(f"Loaded {len(df_test)} questions from {test_csv}")
    else:
        log.info(f"test.csv not found — auto-discovering images from {images_dir}")
        png_files = sorted(images_dir.glob("*.png"))
        if not png_files:
            log.error(f"No PNG images found in {images_dir}")
            sys.exit(1)
        # Build dataframe: image_name is the filename without extension
        image_names = [f.stem for f in png_files]
        df_test = pd.DataFrame({"image_name": image_names})
        log.info(f"Found {len(df_test)} images: {image_names[:5]}{'...' if len(image_names) > 5 else ''}")

    # Setup device
    device_cfg = get_device_config()
    device = device_cfg["device"]

    # Load model
    model, processor = load_model(args.model_path, device_cfg)

    # Process each question
    enable_thinking = not args.no_think
    log.info(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    log.info(f"Starting inference on {len(df_test)} questions...\n")

    results = []
    total_t0 = time.time()

    for idx, row in df_test.iterrows():
        image_name = row["image_name"]
        image_path = images_dir / f"{image_name}.png"

        if not image_path.exists():
            log.warning(f"[{idx+1}/{len(df_test)}] Image not found: {image_path} — abstaining")
            results.append({"image_name": image_name, "option": ABSTAIN})
            continue

        log.info(f"[{idx+1}/{len(df_test)}] Processing {image_name}...")
        t0 = time.time()

        answer, reasoning = answer_question(
            str(image_path),
            model,
            processor,
            device,
            enable_thinking=enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )

        elapsed = time.time() - t0
        log.info(f"  → Answer: {answer}  ({elapsed:.1f}s)")

        # Log first few lines of reasoning for debugging
        reasoning_preview = reasoning.replace("\n", " ")[:150]
        log.info(f"  → Reasoning: {reasoning_preview}...")

        results.append({"image_name": image_name, "option": answer})

        # Free memory between questions
        clear_cache(device)

    total_elapsed = time.time() - total_t0

    # Build submission
    df_submission = pd.DataFrame(results)
    df_submission.to_csv(args.output, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"DONE — {len(results)} questions answered in {total_elapsed:.1f}s")
    log.info(f"Average: {total_elapsed/max(len(results),1):.1f}s per question")
    log.info(f"Submission saved to: {args.output}")
    log.info(f"Answer distribution: {df_submission['option'].value_counts().to_dict()}")
    log.info(f"{'='*60}")

    # Final validation — ensure no hallucinated values
    invalid = df_submission[~df_submission["option"].isin(VALID_ANSWERS)]
    if len(invalid) > 0:
        log.error(f"CRITICAL: Found {len(invalid)} invalid answers! Fixing to {ABSTAIN}...")
        df_submission.loc[~df_submission["option"].isin(VALID_ANSWERS), "option"] = ABSTAIN
        df_submission.to_csv(args.output, index=False)
        log.info(f"Fixed submission saved to: {args.output}")


if __name__ == "__main__":
    main()
