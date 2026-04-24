#!/usr/bin/env python3
"""
Test script for the GNR638 MCQ pipeline.
Tests all logic WITHOUT requiring model weights to be downloaded.
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Import the functions we want to test from solve_mcq
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from solve_mcq import (
    ABSTAIN,
    SYSTEM_PROMPT,
    VALID_ANSWERS,
    clear_cache,
    get_device_config,
    parse_answer,
)

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}  {detail}")


# ===================================================================
# Test 1: parse_answer
# ===================================================================
print("\n🧪 Test 1: parse_answer")

# Standard ANSWER: X format
check("ANSWER: 1", parse_answer("ANSWER: 1") == 1)
check("ANSWER: 2", parse_answer("ANSWER: 2") == 2)
check("ANSWER: 3", parse_answer("ANSWER: 3") == 3)
check("ANSWER: 4", parse_answer("ANSWER: 4") == 4)
check("ANSWER: 5 (abstain)", parse_answer("ANSWER: 5") == 5)

# Case insensitivity
check("answer: 3", parse_answer("answer: 3") == 3)
check("Answer: 2", parse_answer("Answer: 2") == 2)

# With surrounding text
check("Embedded ANSWER", parse_answer("The correct choice is A.\nANSWER: 1") == 1)
check("Thinking then ANSWER", parse_answer(
    "Let me think step by step...\n"
    "The neural network has 3 layers...\n"
    "Option A matches because...\n"
    "ANSWER: 1"
) == 1)

# Whitespace variations
check("ANSWER:1 (no space)", parse_answer("ANSWER:1") == 1)
check("ANSWER:  3 (extra space)", parse_answer("ANSWER:  3") == 3)

# Fallback: last digit in text
check("Fallback digit", parse_answer("The answer is option 2") == 2)
check("Fallback last digit", parse_answer("Options are 1,2,3 but correct is 4") == 4)

# No valid answer
check("No answer -> None", parse_answer("I don't know") is None)
check("Empty string -> None", parse_answer("") is None)

# Edge: digit outside 1-5
check("Digit 0 ignored", parse_answer("The answer is 0") is None)
check("Digit 6 ignored", parse_answer("There are 6 layers") is None)

# Thinking mode output format (with <think> tags stripped)
check("After thinking block", parse_answer(
    "Let me analyze this step by step.\n"
    "The question asks about CNN output size.\n"
    "Conv layer: (64-3+2*1)/2 + 1 = 32\n"
    "MaxPool: 32/2 = 16\n"
    "So the output is 16x16, which is option A.\n"
    "ANSWER: 1"
) == 1)


# ===================================================================
# Test 2: get_device_config
# ===================================================================
print("\n🧪 Test 2: get_device_config")

config = get_device_config()
check("Returns dict", isinstance(config, dict))
check("Has 'device' key", "device" in config)
check("Has 'device_map' key", "device_map" in config)
check("Has 'torch_dtype' key", "torch_dtype" in config)
check("Has 'attn_implementation' key", "attn_implementation" in config)
check("Device is mps/cuda/cpu", config["device"] in ("mps", "cuda", "cpu"))
print(f"     Detected device: {config['device']}")


# ===================================================================
# Test 3: clear_cache (should not crash on any device)
# ===================================================================
print("\n🧪 Test 3: clear_cache")

try:
    clear_cache("cpu")
    check("clear_cache('cpu')", True)
except Exception as e:
    check("clear_cache('cpu')", False, str(e))

try:
    clear_cache(config["device"])
    check(f"clear_cache('{config['device']}')", True)
except Exception as e:
    check(f"clear_cache('{config['device']}')", False, str(e))


# ===================================================================
# Test 4: CSV reading and submission format
# ===================================================================
print("\n🧪 Test 4: CSV / submission format")

test_csv = Path("sample_test_project_2/test.csv")
sample_sub = Path("sample_test_project_2/sample_submission.csv")

check("test.csv exists", test_csv.exists())
check("sample_submission.csv exists", sample_sub.exists())

df_test = pd.read_csv(test_csv)
check("test.csv has image_name column", "image_name" in df_test.columns)
check("test.csv has 2 rows", len(df_test) == 2)

df_sample = pd.read_csv(sample_sub)
check("sample_submission has image_name column", "image_name" in df_sample.columns)
check("sample_submission has option column", "option" in df_sample.columns)

# Verify image files exist
images_dir = Path("sample_test_project_2/images")
for _, row in df_test.iterrows():
    img_path = images_dir / f"{row['image_name']}.png"
    check(f"Image exists: {row['image_name']}.png", img_path.exists())

# Test submission generation
results = [
    {"image_name": "image_1", "option": 1},
    {"image_name": "image_2", "option": 3},
]
df_out = pd.DataFrame(results)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    df_out.to_csv(f.name, index=False)
    df_read = pd.read_csv(f.name)
    check("Output CSV columns match", list(df_read.columns) == ["image_name", "option"])
    check("Output CSV rows match", len(df_read) == 2)
    check("All options valid", all(v in VALID_ANSWERS for v in df_read["option"]))
    os.unlink(f.name)


# ===================================================================
# Test 5: Hallucination guard
# ===================================================================
print("\n🧪 Test 5: Hallucination guard")

# Simulate output validation
test_outputs = [1, 2, 3, 4, 5, 0, 6, -1, 99, "A"]
for val in test_outputs:
    is_valid = val in VALID_ANSWERS
    if val in (1, 2, 3, 4, 5):
        check(f"Value {val} is valid", is_valid)
    else:
        check(f"Value {val} is INVALID (would be caught)", not is_valid)


# ===================================================================
# Test 6: System prompt quality
# ===================================================================
print("\n🧪 Test 6: System prompt checks")

check("Prompt mentions step-by-step", "step-by-step" in SYSTEM_PROMPT.lower())
check("Prompt mentions ANSWER:", "ANSWER:" in SYSTEM_PROMPT)
check("Prompt mentions options 1-4", "1, 2, 3, or 4" in SYSTEM_PROMPT)
check("Prompt mentions abstain (5)", "5" in SYSTEM_PROMPT)
check("ABSTAIN value is 5", ABSTAIN == 5)


# ===================================================================
# Test 7: End-to-end mock pipeline
# ===================================================================
print("\n🧪 Test 7: End-to-end mock pipeline (simulated)")

# Simulate what happens with model outputs
mock_outputs = {
    "image_1": "Let me analyze this MLP question.\nThe network is defined as h1=ReLU(W1*x+b1), h2=ReLU(W2*h1+b2), y=W3*h2+b3.\nThis requires Linear->ReLU->Linear->ReLU->Linear.\nOption A matches: nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, out_dim))\nANSWER: 1",
    "image_2": "Input: 64x64\nConv(k=3, s=2, p=1): output = floor((64-3+2*1)/2) + 1 = 32\nMaxPool(k=2): output = 32/2 = 16\nFinal size: 16x16, which is option A.\nANSWER: 1",
}

mock_results = []
for image_name, output in mock_outputs.items():
    answer = parse_answer(output)
    check(f"Mock {image_name} -> answer={answer}", answer is not None and answer in VALID_ANSWERS)
    mock_results.append({"image_name": image_name, "option": answer})

df_mock = pd.DataFrame(mock_results)
check("Mock submission has correct shape", df_mock.shape == (2, 2))
check("Mock submission all valid", all(v in VALID_ANSWERS for v in df_mock["option"]))

# Expected answers for sample images
check("Image 1 (MLP→PyTorch) = 1 (option A)", mock_results[0]["option"] == 1)
check("Image 2 (CNN shape) = 1 (option A: 16×16)", mock_results[1]["option"] == 1)


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED} tests")
print(f"{'='*60}")

if FAILED > 0:
    sys.exit(1)
else:
    print("🎉 All tests passed! Pipeline logic is verified.")
    print("   Model download is needed for actual inference.")
    sys.exit(0)
