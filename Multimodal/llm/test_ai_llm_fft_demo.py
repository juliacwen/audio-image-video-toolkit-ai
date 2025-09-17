#!/usr/bin/env python3
"""
test_ai_llm_fft_demo.py

 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-17

pytest wrapper for ai_llm_fft_demo
"""
import os
import pytest
from ai_llm_fft_demo import extract_windowed_fft, compute_summary, save_fft_excel, run_pipeline, DEVICE

TEST_DIR = "test/test_files"
OUTPUT_DIR = "test/test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

wav_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".wav")]

@pytest.mark.parametrize("wav_path,stage", [(f, "pre_llm") for f in wav_files] +
                                       [(f, "llm") for f in wav_files])
def test_fft_pipeline_split(wav_path, stage):
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_base = os.path.join(OUTPUT_DIR, base_name)

    # Print detected device
    print(f"\n⚙ Running pipeline on device: {DEVICE}")

    # Run the pipeline
    run_pipeline(wav_path, output_dir=OUTPUT_DIR, show_plot=False, verbose=False)

    # Check that all outputs exist
    excel_file = f"{output_base}_fft.xlsx"
    json_file = f"{output_base}_fft_log.json"
    png_file = f"{output_base}_fft.png"

    for file_path in [excel_file, json_file, png_file]:
        assert os.path.isfile(file_path), f"Expected output file not found: {file_path}"

    # Basic FFT checks
    rate, window_size, fft_features = extract_windowed_fft(wav_path, window_size=256, hop_size=128)
    summaries = {name: compute_summary(rate, window_size, arr) for name, arr in fft_features.items()}

    if stage == "pre_llm":
        try:
            save_fft_excel(fft_features, output_base)
        except Exception:
            pytest.fail(f"Pre-LLM stage FAILED for {wav_path}")
    else:
        if os.getenv("OPENAI_API_KEY"):
            try:
                from ai_llm_fft_demo import run_llm_explanation_if_available
                llm_result, _ = run_llm_explanation_if_available(rate, window_size, summaries)
                if not llm_result:
                    pytest.fail(f"LLM stage FAILED for {wav_path}")
            except Exception as e:
                pytest.fail(f"LLM stage FAILED for {wav_path}: {e}")
        else:
            pytest.skip(f"OPENAI_API_KEY not set; skipping LLM stage")

