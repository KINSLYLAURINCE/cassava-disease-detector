"""
Run both training scripts sequentially then evaluate.
Expected total time: ~8-13 hours on CPU.
Usage: python run_all_training.py
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def run_script(script_name, label):
    print(f"\n{'='*60}")
    print(f"STARTING: {label}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,   # stream output live to terminal
    )

    elapsed = time.time() - t0
    hrs = elapsed / 3600
    status = "DONE" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n{label}: {status} in {hrs:.1f} hours")
    return result.returncode == 0, elapsed


def main():
    scripts = [
        ("train_image_classifier.py", "MobileNetV2 Image Classifier (~2-3 hrs)"),
        ("train_llm_qlora.py",         "TinyLlama LoRA Fine-tuning (~6-10 hrs)"),
        ("evaluate_models.py",         "Final Evaluation (~20 min)"),
    ]

    overall_start = time.time()
    summary = []

    for script, label in scripts:
        if not Path(script).exists():
            print(f"ERROR: {script} not found, skipping.")
            summary.append({"script": script, "status": "not found"})
            continue

        success, elapsed = run_script(script, label)
        summary.append({
            "script": script,
            "label": label,
            "status": "success" if success else "failed",
            "elapsed_hours": round(elapsed / 3600, 2),
        })

        if not success:
            print(f"\nERROR: {script} failed. Stopping pipeline.")
            break

    total_hours = (time.time() - overall_start) / 3600

    print(f"\n{'='*60}")
    print("TRAINING PIPELINE SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        print(f"  {s['status'].upper():8s} | {s.get('elapsed_hours', 0):.1f}h | {s['label']}")
    print(f"\nTotal time: {total_hours:.1f} hours")

    with open("data/training_pipeline_summary.json", "w") as f:
        json.dump({"summary": summary, "total_hours": round(total_hours, 2)}, f, indent=2)
    print(f"Summary saved to data/training_pipeline_summary.json")


if __name__ == "__main__":
    main()
