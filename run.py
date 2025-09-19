import os
import subprocess
from pathlib import Path
import sys

# ===== Config =====
MODEL_BASE_DIR = Path("./models")
OUTPUT_BASE_DIR = Path("./Output_Withstage")
LOG_DIR = Path("./logs")
MAX_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

MODELS = [
    "Qwen_Qwen2.5-Coder-3B-Instruct"
]

PROMPTS = ["SR"]

# ===== Make sure directories exist =====
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# ===== Helper functions =====
def check_model_exists(model_path):
    if not model_path.exists() or not model_path.is_dir():
        print(f"Warning: Model directory does not exist: {model_path}")
        return False
    return True

def run_evaluation(model_name, prompt_name):
    model_path = MODEL_BASE_DIR / model_name
    output_dir = OUTPUT_BASE_DIR / f"{model_name}_{prompt_name}"
    log_file = LOG_DIR / f"{model_name}_{prompt_name}.log"

    if not check_model_exists(model_path):
        print(f"Skipping {model_name} (model not found)")
        return "FAILED", 0

    output_dir.mkdir(exist_ok=True)

    # Run the Python evaluation script
    cmd = [
        sys.executable,  # ensures the current Python interpreter is used
        "main-Withstages_localLLM.py",
        "--prompt", prompt_name,
        "--run-stage1",
        "--no-stage2",
        "--max-questions", str(MAX_QUESTIONS),
        "--model-path", str(model_path),
        "--output-dir", str(output_dir)
    ]

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    return status, result.returncode

# ===== Main loop =====
summary_file = LOG_DIR / "evaluation_summary.txt"
with open(summary_file, "w") as f:
    f.write(f"DebugBench Evaluation Summary\n")
    f.write("="*40 + "\n")

total_runs = 0
successful_runs = 0
failed_runs = 0

for model in MODELS:
    for prompt in PROMPTS:
        print(f"Running: {model} + {prompt}")
        status, _ = run_evaluation(model, prompt)
        f = open(summary_file, "a")
        f.write(f"{model} + {prompt}: {status}\n")
        f.close()

        total_runs += 1
        if status == "SUCCESS":
            successful_runs += 1
        else:
            failed_runs += 1

# ===== Final summary =====
with open(summary_file, "a") as f:
    f.write(f"\nTotal runs: {total_runs}\n")
    f.write(f"Successful: {successful_runs}\n")
    f.write(f"Failed: {failed_runs}\n")
    f.write(f"Success rate: {successful_runs*100/total_runs:.2f}%\n")

print("All evaluations completed.")
