#!/bin/bash

# Script to run DebugBench evaluation with different model and prompt combinations
# Only uses models with 13B parameters or less

# Define the base directory for models
# MODEL_BASE_DIR="/home/jackcp/aic-nas2/llm/llms_download/models"
MODEL_BASE_DIR="./models"

# Define models (≤13B parameters only)
MODELS=(
    # "codellama_CodeLlama-7b-Instruct-hf"
    # "codellama_CodeLlama-13b-Instruct-hf"
    # "deepseek-ai_deepseek-coder-1.3b-instruct"
    # "meta-llama_Llama-2-7b-chat-hf"
    # "deepseek-ai_DeepSeek-Coder-V2-Instruct-0724"
    # "meta-llama_Llama-2-13b-chat-hf"    
    # "WizardLMTeam_WizardCoder-15B-V1.0"
    # "Qwen_Qwen3-Coder-30B-A3B-Instruct"
    # "deepseek-ai_DeepSeek-Coder-V2-Lite-Instruct"
    # "deepseek-ai_deepseek-coder-33b-instruct"
    # "Phind_Phind-CodeLlama-34B-v2"
    # "Qwen_CodeQwen1.5-7B"
    # "ByteDance-Seed_Seed-Coder-8B-Instruct"
    "Qwen_Qwen2.5-Coder-3B-Instruct"
    # "Qwen_Qwen2.5-Coder-7B-Instruct"
    # "Qwen_Qwen2.5-Coder-3B-Instruct"
    # "models--ByteDance--Seed--Seed-Coder-8B-Instruct"
    # "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    # "models--Qwen--Qwen2.5-Coder-14B-Instruct"
)

# Define prompts
PROMPTS=(
    # "DF_pseu"
    # "pearl"  
    # "DF"
    # "single"
    # "react"
    # "LDB"
    # "flow"
    # "coa"
    "SR"
)

# Default configuration
MAX_QUESTIONS=100  # Default value
# LANGUAGE="python3"
LOG_DIR="logs"
OUTPUT_BASE_DIR="Output_Withstage"

# Function to display usage
usage() {
    echo "Usage: $0 [--max-questions NUMBER]"
    echo "Options:"
    echo "  --max-questions NUMBER    Set the maximum number of questions (default: 1500)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-questions)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                MAX_QUESTIONS="$2"
                shift 2
            else
                echo "Error: --max-questions requires a valid number"
                usage
            fi
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

echo "Running DebugBench evaluation with MAX_QUESTIONS=$MAX_QUESTIONS"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to check if model directory exists
check_model_exists() {
    local model_path="$1"
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model directory does not exist: $model_path"
        return 1
    fi
    return 0
}

# Function to check GPU memory
check_gpu_memory() {
    echo "=== GPU Memory Status ==="
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    echo "========================="
}

# Function to run evaluation
run_evaluation() {
    local model_name="$1"
    local prompt_name="$2"
    local model_path="$MODEL_BASE_DIR/$model_name"
    local output_dir="$OUTPUT_BASE_DIR/${model_name}_${prompt_name}"
    local log_file="$LOG_DIR/${model_name}_${prompt_name}.log"
    
    echo "Starting evaluation:"
    echo "  Model: $model_name"
    echo "  Prompt: $prompt_name"
    echo "  Output: $output_dir"
    echo "  Log: $log_file"
    
    # Check if model exists
    if ! check_model_exists "$model_path"; then
        echo "Skipping $model_name (model not found)"
        return 1
    fi
    
    # Check GPU memory before starting
    check_gpu_memory
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the evaluation
    python main-Withstages_localLLM.py \
        --prompt "$prompt_name" \
        --run-stage1 \
        --no-stage2 \
        --max-questions "$MAX_QUESTIONS" \
        --model-path "$model_path" \
        --output-dir "$output_dir" \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $model_name with $prompt_name"
    else
        echo "❌ Failed: $model_name with $prompt_name (exit code: $exit_code)"
    fi
    
    # Clear GPU cache between runs
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null
    
    return $exit_code
}

# Main execution
echo "Starting DebugBench evaluation with multiple models and prompts"
echo "Models to test: ${MODELS[*]}"
echo "Prompts to test: ${PROMPTS[*]}"
echo "Max questions per run: $MAX_QUESTIONS"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo

# Create summary file
SUMMARY_FILE="$LOG_DIR/evaluation_summary.txt"
echo "DebugBench Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"

total_runs=0
successful_runs=0
failed_runs=0

# Run all combinations
for model in "${MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        echo
        echo "=================================================="
        echo "Run $((total_runs + 1)): $model + $prompt"
        echo "=================================================="
        
        start_time=$(date +%s)
        
        if run_evaluation "$model" "$prompt"; then
            status="SUCCESS"
            ((successful_runs++))
        else
            status="FAILED"
            ((failed_runs++))
        fi
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Log to summary
        echo "$model + $prompt: $status (${duration}s)" >> "$SUMMARY_FILE"
        
        ((total_runs++))
        
        echo "Status: $status"
        echo "Duration: ${duration} seconds"
        echo "Progress: $total_runs/${#MODELS[@]}x${#PROMPTS[@]} runs completed"
        
        # Optional: Add delay between runs to let GPU cool down
        # sleep 10
    done
done

# Final summary
echo >> "$SUMMARY_FILE"
echo "Total runs: $total_runs" >> "$SUMMARY_FILE"
echo "Successful: $successful_runs" >> "$SUMMARY_FILE"
echo "Failed: $failed_runs" >> "$SUMMARY_FILE"
echo "Success rate: $(( successful_runs * 100 / total_runs ))%" >> "$SUMMARY_FILE"

echo
echo "=================================================="
echo "All evaluations completed!"
echo "Total runs: $total_runs"
echo "Successful: $successful_runs"
echo "Failed: $failed_runs"
echo "Success rate: $(( successful_runs * 100 / total_runs ))%"
echo "Summary saved to: $SUMMARY_FILE"
echo "Individual logs saved to: $LOG_DIR/"
echo "=================================================="

# Show final GPU status
check_gpu_memory