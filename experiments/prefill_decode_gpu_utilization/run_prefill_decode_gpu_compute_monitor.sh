#!/bin/bash
# experiments/run_prefill_decode_gpu_compute_monitor.sh

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Create logs directory
mkdir -p logs

# Create a timestamp for output
timestamp=$(date +"%Y%m%d_%H%M%S")

# Define where we’ll store the generated charts
output_dir="/work/hdd/bdkz/yyu69/verl/experiment_results/compute/${timestamp}"
mkdir -p "${output_dir}"

# Define the log file name
log_file="logs/gpu_compute_${timestamp}.log"

echo "Running GPU compute monitor..."
echo "Logs will be saved to: ${log_file}"
echo "Charts will be saved to: ${output_dir}"
echo

# Execute the Python script with the necessary arguments
python -u experiments/prefill_decode_gpu_compute/prefill_decode_gpu_compute_monitor.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt-file /work/hdd/bdkz/yyu69/verl/experiments/prompts/test_2/5_prompts.json \
    --output-dir "${output_dir}" \
    --max-new-tokens 50 \
    --use-wandb \
    --wandb-project "gpu-compute-monitor" \
    --wandb-run-name "test" \
    2>&1 | tee "${log_file}"

echo
echo "Monitoring complete."
echo "Output charts are located in: ${output_dir}"
echo "Log file is located at: ${log_file}"