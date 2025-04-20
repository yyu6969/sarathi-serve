import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import tempfile
import json
import sys
from io import StringIO
import wandb

from sarathi.config import ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig, SystemConfig, ReplicaConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.metrics.constants import SequenceMetricsTimeDistributions

OUTPUT_DIR = "/work/nvme/bdkz/yyu69/sarathi-serve/experiment_results"

def load_prompts(path: str) -> List[str]:
    try:
        with open(path, "r") as f:
            prompts_data = json.load(f)
            return prompts_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file {path} not found. Please create the file first.")

def load_prompts_from_csv(path: str, column_name: str = "question") -> List[str]:
    """
    Load prompts from a CSV file using the specified column.
    
    Args:
        path: Path to the CSV file containing prompts
        column_name: Name of the column containing the prompts (default: "question")
        
    Returns:
        List of prompt strings
    """
    try:
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Check if the column exists
        if column_name not in df.columns:
            available_columns = ", ".join(df.columns)
            raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {available_columns}")
        
        # Extract prompts from the specified column
        prompts = df[column_name].tolist()
        
        # Remove any NaN values and convert to strings
        prompts = [str(prompt) for prompt in prompts if not pd.isna(prompt)]
        
        print(f"Loaded {len(prompts)} prompts from {path}")
        return prompts[:8]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {path} not found. Please create the file first.")
    except Exception as e:
        raise Exception(f"Error loading prompts from CSV: {str(e)}")

def generate(llm_engine: LLMEngine, prompts: List[str], sampling_params: SamplingParams) -> List[RequestOutput]:
    for prompt in prompts:
        llm_engine.add_request(prompt, sampling_params)

    num_requests = llm_engine.get_num_unfinished_requests()
    pbar = tqdm(total=num_requests, desc="Processed prompts")

    # Run the engine
    outputs: List[RequestOutput] = []
    while llm_engine.has_unfinished_requests():
        step_outputs = llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                pbar.update(1)

    pbar.close()
    # Sort the outputs by request ID.
    # This is necessary because some requests may be finished earlier than
    # its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.seq_id))
    return outputs

def run_inference_with_chunk_size(chunk_size: int, prompts: List[str], sampling_params: SamplingParams, model_name: str, base_output_dir: str) -> Dict:
    import time
    time.sleep(2)
    
    # Use a temporary directory for model outputs
    import tempfile
    temp_dir = tempfile.mkdtemp()

    # Get tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count tokens for all prompts and calculate average
    prompt_tokens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens)
    print(f"Token counts for prompts: {prompt_tokens}")
    print(f"Average token count: {avg_prompt_tokens}")

    replica_config = ReplicaConfig(
        output_dir=temp_dir,  # Use temporary directory instead of base_output_dir
    )

    model_config = ModelConfig(
        model=model_name,
        dtype="float16",
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    scheduler_config = SarathiSchedulerConfig(
        chunk_size=chunk_size,
        max_num_seqs=16,
    )

    metrics_config = MetricsConfig(
        write_metrics=True,
        enable_chrome_trace=True,
    )

    system_config = SystemConfig(
        replica_config=replica_config,
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        metrics_config=metrics_config,
    )

    llm_engine = LLMEngine.from_system_config(system_config)

    # Generate texts from the prompts
    outputs = generate(llm_engine, prompts, sampling_params)
    
    # Print prompts and answers
    print("\n----- PROMPTS AND ANSWERS -----")
    for i, output in enumerate(outputs):
        print(f"\nRequest {i+1}:")
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.text}")
    print("----- END OF PROMPTS AND ANSWERS -----\n")
    
    # Pull metrics to analyze
    llm_engine.pull_worker_metrics()
    llm_engine.plot_metrics()
    
    # Extract the e2e time metrics
    metrics_store = llm_engine.get_metric_store()
    e2e_time_metrics = metrics_store.seq_metrics_time_distributions[SequenceMetricsTimeDistributions.REQUEST_E2E_TIME]
    prefill_time_metrics = metrics_store.seq_metrics_time_distributions[SequenceMetricsTimeDistributions.PREFILL_TIME_E2E]
    
    # Extract decode time metrics from the normalized distribution
    decode_time_metrics_normalized = metrics_store.seq_metrics_time_distributions[SequenceMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED]
    
    # Calculate average e2e time
    e2e_times = [y for _, y in e2e_time_metrics.data_series]
    avg_e2e_time = sum(e2e_times) / len(e2e_times) if e2e_times else 0
    
    # Calculate prefill times
    prefill_times = [y for _, y in prefill_time_metrics.data_series]
    avg_prefill_time = sum(prefill_times) / len(prefill_times) if prefill_times else 0
    
    # Calculate decode times (total decode time = e2e time - prefill time)
    decode_times = [e - p for e, p in zip(e2e_times, prefill_times)]
    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
    
    # Calculate avg token/sec for decode phase
    output_tokens = [len(output.text.split()) for output in outputs]
    token_per_sec = [tokens / time for tokens, time in zip(output_tokens, decode_times)]
    avg_token_per_sec = sum(token_per_sec) / len(token_per_sec) if token_per_sec else 0
    
    # Print summary for this run
    print("===========================================================")
    print(f"Chunk size: {chunk_size}")
    print(f"e2e time: {avg_e2e_time:.4f} seconds")
    print("===========================================================")

    return {
        "chunk_size": chunk_size,
        "avg_e2e_time": avg_e2e_time,
        "e2e_times": e2e_times,
        "avg_prefill_time": avg_prefill_time,
        "prefill_times": prefill_times,
        "avg_decode_time": avg_decode_time,
        "decode_times": decode_times,
        "avg_token_per_sec": avg_token_per_sec,
        "prompt_tokens": prompt_tokens,
        "avg_prompt_tokens": avg_prompt_tokens
    }

def plot_combined_results(all_results: List[List[Dict]], output_dir: str, timestamp: str = None):
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    results_dir = f"{output_dir}/chunk_size_experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot combined e2e time chart
    plt.figure(figsize=(12, 8))
    
    # Colors and markers for different prompts
    styles = [
        {'color': 'blue', 'marker': 'o', 'label': 'Prompt 1'},
        {'color': 'green', 'marker': 's', 'label': 'Prompt 2'},
        {'color': 'red', 'marker': '^', 'label': 'Prompt 3'}
    ]
    
    # Get all unique chunk sizes and create evenly spaced x positions
    all_chunk_sizes = sorted(list(set([
        size for results in all_results 
        for result in results 
        for size in [result["chunk_size"]]
    ])))
    
    # Create evenly spaced x positions for plotting
    x_positions = np.arange(len(all_chunk_sizes))
    
    # Create mapping from chunk size to x position
    chunk_size_to_pos = {size: pos for pos, size in zip(x_positions, all_chunk_sizes)}
    
    # Plot each prompt's results
    for idx, results in enumerate(all_results):
        chunk_sizes = [result["chunk_size"] for result in results]
        avg_e2e_times = [result["avg_e2e_time"] for result in results]
        avg_prompt_tokens = results[0]["avg_prompt_tokens"]
        
        # Sort by chunk size
        sorted_indices = np.argsort(chunk_sizes)
        sorted_chunk_sizes = [chunk_sizes[i] for i in sorted_indices]
        sorted_e2e_times = [avg_e2e_times[i] for i in sorted_indices]
        
        # Convert chunk sizes to x positions
        x_vals = [chunk_size_to_pos[size] for size in sorted_chunk_sizes]
        
        # Plot with different style for each prompt
        plt.plot(x_vals, sorted_e2e_times, 
                marker=styles[idx]['marker'], 
                color=styles[idx]['color'],
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=f'{styles[idx]["label"]} (avg {avg_prompt_tokens:.1f} tokens)')
        
        # Add value annotations
        for x, y, chunk_size in zip(x_vals, sorted_e2e_times, sorted_chunk_sizes):
            plt.annotate(f"{y:.2f}s", 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        color=styles[idx]['color'],
                        fontsize=8)
    
    # Set x-axis ticks to show all chunk sizes
    plt.xticks(x_positions, all_chunk_sizes, rotation=45)
    
    plt.xlabel('Chunk Size')
    plt.ylabel('E2E Time (s)')
    plt.title('Chunk Size vs Request End-to-End Time\nComparison of Different Prompt Lengths (llama-3-8b)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    e2e_plot_path = f"{results_dir}/combined_chunk_size_vs_e2e_time.png"
    plt.savefig(e2e_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined E2E time plot saved to {e2e_plot_path}")
    
    plt.close()
    
    # Save detailed timing results as CSV
    detailed_results = []
    for prompt_idx, results in enumerate(all_results):
        for result in results:
            chunk_size = result["chunk_size"]
            for i in range(len(result["e2e_times"])):
                detailed_results.append({
                    "prompt_idx": prompt_idx + 1,
                    "chunk_size": chunk_size,
                    "e2e_time": result["e2e_times"][i],
                    "prefill_time": result["prefill_times"][i],
                    "decode_time": result["decode_times"][i],
                    "prompt_tokens": result["prompt_tokens"][i] if i < len(result["prompt_tokens"]) else None,
                })
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = f"{results_dir}/detailed_timing_metrics.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed timing metrics saved to {detailed_csv_path}")
    
    # Create summary table with averages
    summary_results = []
    for prompt_idx, results in enumerate(all_results):
        for result in results:
            summary_results.append({
                "prompt_idx": prompt_idx + 1,
                "chunk_size": result["chunk_size"],
                "avg_prompt_tokens": result["avg_prompt_tokens"],
                "avg_e2e_time": result["avg_e2e_time"],
                "avg_prefill_time": result["avg_prefill_time"],
                "avg_decode_time": result["avg_decode_time"],
                "avg_token_per_sec": result["avg_token_per_sec"],
            })
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = f"{results_dir}/timing_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Timing summary saved to {summary_csv_path}")
    
    # Create an additional plot for prefill and decode times
    plt.figure(figsize=(12, 8))
    
    # For each chunk size, plot e2e, prefill, and decode times as stacked bar chart
    chunk_sizes = sorted(list(set([result["chunk_size"] for results in all_results for result in results])))
    x = np.arange(len(chunk_sizes))
    width = 0.25
    
    # Get averages for each chunk size
    chunk_size_avg = {}
    for chunk_size in chunk_sizes:
        chunk_results = [r for results in all_results for r in results if r["chunk_size"] == chunk_size]
        chunk_size_avg[chunk_size] = {
            "prefill": sum(r["avg_prefill_time"] for r in chunk_results) / len(chunk_results),
            "decode": sum(r["avg_decode_time"] for r in chunk_results) / len(chunk_results)
        }
    
    # Plot prefill times
    prefill_times = [chunk_size_avg[cs]["prefill"] for cs in chunk_sizes]
    decode_times = [chunk_size_avg[cs]["decode"] for cs in chunk_sizes]
    
    plt.bar(x, prefill_times, width, label='Prefill Time')
    plt.bar(x, decode_times, width, bottom=prefill_times, label='Decode Time')
    
    plt.xlabel('Chunk Size')
    plt.ylabel('Time (s)')
    plt.title('Prefill and Decode Times by Chunk Size')
    plt.xticks(x, chunk_sizes)
    plt.legend()
    
    # Save timing breakdown plot
    timing_plot_path = f"{results_dir}/prefill_decode_time_breakdown.png"
    plt.savefig(timing_plot_path, dpi=300, bbox_inches='tight')
    print(f"Timing breakdown plot saved to {timing_plot_path}")
    
    plt.close()

def main():
    # Create timestamp for the experiment
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Initialize wandb with a dummy project if you don't want to use wandb directly
    wandb_run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"chunk_size_experiment_{timestamp}",
        config={
            "model": args.model,
            "output_dir": args.output_dir,
            "experiment_timestamp": timestamp,
            "experiment_name": args.experiment_name,
        }
    )
    
    print(f"Starting experiment at {timestamp}")
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling parameters: temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}\n")
    
    # Define chunk sizes and prompt paths
    prompt_configs = [
        {
            "path": "/work/nvme/bdkz/yyu69/sarathi-serve/data/long-questions-in-train-set.csv",
            "chunk_sizes": [16, 32, 64, 128, 256, 512, 1024, 2048]
        },
    ]
    
    all_results = []
    
    # Run experiments for each prompt file with its corresponding chunk sizes
    for config in prompt_configs:
        results = []
        prompts = load_prompts_from_csv(config["path"])
        
        print(f"\nRunning inference for prompts from {config['path']}")
        for chunk_size in config["chunk_sizes"]:
            print(f"\nRunning with chunk_size = {chunk_size}")
            result = run_inference_with_chunk_size(
                chunk_size=chunk_size,
                prompts=prompts,
                sampling_params=sampling_params,
                model_name=args.model,
                base_output_dir=None
            )
            results.append(result)
        
        all_results.append(results)
    
    # Plot combined results
    plot_combined_results(all_results, args.output_dir, timestamp)
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save experiment results and plots")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Model to use for inference")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional name for the experiment (will be included in output directory)")
    parser.add_argument("--wandb-project", type=str, default="sarathi-chunk-size-experiment", help="Weights & Biases project name")
    parser.add_argument("--wandb-group", type=str, default="chunk-size-vs-e2e-time", help="Weights & Biases group name")
    args = parser.parse_args()
    
    main()