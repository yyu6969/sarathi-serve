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

from sarathi.config import ModelConfig, ParallelConfig, SarathiSchedulerConfig, MetricsConfig, SystemConfig, ReplicaConfig
from sarathi import LLMEngine, SamplingParams, RequestOutput
from sarathi.metrics.constants import SequenceMetricsTimeDistributions

OUTPUT_DIR = "/work/nvme/bdkz/yyu69/sarathi-serve/experiment_results"

def load_prompts(path: str) -> List[str]:
    """
    Load prompts from a JSON file based on size ('small', 'medium', 'large', 'xlarge')
    Returns a list of 4 prompts for the given size
    """
    try:
        with open(path, "r") as f:
            prompts_data = json.load(f)
            if len(prompts_data) < 4:
                raise ValueError(f"Not enough prompts in {path}. Expected 4, got {len(prompts_data)}")
            return prompts_data[:4]  # Take first 4 prompts if more are provided
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file {path} not found. Please create the file first.")

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
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )

    scheduler_config = SarathiSchedulerConfig(
        chunk_size=chunk_size,
        max_num_seqs=8,
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
    
    # Calculate average e2e time
    e2e_times = [y for _, y in e2e_time_metrics.data_series]
    avg_e2e_time = sum(e2e_times) / len(e2e_times) if e2e_times else 0

    # !!! ADD SHUTDOWN CALL HERE !!!
    print("Shutting down LLM engine...")
    try:
        if hasattr(llm_engine, 'shutdown'):
            llm_engine.shutdown()
        elif hasattr(llm_engine, 'close'):
             llm_engine.close()
        else:
            print("Warning: Could not find explicit engine shutdown method.")
    except Exception as e:
        print(f"Error during engine shutdown: {e}")
    print("Engine shutdown complete.")
    
    # Print summary for this run
    print("===========================================================")
    print(f"Chunk size: {chunk_size}")
    print(f"e2e time: {avg_e2e_time:.4f} seconds")
    print("===========================================================")

    return {
        "chunk_size": chunk_size,
        "avg_e2e_time": avg_e2e_time,
        "e2e_times": e2e_times,
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
    plt.title('Chunk Size vs Request End-to-End Time\nComparison of Different Prompt Lengths')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    e2e_plot_path = f"{results_dir}/combined_chunk_size_vs_e2e_time.png"
    plt.savefig(e2e_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined E2E time plot saved to {e2e_plot_path}")
    
    plt.close()

def main():
    # Create timestamp for the experiment
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
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
            "path": "/work/nvme/bdkz/yyu69/sarathi-serve/data/prompt_4096.json",
            "chunk_sizes": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        },
    ]
    
    all_results = []
    
    # Run experiments for each prompt file with its corresponding chunk sizes
    for config in prompt_configs:
        results = []
        prompts = load_prompts(config["path"])
        
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
    
    print(f"\nExperiment completed. Results saved to {args.output_dir}/chunk_size_experiment_{timestamp}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save experiment results and plots")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use for inference")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional name for the experiment (will be included in output directory)")
    args = parser.parse_args()
    
    main()