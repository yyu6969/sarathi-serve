import torch
import matplotlib.pyplot as plt
import time
import os
import json
import threading
import pynvml
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb

class PrefillDecodeGPUComputeMonitor:
    def __init__(self, device_index=0, poll_interval=0.01):
        self.device_index = device_index
        self.poll_interval = poll_interval

        # Timeline measurements
        self.prefill_data = {}       # prefill_data[prompt_id] -> list of (time, utilization)
        self.decode_data = {}        # decode_data[prompt_id]  -> list of (time, utilization)

        # Prompt info
        self.prompt_token_lengths = {}  # prompt_token_lengths[prompt_id] -> int

        # We'll store the raw generation outputs (token IDs) here:
        self.decoded_outputs = {}       # decoded_outputs[prompt_id] -> tensor from model.generate()

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def _poll_gpu_utilization(self, running_flag, data_store):
        """
        Continuously poll GPU utilization and store (elapsed_time, util)
        until running_flag[0] becomes False.
        """
        start_time = time.time()
        while running_flag[0]:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            elapsed = time.time() - start_time
            data_store.append((elapsed, util))
            time.sleep(self.poll_interval)

    def _run_with_monitoring(self, func):
        """
        1) Starts a monitoring thread
        2) Calls 'func()'
        3) Stops the monitoring thread
        4) Returns the collected data as a list of (time, util)
        """
        data_store = []
        running_flag = [True]
        monitor_thread = threading.Thread(
            target=self._poll_gpu_utilization,
            args=(running_flag, data_store)
        )
        monitor_thread.start()

        # Run the actual function
        func()

        # Signal the monitor thread to stop
        running_flag[0] = False
        monitor_thread.join()

        return data_store

    def record_prefill(self, prompt_id, model, inputs):
        """
        Record GPU utilization for the 'prefill' phase with batch processing.
        """
        # Get the single prompt and replicate it to batch size 8
        batch_size = 4
        input_ids = inputs['input_ids'].repeat(batch_size, 1)
        attention_mask = inputs['attention_mask'].repeat(batch_size, 1)
        batched_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Store the prompt token length (same for all in batch)
        prompt_token_len = inputs['input_ids'].shape[1]
        self.prompt_token_lengths[prompt_id] = prompt_token_len

        # Warm-up run without recording
        def warmup_func():
            with torch.no_grad():
                _ = model(**batched_inputs, use_cache=True)
        
        # Run warm-up without recording
        warmup_func()
        
        # Actual recorded run
        def prefill_func():
            with torch.no_grad():
                _ = model(**batched_inputs, use_cache=True)

        data = self._run_with_monitoring(prefill_func)
        self.prefill_data[prompt_id] = data

    def record_decode(self, prompt_id, model, inputs, max_new_tokens=50):
        """
        Record GPU utilization for the 'decode' phase with batch processing.
        """
        # Create batched inputs
        batch_size = 4
        input_ids = inputs['input_ids'].repeat(batch_size, 1)
        attention_mask = inputs['attention_mask'].repeat(batch_size, 1)
        batched_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        container = {}

        # Warm-up run without recording
        def warmup_func():
            outputs = model.generate(**batched_inputs, max_new_tokens=max_new_tokens)
        
        # Run warm-up without recording
        warmup_func()
        
        # Actual recorded run
        def decode_func():
            outputs = model.generate(**batched_inputs, max_new_tokens=max_new_tokens)
            container['outputs'] = outputs

        data = self._run_with_monitoring(decode_func)
        self.decode_data[prompt_id] = data

        # Save the generated token IDs for later decoding (we'll save all batch outputs)
        self.decoded_outputs[prompt_id] = container['outputs']

    def plot_time_utilization(self, output_dir="."):
        """
        Modified to show batch information in the plot titles and printouts.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_fig_paths = []

        all_prompt_ids = set(self.prefill_data.keys()) | set(self.decode_data.keys())

        for pid in sorted(all_prompt_ids):
            prefill = self.prefill_data.get(pid, [])
            decode = self.decode_data.get(pid, [])

            if not prefill and not decode:
                continue

            # Durations
            prefill_duration = prefill[-1][0] if prefill else 0.0
            decode_duration = decode[-1][0] if decode else 0.0

            # Token length
            token_len = self.prompt_token_lengths.get(pid, "N/A")

            print(f"Prompt {pid} (Batch Size = 8, with warm-up):")
            print(f"  Token length      = {token_len}")
            print(f"  Prefill duration  = {prefill_duration:.4f} seconds (avg per batch)")
            print(f"  Decode duration   = {decode_duration:.4f} seconds (avg per batch)\n")

            # Shift decode times so it follows prefill
            decode_shifted = []
            if prefill:
                for (t, util) in decode:
                    decode_shifted.append((t + prefill_duration, util))
            else:
                decode_shifted = decode

            fig, ax = plt.subplots()

            # Prefill
            if prefill:
                times_prefill = [x[0] for x in prefill]
                utils_prefill = [x[1] for x in prefill]
                ax.plot(times_prefill, utils_prefill, label="Prefill")

            # Decode
            if decode_shifted:
                times_decode = [x[0] for x in decode_shifted]
                utils_decode = [x[1] for x in decode_shifted]
                ax.plot(times_decode, utils_decode, label="Decode")

            ax.set_title(f"GPU Utilization - {pid} - {token_len} tokens (Batch Size = 8)")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("GPU Utilization (%)")
            ax.set_ylim(0, 100)
            ax.legend()

            save_path = os.path.join(output_dir, f"gpu_util_{pid}.png")
            plt.savefig(save_path)
            plt.close(fig)
            saved_fig_paths.append((pid, save_path))

        return saved_fig_paths

    def plot_bar_gpu_utilization(self, output_path="gpu_bar.png", title="GPU Compute Utilization: Prefill vs Decode"):
        """
        Create a side-by-side bar chart of average GPU utilization 
        for prefill vs. decode, keyed by prompt token length.
        
        Returns output_path so we can log it to wandb.
        """

        prompt_ids_sorted = sorted(set(self.prefill_data.keys()) | set(self.decode_data.keys()))
        
        token_lengths = []
        avg_prefill_utils = []
        avg_decode_utils = []

        for pid in prompt_ids_sorted:
            prefill = self.prefill_data.get(pid, [])
            decode = self.decode_data.get(pid, [])

            token_len = self.prompt_token_lengths.get(pid, 0)
            token_lengths.append(token_len)

            if len(prefill) > 0:
                avg_prefill = sum(u for (_, u) in prefill) / len(prefill)
            else:
                avg_prefill = 0.0

            if len(decode) > 0:
                avg_decode = sum(u for (_, u) in decode) / len(decode)
            else:
                avg_decode = 0.0

            avg_prefill_utils.append(avg_prefill)
            avg_decode_utils.append(avg_decode)

        x_indices = np.arange(len(prompt_ids_sorted))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars_prefill = ax.bar(
            x_indices - bar_width/2, avg_prefill_utils, 
            width=bar_width, label="Prefill", color="tab:blue"
        )
        bars_decode = ax.bar(
            x_indices + bar_width/2, avg_decode_utils, 
            width=bar_width, label="Decode", color="tab:orange"
        )

        ax.set_title(title)
        ax.set_ylabel("GPU Compute Utilization (%)")
        ax.set_ylim(0, 100)

        x_labels = [f"{tl} tokens" for tl in token_lengths]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=0)

        ax.legend()

        # Numeric labels
        for bar in bars_prefill:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bars_decode:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        fig.tight_layout()

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)

        return output_path

def run_compute_utilization_test(model_path, 
                                 prompt_file, 
                                 output_dir=".", 
                                 max_new_tokens=50,
                                 use_wandb=False,
                                 wandb_project="gpu-utilization",
                                 wandb_run_name=None):
    """
    Main driver function to:
      1. Load model & prompts
      2. Capture GPU utilization for prefill & decode per prompt
      3. Plot line charts (per prompt) + bar chart 
      4. Print out the generation text for each prompt
      5. Optionally log to Weights & Biases (wandb).
    """

    # Optionally init wandb
    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    monitor = PrefillDecodeGPUComputeMonitor(device_index=0, poll_interval=0.01)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    # Load prompts
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    # Run each prompt
    for i, prompt in enumerate(prompts):
        prompt_id = f"prompt_{i+1}"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Prefill measurement
        monitor.record_prefill(prompt_id, model, inputs)
        # Decode measurement
        monitor.record_decode(prompt_id, model, inputs, max_new_tokens=max_new_tokens)

    # -------------------------------------------------
    # Print out the generated text for each prompt here
    # -------------------------------------------------
    print("\n===== Generated Text Outputs =====")
    for pid in sorted(monitor.decoded_outputs.keys()):
        gen_tokens = monitor.decoded_outputs[pid]
        # decode (could be batch size > 1, but here likely 1)
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        # We'll assume single example, so gen_texts[0] 
        print(f"Prompt {pid} Generation:\n{gen_texts[0]}\n{'-'*40}")

    # 1) Plot the time-series
    line_plot_info = monitor.plot_time_utilization(output_dir)

    # 2) Plot bar chart
    bar_plot_path = os.path.join(output_dir, "gpu_bar.png")
    monitor.plot_bar_gpu_utilization(
        output_path=bar_plot_path, 
        title="GPU Compute Utilization: Prefill vs Decode"
    )

    # 3) If wandb, log the images + some metrics
    if use_wandb:
        # Log each line chart
        for (prompt_id, fig_path) in line_plot_info:
            wandb.log({f"util_plot_{prompt_id}": wandb.Image(fig_path)})

        # Log the bar chart
        wandb.log({"gpu_bar_chart": wandb.Image(bar_plot_path)})

        # Example: overall average
        total_prefill, count_prefill = 0.0, 0
        total_decode, count_decode = 0.0, 0
        for pid in monitor.prefill_data.keys():
            for (_, util) in monitor.prefill_data[pid]:
                total_prefill += util
                count_prefill += 1
        for pid in monitor.decode_data.keys():
            for (_, util) in monitor.decode_data[pid]:
                total_decode += util
                count_decode += 1

        avg_prefill_overall = total_prefill / count_prefill if count_prefill else 0
        avg_decode_overall = total_decode / count_decode if count_decode else 0

        wandb.log({
            "average_prefill_util": avg_prefill_overall,
            "average_decode_util": avg_decode_overall
        })

        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or name of the model.")
    parser.add_argument("--prompt-file", required=True, help="JSON file with list of prompts.")
    parser.add_argument("--output-dir", default=".", help="Output dir for the images.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Tokens for decode.")
    parser.add_argument("--use-wandb", action='store_true', help="Log GPU usage to wandb?")
    parser.add_argument("--wandb-project", default="gpu-utilization", help="wandb project name.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional wandb run name.")
    args = parser.parse_args()

    run_compute_utilization_test(
        model_path=args.model,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
