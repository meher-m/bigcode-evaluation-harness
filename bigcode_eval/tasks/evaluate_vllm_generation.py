import argparse
import pandas as pd
import requests
import aiobotocore.session
import s3fs
import json
import os
import random
import time

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from requests.auth import HTTPBasicAuth


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate generations made using VLLM')
    parser.add_argument('--generation_dir', type=str, help='Directory to save generations to. Will be prepended by ~/bigcode-evaluation-harness/bigcode_eval/tasks/')
    parser.add_argument('--model_path', type=str, default="~/bigcode-evaluation-harness/Meta-Llama-3-8B/", help='Local path of model to use for evaluation')
    parser.add_argument('--result_file', type=str, default="e2e_all_turns_results_llama_3.json", help='Name of the file to save the results to')
    return parser.parse_args()


def main():

    args = parse_arguments()
    
    results = {}
    home_directory = os.path.expanduser('~')

    # Number of tasks in the OTS dataset I pulled from public.otstasks
    for i in range(323):
        print(f"Starting on element {i}")
        
        # Open the json file and reformat it
        with open(f"{home_directory}/bigcode-evaluation-harness/bigcode_eval/tasks/{args.generation_dir}/vllm_generation_results_{i}.json", "r") as f:
            data = json.load(f)

        new_data = []
        for task in data:
            new_data.append([task])

        with open(f"{home_directory}/bigcode-evaluation-harness/bigcode_eval/tasks/{args.generation_dir}/vllm_generation_results_{i}_copy.json", "w") as f:
            json.dump(new_data, f)

        # Run nuggets in generation only mode and write metrics to a file 
        cmd = f"cd ../../ && accelerate launch --num_processes 2 main.py \
            --model {args.model_path} \
            --max_length_generation 2048 \
            --tasks humaneval \
            --precision bf16 \
            --do_sample False \
            --temperature 0.0 \
            --allow_code_execution \
            --metric_output_path llama_2/llama_2_metrics/public_ots_{i}.json \
            --load_generations_path bigcode_eval/tasks/{args.generation_dir}/vllm_generation_results_{i}_copy.json"
        
        # Read metrics from that file
        os.system(cmd)

        # Read the result from llama_3_metrics/public_ots_{i}.json and save to results dict
        with open(f"{home_directory}/bigcode-evaluation-harness/llama_2/llama_2_metrics/public_ots_{i}.json", "r") as f:
            res = json.load(f)
            try:
                results[i] = res["humaneval"]["pass@1"]
            except:
                import pdb; pdb.set_trace()
                pass

        # Delete metrics file
        os.remove(f"{home_directory}/bigcode-evaluation-harness/llama_2/llama_2_metrics/public_ots_{i}.json")
        os.remove(f"{home_directory}/bigcode-evaluation-harness/bigcode_eval/tasks/{args.generation_dir}/vllm_generation_results_{i}_copy.json")

        with open(args.result_file, "w") as f:
            json.dump(results, f)

    # Save results dict locally
    with open(args.result_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
