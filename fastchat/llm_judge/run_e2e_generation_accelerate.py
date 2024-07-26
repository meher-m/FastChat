import os
import json
import argparse
import time
import pandas as pd
import requests
import aiobotocore.session
import s3fs
import random

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from requests.auth import HTTPBasicAuth


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate code using VLLM')
    # parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    # parser.add_argument("--tmp_generation_dir", type=str, default="vllm_few_shot_examples", help="Directory to save temporary generations to")
    # parser.add_argument('--s3_dataset_path', type=str, help='Path to read dataset from S3. Should end in /dataset/')
    # parser.add_argument('--generation_dir', type=str, help='Directory to save generations to. Will be prepended by ~/bigcode-evaluation-harness/bigcode_eval/tasks/')
    parser.add_argument('--model', type=str, default="codellama/CodeLlama-7b-Instruct-hf", help='Name of the hf model to be used for generation')
    parser.add_argument('--model_id_prefix', type=str, default="code-llama-7b-instruct", help='Name you want to save the model as. ')
    parser.add_argument('--openai_key', type=str, default=None)
    # parser.add_argument('--use_chat_template', action="store_true", help='Use the chat template for generation')
    return parser.parse_args()


def get_s3fs():
    s3_session = aiobotocore.session.AioSession(profile="ml-worker")
    storage_options = {"session": s3_session}
    fs = s3fs.S3FileSystem(**storage_options)
    return fs

def save_ds_s3(ds, path: str):
    fs = get_s3fs()
    ds.save_to_disk(path, storage_options=fs.storage_options)


def load_ds_s3(path: str):
    fs = get_s3fs()
    dataset = load_from_disk(path, storage_options=fs.storage_options)
    return dataset


def main():
    
    full_start_time = time.time()
    args = parse_arguments()

    # Adding code

    # Get the example from s3 dataset... loop over examples 
    s3_dataset = load_from_disk("/tmp/mt-bench-dataset/")
    train_ds = s3_dataset["train"]
    messages = train_ds["messages"]

    # zeroshot_baseline_cmd = f'python gen_model_answer.py --model-path {args.model} --model-id {} --openai_key {args.openai_key}'
    # print(f"The command is: {zeroshot_baseline_cmd}")
    # os.system(zeroshot_baseline_cmd)
    start_time = time.time()
    for idx, example in enumerate(messages[:5]):

        # Write example to a temporary file
        tmp_file = f"/tmp/example_{idx}.json"
        with open(tmp_file, 'w') as f:
            json.dump(example, f)

        model_id = f"{args.model_id_prefix}_{idx}_accelerate"

        cmd = f'python gen_model_answer.py --model-path {args.model} --model-id {model_id} --one_shot_example {tmp_file}'

        print(f"The command is: {cmd}")

        os.system(cmd)

        # Delete the temporary file
        os.remove(tmp_file)

    # return

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    main()


