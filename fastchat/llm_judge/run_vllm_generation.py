import argparse
import os
import time
from multiprocessing import Process


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate code using VLLM')
    parser.add_argument('--openai_key', type=str, default=None)
    return parser.parse_args()


def run_command(command):
    os.system(command)


def main():

    args = parse_arguments()

    processes=[]
    p1 = Process(target=run_command, args=("python3 -m fastchat.serve.controller",))
    processes.append(p1)
    p1.start()

    p2 = Process(target=run_command, args=("python3 -m fastchat.serve.vllm_worker --model-path meta-llama/Meta-Llama-3-8B-Instruct",))
    processes.append(p2)
    p2.start()

    p3 = Process(target=run_command, args=("python3 -m fastchat.serve.openai_api_server --host localhost --port 8000",))
    processes.append(p3)
    p3.start()

    time.sleep(120)

    client_process = Process(target=run_command, args=(f"python run_e2e_generation.py --openai_key {args.openai_key} --model Meta-Llama-3-8B-Instruct",))
    client_process.start()

    client_process.join()


if __name__=="__main__":
    main()