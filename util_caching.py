from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
import hashlib
import time
import ast
import os

from util_modeling import is_language_model


cache_frame = {}

def distributed_cache_write(rank, world_size, model_name, dataset_name, icl_method, eval_set, temperature, inference_logs, adaptive_model, entry):
    distributed_rewrites_cache = None
    cache_write_steps = 1
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            distributed_rewrites_cache = [[] for i in range(world_size)]

        if is_cache_write_step := len(inference_logs) % cache_write_steps == 0:
            print(f"Gathering cached rewrites across ranks")
            dist.gather_object(entry, distributed_rewrites_cache)

        if rank == 0 and is_cache_write_step:
            writable_entries = [write_entry for write_entry in distributed_rewrites_cache if not write_entry["rewrite_cache_hit"]]
            if len(writable_entries) == 0:
                print("Skipping cache writes because all entries were cache hits")
                return

            description = f"Writing {len(writable_entries)} rewrites for {dataset_name}-{eval_set} with {model_name} using {icl_method}"
            print(description)
            cache_style_prompts = [write_entry["style_prompt"] for write_entry in writable_entries]
            cache_texts = [write_entry["text"] for write_entry in writable_entries]
            write_cached_rewrites(dataset_name, adaptive_model, temperature, cache_style_prompts, cache_texts)
    else:
        if entry["rewrite_cache_hit"] is True:
            print(f"Skipping cache write because entry was a cache hit")
        else:
            write_cached_rewrites(dataset_name, adaptive_model, temperature, entry["style_prompt"], entry["text"])


def flush_local_cache():
    global cache_frame
    cache_frame = {}


def get_cached_rewrites(dataset_name, rewrite_model, temperature, input_prompt):
    global cache_frame

    # set stopwatch for cache read
    start_time = time.perf_counter()

    try:

        if not os.path.exists("cached_rewrites"):
            os.mkdir("cached_rewrites")

        cache_path = f"cached_rewrites/{dataset_name}_{rewrite_model.name_or_path.replace('/', '_')}.csv"
        if is_language_model(rewrite_model.name_or_path):
            cache_path = cache_path.replace(".csv", f"_temp={temperature}.csv")

        if os.path.exists(cache_path) and cache_path not in cache_frame:
            cache_frame[cache_path] = pd.read_csv(cache_path, on_bad_lines="warn", engine="python")

        if cache_path in cache_frame and cache_frame[cache_path] is not None:
            hashed_prompt = hashlib.sha256(input_prompt.encode()).hexdigest()
            read_frame_start = time.perf_counter()
            cached_inference = cache_frame[cache_path][cache_frame[cache_path]["prompt_hash"] == hashed_prompt]
            end_time = time.perf_counter()
            if len(cached_inference) > 0:
                print(f"Found cached rewrites for {rewrite_model.name_or_path}. Overall Latency = {round(end_time - start_time, 2)} seconds & Search Latency = {round(end_time - read_frame_start, 2)} seconds")
                return ast.literal_eval(cached_inference.iloc[0]["rewrites"])
    except Exception as e:
        end_time = time.perf_counter()
        print(f"Error reading cached rewrites with Latency = {round(end_time - start_time, 2)}: {e}")

    return None


def write_cached_rewrites(dataset_name, rewrite_model, temperature, input_prompt, rewrites):
    try:
        cache_path = f"cached_rewrites/{dataset_name}_{rewrite_model.name_or_path.replace('/', '_')}.csv"
        if is_language_model(rewrite_model.name_or_path):
            cache_path = cache_path.replace(".csv", f"_temp={temperature}.csv")

        input_prompt = input_prompt if isinstance(input_prompt, list) else [input_prompt]
        cache_miss_frame = pd.DataFrame({
                    "prompt_hash": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in input_prompt],
                    "prompt": input_prompt,
                    "rewrites": rewrites if isinstance(rewrites[0], list) else [rewrites],
        })

        cache_frame = pd.read_csv(cache_path, on_bad_lines="warn", engine="python") if os.path.exists(cache_path) else None
        if cache_frame is not None and cache_miss_frame["prompt_hash"].isin(cache_frame["prompt_hash"]).any():
            print(f"Skipping cache write because prompt already exists in cache")
            return

        updated_cache_frame = cache_miss_frame if cache_frame is None else pd.concat([cache_frame, cache_miss_frame])
        updated_cache_frame.to_csv(cache_path, index=False)
    except Exception as e:
        print(f"Error writing cached rewrites: {e}")