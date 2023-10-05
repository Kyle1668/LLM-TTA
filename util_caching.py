from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
import hashlib
import ast
import os

from util_modeling import is_language_model


def distributed_cache_write(rank, world_size, model_name, dataset_name, icl_method, eval_set, temperature, inference_logs, adaptive_model, entry):
    distributed_rewrites_cache = None
    cache_write_steps = 1
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            distributed_rewrites_cache = [[] for i in range(world_size)]

        if is_cache_write_step := len(inference_logs) % cache_write_steps == 0:
            print(f"Gathering cached rewrites across ranks")
            dist.gather_object([entry["style_prompt"], entry["text"]], distributed_rewrites_cache)

        if rank == 0 and is_cache_write_step:
            description = f"Writing cached rewrites for {dataset_name}-{eval_set} with {model_name} using {icl_method}"
            for style_prompt, rewrites in distributed_rewrites_cache:
                write_cached_rewrites(adaptive_model, temperature, style_prompt, rewrites)
    else:
        write_cached_rewrites(adaptive_model, temperature, entry["style_prompt"], entry["text"])


def get_cached_rewrites(rewrite_model, temperature, input_prompt):
    try:
        cache_path = f"cached_rewrites/{rewrite_model.name_or_path.replace('/', '_')}.csv"
        if is_language_model(rewrite_model.name_or_path):
            cache_path = cache_path.replace(".csv", f"_temp={temperature}.csv")

        if os.path.exists(cache_path):
            cache_frame = pd.read_csv(cache_path)
            hashed_prompt = hashlib.sha256(input_prompt.encode()).hexdigest()
            cached_inference = cache_frame[cache_frame["prompt_hash"] == hashed_prompt]
            if len(cached_inference) > 0:
                print(f"Found cached rewrites for {rewrite_model.name_or_path}")
                return ast.literal_eval(cached_inference.iloc[0]["rewrites"])
    except Exception as e:
        print(f"Error reading cached rewrites: {e}")

    return None


def write_cached_rewrites(rewrite_model, temperature, input_prompt, rewrites):
    try:
        cache_path = f"cached_rewrites/{rewrite_model.name_or_path.replace('/', '_')}.csv"
        if is_language_model(rewrite_model.name_or_path):
            cache_path = cache_path.replace(".csv", f"_temp={temperature}.csv")

        hashed_prompt = hashlib.sha256(input_prompt.encode()).hexdigest()
        cache_miss_frame = pd.DataFrame({
                    "prompt_hash": [hashed_prompt],
                    "prompt": [input_prompt],
                    "rewrites": [rewrites],
        })

        cache_frame = pd.read_csv(cache_path) if os.path.exists(cache_path) else None
        if cache_frame is not None and cache_miss_frame["prompt_hash"].isin(cache_frame["prompt_hash"]).any():
            print(f"Skipping cache write because prompt already exists in cache")
            return

        updated_cache_frame = cache_miss_frame if cache_frame is None else pd.concat([cache_frame, cache_miss_frame])
        updated_cache_frame.to_csv(cache_path, index=False)
    except Exception as e:
        print(f"Error writing cached rewrites: {e}")