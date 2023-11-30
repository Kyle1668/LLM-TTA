from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import pandas as pd
import json
import wandb
import os
import hashlib
tqdm.pandas()

if __name__ == "__main__":
    print("Pulling cache from HF")
    hf_cache = load_dataset("Kyle1668/LLM-TTA-Cached-Rewrites")

    if os.path.exists("cached_rewrites"):
        print("Removing old cache")
        os.system("rm -rf cached_rewrites")
    
    os.mkdir("cached_rewrites")
    
    for split_name in hf_cache:
        local_file_name = split_name.replace("dot", ".").replace("equals", "=")

        if "back_translate" in local_file_name:
            local_file_name = local_file_name.replace("back_translate", "back-translate")
        if "StableBeluga_" in local_file_name:
            local_file_name = local_file_name.replace("StableBeluga_", "StableBeluga-")

        print(f"Writing {local_file_name} to disk")
        local_file_name += ".csv"
        hf_cache[split_name].to_csv(f"cached_rewrites/{local_file_name}", index=False)

        