import os
import torch
import faiss
import argparse
import pandas as pd
import torch.nn.functional as F
import nlpaug.augmenter.word as naw
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
tqdm.pandas()

from util_modeling import get_model_objects
from util_data import get_formatted_dataset, get_num_labels
from adaptive_methods import get_paraphrase_augmentations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--centroid_size", type=int, default=4)
    args = parser.parse_args()

    formatted_dataset = get_formatted_dataset(args.dataset, args.max_examples)["train"].to_pandas()
    formatted_dataset.rename(columns={"label": "class","text": "label"}, inplace=True)




if __name__ == "__main__":
    main()