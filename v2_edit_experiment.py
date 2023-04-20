from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import torch
import json
import os


def get_formatted_dataset(set_name, sample_size=None):
    hf_paths = {
        "toxigen": "skg/toxigen-data",
        "disaster_tweets": "venetis/disaster_tweets",
        "amazon_polarity": "amazon_polarity",
        "imdb": "imdb"
    }
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
        "imdb": ("text", "label"),
        "sst2": ("sentence", "label"),
    }
    hf_subset_name_mappings = {
        "toxigen": "train",
        "disaster_tweets": "train",
    }

    hf_dataset = None
    if set_name in hf_subset_name_mappings:
        hf_dataset = load_dataset(hf_paths[set_name], hf_subset_name_mappings[set_name], use_auth_token=True)["train"].to_pandas()
    else:
        hf_dataset = load_dataset(hf_paths[set_name])["train"].to_pandas()

    formatted_dataset = pd.DataFrame()
    formatted_dataset["prompt"] = hf_dataset[hf_sets_columns_mappings[set_name][0]]
    formatted_dataset["label"] = hf_dataset[hf_sets_columns_mappings[set_name][1]]

    if sample_size:
        half_count = int(sample_size / 2)
        positives = formatted_dataset[formatted_dataset["label"] == 1].sample(half_count)
        negatives = formatted_dataset[formatted_dataset["label"] == 0].sample(sample_size - half_count)
        formatted_dataset = pd.concat([positives, negatives]).sample(frac=1).reset_index(drop=True)

    return formatted_dataset


def format_judment(judgment_string):
    if judgment_string == "0":
        return 0
    elif judgment_string == "1":
        return 1
    else:
        return -1







def evaluate_performance(experiment_id, model, tokenizer, dataset_name, dataset, prompt_strategy, num_shots=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model.name_or_path.replace("/", "-")
    zero_token = tokenizer.encode("0")[0]
    one_token = tokenizer.encode("1")[0]
    judgments = []
    labels = []
    probs = {
        0: [],
        1: []
    }

    with torch.no_grad():
        progress_description = f"Evaluating {model_name} on {dataset_name} {prompt_strategy} with {num_shots} shots"
        for index in tqdm(range(len(dataset)), desc=progress_description):
            row = dataset.iloc[index]
            example = row["prompt"]

            edit_label = None
            if prompt_strategy == "flip":
                if "judgment" in dataset.columns:
                    edit_label = 1 if row["judgment"] == 0 else 0
                else:
                    edit_label = 1 if row["label"] == 0 else 0

            prompt = get_complete_prompt(example, dataset_name, prompt_strategy, num_shots, edit_label)
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                tokenized_prompt,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id)

            jdugment = format_judment(tokenizer.decode(outputs["sequences"][0][-1]))
            positive_prob = outputs["scores"][0][-1][one_token].detach().item()
            negative_prob = outputs["scores"][0][-1][zero_token].detach().item()

            judgments.append(jdugment)
            correct_label = row["label"]
            labels.append(correct_label)
            probs[0].append(negative_prob)
            probs[1].append(positive_prob)

    results_frame = pd.DataFrame(
        {
            "prompt_Strategy": prompt_strategy,
            "judgment": judgments,
            "label": labels,
            "neg_prob": probs[0],
            "pos_prob": probs[1],
            "prompt": dataset["prompt"],
        }
    )

    if not os.path.exists(f"results/{experiment_id}"):
        os.mkdir(f"results/{experiment_id}")
    results_frame.to_csv(f"results/{experiment_id}/{dataset_name}_{model_name}_{prompt_strategy}_{num_shots}.csv", index=False)

    report_string = classification_report(results_frame["label"], results_frame["judgment"])
    print(report_string)

    return results_frame



def evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model):
    # 1-shot with default prompt
    results_frame = evaluate_performance(experiment_id, model, tokenizer, dataset_name, dataset, "default")

    # Set the current input sequence as the prompt example
    mistakes = results_frame[results_frame["label"] != results_frame["judgment"]]
    evaluate_performance(experiment_id, model, tokenizer, dataset_name, mistakes, "flip")

    # Flip labels for incorrect predictions with multiple duplicate examples
    evaluate_performance(experiment_id, model, tokenizer, dataset_name, mistakes, "flip", 3)

    # Flip labels for all predictions
    # evaluate_performance(experiment_id, model, tokenizer, dataset_name, dataset, "flip")

    # Rerun baseline with semanticly similiar prompts based off previous mistakes


def main():
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # dataset_names = ["imdb", "toxigen", "disaster_tweets", "amazon_polarity"]
    # model_names = [
    #     "cerebras/Cerebras-GPT-2.7B",
    #     "cerebras/Cerebras-GPT-6.7B",
    #     "EleutherAI/gpt-j-6b",
    #     "facebook/opt-6.7b",
    #     "databricks/dolly-v2-6-9b",
    #     "EleutherAI/gpt-neox-20b",
    # ]

    sample_size = 10000
    dataset_names = ["sst2"]
    model_names = ["EleutherAI/gpt-j-6b"]
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        for dataset_name in dataset_names:
            dataset = get_formatted_dataset(dataset_name, sample_size)
            print(f"Evaluating {dataset_name} with {model_name}...")
            evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model)


if __name__ == "__main__":
    main()
