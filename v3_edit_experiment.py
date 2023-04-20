from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import os

from data_util import get_formatted_dataset, get_static_examplars, get_complete_prompt, format_judgment


def eval_technique(experiment_id, dataset_name, dataset, tokenizer, model, num_shots, technique, edit_count=None):
    task_instructions = json.load(open("prompts/instructions.json", encoding="utf-8"))[dataset_name]
    exemplars = get_static_examplars(dataset_name, dataset, num_shots, use_default=False)
    input_sequences = pd.concat([dataset,exemplars]).drop_duplicates(keep=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model.name_or_path.replace("/", "-")
    zero_token = tokenizer.encode("0")[0]
    one_token = tokenizer.encode("1")[0]
    generated_tokens = []
    judgments = []
    labels = []
    probs = {
        0: [],
        1: []
    }

    with torch.no_grad():
        progress_description = f"Evaluating {model_name} on {dataset_name} using {technique} with {num_shots} shots"
        for index in tqdm(range(len(input_sequences)), desc=progress_description):
            row = input_sequences.iloc[index]
            example = row["prompt"]
            edit_label = row["label"] if technique == "label_flipping" else None

            prompt = get_complete_prompt(example, task_instructions, exemplars, technique, edit_count, edit_label)
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                tokenized_prompt,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id)

            generation = tokenizer.decode(outputs["sequences"][0][-1])
            judgment = format_judgment(generation)
            positive_prob = outputs["scores"][0][-1][one_token].detach().item()
            negative_prob = outputs["scores"][0][-1][zero_token].detach().item()

            judgments.append(judgment)
            correct_label = row["label"]
            labels.append(correct_label)
            probs[0].append(negative_prob)
            probs[1].append(positive_prob)
            generated_tokens.append(generation)

    results_frame = pd.DataFrame(
        {
            "technique": technique,
            "judgment": judgments,
            "generated_token": generated_tokens,
            "label": labels,
            "neg_prob": probs[0],
            "pos_prob": probs[1],
            "prompt": input_sequences["prompt"],
        }
    )

    save_report(experiment_id, dataset_name, num_shots, model_name, technique, results_frame)
    return results_frame


def save_report(experiment_id, dataset_name, num_shots, model_name, technique, results_frame):
    if not os.path.exists(f"results/{experiment_id}"):
        os.mkdir(f"results/{experiment_id}")

    results_frame.to_csv(f"results/{experiment_id}/{dataset_name}_{model_name}_{technique}_{num_shots}.csv", index=False)

    report_json = classification_report(results_frame["label"], results_frame["judgment"], output_dict=True)
    json.dump(report_json, open(f"results/{experiment_id}/{dataset_name}_{model_name}_{technique}_{num_shots}_report.json", "w"))

    report_string = classification_report(results_frame["label"], results_frame["judgment"])
    print(report_string)


def evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model):
    for shots in [4, 8, 16, 32]:
        # Evaluate baseline perf at varying shot levels and get the models mistakes
        mistakes = eval_technique(experiment_id, dataset_name, dataset, tokenizer, model, shots, "baseline")

        # # Evaluate success of label flipping for the model's by with replacing an increasing number of the cases
        # for edit_ratio in [0.25, 0.5, 0.75, 1.0]:
        #     exemplar_edit_count = int(shots * edit_ratio)
        #     eval_technique(experiment_id, dataset_name, mistakes, tokenizer, model, shots, "label_flipping", exemplar_edit_count)

        # # Iterate through the dataset again, this time time saving mistakes to the edit pool. Whenever a new sequence is encountered
        # # which is close enough in embedding space to a mistake in the edit pool, replace a random exemplar with the edit sequence
        # # TODO: Also evaluate on previous mistakes and some holdout set to compute the edit score
        # for edit_ratio in [0.25, 0.5, 0.75, 1.0]:
        #     exemplar_edit_count = int(shots * edit_ratio)
        #     eval_technique(experiment_id, dataset_name, dataset, tokenizer, model, shots, "kNE", exemplar_edit_count)


def main():
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = ["sst2"]
    model_names = [
        "EleutherAI/gpt-j-6b",
    ]
    sample_size = 250
    np.random.seed(42)

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        for dataset_name in dataset_names:
            dataset = get_formatted_dataset(dataset_name, sample_size)
            print(f"Evaluating {dataset_name} with {model_name}...")
            evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model)


if __name__ == "__main__":
    main()
