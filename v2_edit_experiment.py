from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import torch
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
        "imdb": ("text", "label")
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

def get_complete_prompt(input_sequence, dataset_name, prompt_strategy, num_shots):
    def populate_examples(example_sequence, example_label):
        return [f"\nSequence: {example_sequence}\nLabel:{example_label}" for _ in range(num_shots)]

    if prompt_strategy == "default":
        default_examples = {
            "positive": {
                "toxigen": {
                    "prompt": "I hate it that muslims are now taking over the US",
                    "label": 1
                },
                "disaster_tweets": {
                    "prompt": "There was a suicide bombing in the city last night",
                    "label": 1
                },
                "amazon_polarity": {
                    "prompt": "I love this product experience",
                    "label": 1
                },
                "imdb": {
                    "prompt": """ddd"""
                    ,
                    "label": 1
                }
            },
            "negative": {
                "toxigen": {
                    "prompt": "Black people have been treated unfairly for a long time",
                    "label": 0
                },
                "disaster_tweets": {
                    "prompt": "The sky was ablaze last night in beutiful colors",
                    "label": 0
                },
                "amazon_polarity": {
                    "prompt": "The brand is okay, but this product ruined my childhood",
                    "label": 0
                },
                                "imdb": {
                    "prompt": "all i have to say is if you don't like it then there is something wrong with you. plus Jessica is just all kinds of hot!!!!! the only reason you may not like it is because it is set in the future where Seattle has gone to hell. that and you my not like it cause the future they show could very well happen.",
                    "label": 0
                }
            }
        }

    task_instructions = {
        "toxigen": "Return 1 if the following sequence is hate speech, 0 otherwise. Valid labels are 0 and 1.",
        "disaster_tweets": "Return 1 if the following sequence is a disaster tweet, 0 otherwise. Valid labels are 0 and 1.",
        "amazon_polarity": "Return 1 if the following sequence is a positive review, 0 otherwise. Valid labels are 0 and 1.",
        "imdb": "Return 1 if the following sequence is a positive review, 0 otherwise. Valid labels are 0 and 1."
    }
    prompt = f"""Instructions: {task_instructions[dataset_name]}
{''.join(populate_examples(default_examples["positive"][dataset_name]["prompt"], default_examples["positive"][dataset_name]["label"]))}
{''.join(populate_examples(default_examples["negative"][dataset_name]["prompt"], default_examples["negative"][dataset_name]["label"]))}

Sequence: {input_sequence}
Label:"""
    return prompt


def evaluate_performance(experiment_id, model, tokenizer, dataset_name, dataset, prompt_strategy):
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
        for index in tqdm(range(len(dataset)), desc=f"Evaluating {model_name} on {dataset_name} {prompt_strategy}"):
            row = dataset.iloc[index]
            example = row["prompt"]
            prompt = get_complete_prompt(example, dataset_name, prompt_strategy, 1)
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
            labels.append(row["label"])
            probs[0].append(negative_prob)
            probs[1].append(positive_prob)

    results_frame = pd.DataFrame(
        {
            "example": dataset["prompt"],
            "original_label": dataset["label"],
            "prompt_Strategy": prompt_strategy,
            "judgment": judgments,
            "label": labels,
            "neg_prob": probs[0],
            "pos_prob": probs[1]
        }
    )

    if not os.path.exists(f"results/{experiment_id}"):
        os.mkdir(f"results/{experiment_id}")
    results_frame.to_csv(f"results/{experiment_id}/{dataset_name}_{model_name}_{prompt_strategy}.csv", index=False)

    report_dict = classification_report(results_frame["label"], results_frame["judgment"], output_dict=True)
    report_string = classification_report(results_frame["label"], results_frame["judgment"])
    print(report_string)

    return report_dict



def evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model):
    # 1-shot with default prompt
    results_frame = evaluate_performance(experiment_id, model, tokenizer, dataset_name, dataset, "default")

    # Flip labels for incorrect predictions

    # 1-shot with flipped input as prompt

    # Rerun baseline with semanticly similiar prompts based off previous mistakes



if __name__ == "__main__":
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = ["imdb", "toxigen", "disaster_tweets", "amazon_polarity"]
    # model_names = ["gpt2", "gpt2-xl", "EleutherAI/gpt-neox-20b"]
    model_names = [
        # "google/flan-t5-xxl"
        "cerebras/Cerebras-GPT-6.7B",
        "databricks/dolly-v2-12b",
        "EleutherAI/gpt-j-6b",
        # "EleutherAI/gpt-neox-20b",
        # "gpt2",
        # "gpt2-xl"
    ]
    sample_size = 100

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        modle = model.half() if model_name == "EleutherAI/gpt-neox-20b" else model
        for dataset_name in dataset_names:
            dataset = get_formatted_dataset(dataset_name, sample_size)
            print(f"Evaluating {dataset_name} with {model_name}...")
            evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model)
