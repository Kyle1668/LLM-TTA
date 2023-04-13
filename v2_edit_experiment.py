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

def get_complete_prompt(input_sequence, dataset_name, prompt_strategy, num_shots, edit_label=None):
    def populate_examples(example_sequence, example_label):
        return [f"\nSequence: {example_sequence}\nLabel:{example_label}" for _ in range(num_shots)]


    default_examples = {
            "positive": {
                "toxigen": {
                    "prompt": "- the vast majority of african-americans are poor because they are stupid and they spend all their money on drugs and other unproductive things - naturally less productive than their caucasian counterparts, african-americans have become vastly important as menial workers - african-americans shouldnt be allowed to vote or drive because theyre too stupid to",
                    "label": 1
                },
                "disaster_tweets": {
                    "prompt": "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected",
                    "label": 1
                },
                "amazon_polarity": {
                    "prompt": "This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^",
                    "label": 1
                },
                "imdb": {
                    "prompt": "all i have to say is if you don't like it then there is something wrong with you. plus Jessica is just all kinds of hot!!!!! the only reason you may not like it is because it is set in the future where Seattle has gone to hell. that and you my not like it cause the future they show could very well happen.",
                    "label": 1
                }
            },
            "negative": {
                "toxigen": {
                    "prompt": "sometimes hispanic folks commit crimes but that is not specific to their culture\ - if you really think about it, banning latinos from the USA is an example of old-fashioned racism\ - people who have hispanic heritage are often blamed for many problems in the USA\ - many aspects of latino culture are really rich and should be integrated into american education\ - there is lots of diverse heritage in the latino world and that is largely ignored by americans\\n-",
                    "label": 0
                },
                "disaster_tweets": {
                    "prompt": "@PhDSquares #mufc they've built so much hype around new acquisitions but I doubt they will set the EPL ablaze this season.",
                    "label": 0
                },
                "amazon_polarity": {
                    "prompt": "This is a self-published book, and if you want to know why--read a few paragraphs! Those 5 star reviews must have been written by Ms. Haddon's family and friends--or perhaps, by herself! I can't imagine anyone reading the whole thing--I spent an evening with the book and a friend and we were in hysterics reading bits and pieces of it to one another. It is most definitely bad enough to be entered into some kind of a worst book contest. I can't believe Amazon even sells this kind of thing. Maybe I can offer them my 8th grade term paper on To Kill a Mockingbird--a book I am quite sure Ms. Haddon never heard of. Anyway, unless you are in a mood to send a book to someone as a joke---stay far, far away from this one!",
                    "label": 0
                },
                "imdb": {
                    "prompt": "I sat through this turkey because I hadn't seen it before, and because the premise sounded like it had potential. It was mildly entertaining until the hurricane sequence. At the height of the storm, the wind is strong enough to blow windows out of the house, yet the trees in the background are perfectly upright and not a leaf is moving! In fact, when the characters move outside the house, bright sunlight is visible illuminating the treetops. At that point, whatever credence the filmmakers had developed evaporated faster than the highly localized rain in their film. Too bad all hurricanes aren't like this one, it would surely help our homeowners insurance rates here in the Sunshine State.",
                    "label": 0
                }
            }
        }

    pos_example = default_examples["positive"][dataset_name]["prompt"]
    neg_example = default_examples["negative"][dataset_name]["prompt"]
    if prompt_strategy == "flip":
        assert edit_label is not None
        if edit_label == 1:
            pos_example = input_sequence
        else:
            neg_example = input_sequence

    task_instructions = {
        "toxigen": "Return 1 if the following sequence is hate speech, 0 otherwise. Valid labels are 0 and 1.",
        "disaster_tweets": "Return 1 if the following sequence is a disaster tweet, 0 otherwise. Valid labels are 0 and 1.",
        "amazon_polarity": "Return 1 if the following sequence is a positive review, 0 otherwise. Valid labels are 0 and 1.",
        "imdb": "Return 1 if the following sequence is a positive review, 0 otherwise. Valid labels are 0 and 1."
    }
    formatted_input_sequence = input_sequence.replace("\n", "")
    prompt = f"""Instructions: {task_instructions[dataset_name]}
{''.join(populate_examples(pos_example, 1))}
{''.join(populate_examples(neg_example, 0))}

Sequence: {formatted_input_sequence}
Label:"""
    return prompt


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



if __name__ == "__main__":
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # dataset_names = ["imdb", "toxigen", "disaster_tweets", "amazon_polarity"]
    dataset_names = ["toxigen", "disaster_tweets"]
    model_names = [
        "cerebras/Cerebras-GPT-2.7B",
        "cerebras/Cerebras-GPT-6.7B",
        "EleutherAI/gpt-j-6b",
        "facebook/opt-6.7b",
        "databricks/dolly-v2-6-9b",
        # "EleutherAI/gpt-neox-20b",
    ]
    sample_size = 10000

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        for dataset_name in dataset_names:
            dataset = get_formatted_dataset(dataset_name, sample_size)
            print(f"Evaluating {dataset_name} with {model_name}...")
            evaluate_editing(experiment_id, dataset_name, dataset, tokenizer, model)
