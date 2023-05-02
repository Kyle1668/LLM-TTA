from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from faiss import IndexIDMap, IndexFlatIP
from datasets import load_dataset
from datetime import datetime
from wilds import get_dataset
from openicl import (
    DatasetReader,
    PromptTemplate,
    TopkRetriever,
    MDLRetriever,
    RandomRetriever,
    PPLInferencer
)
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import os


ENABLE_EDITS = False


def get_formatted_dataset(set_name, max_examples=None):
    hf_paths = {
        "sst2": "gpt3mix/sst2",
        "toxigen": "skg/toxigen-data",
        "disaster_tweets": "venetis/disaster_tweets"
    }
    hf_sets_columns_mappings = {
        "toxigen": ("prompt", "prompt_label"),
        "disaster_tweets": ("text", "target"),
        "amazon_polarity": ("content", "label"),
        "imdb": ("text", "label"),
        "sst2": ("sentence", "label"),
        "ag_news": ("text", "label"),
    }

    hf_dataset = None
    hf_path = hf_paths[set_name] if set_name in hf_paths else set_name
    if set_name.startswith("wilds_"):
        hf_dataset = load_wilds_dataset(hf_path)
    elif set_name == "scotus":
        hf_dataset = load_scotus_dataset()
    else:
        hf_dataset = load_dataset(hf_path)

    if "text" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][0], "text")
    if "label" not in hf_dataset["train"][0].keys():
        hf_dataset["train"] = hf_dataset["train"].rename_column(hf_sets_columns_mappings[set_name][1], "label")
        hf_dataset["test"] = hf_dataset["test"].rename_column(hf_sets_columns_mappings[set_name][1], "label")

    # Create a validation set from the same distirbution as the train set - if none already exist
    if "validation" not in hf_dataset.keys():
        train_set = hf_dataset["train"].to_pandas()
        validation_set = train_set.sample(frac=0.2)
        train_set = train_set.drop(validation_set.index)
        hf_dataset["train"] = Dataset.from_pandas(train_set)
        hf_dataset["validation"] = Dataset.from_pandas(validation_set)

    if max_examples is not None:
        for split in ["train", "validation", "test"]:
            if max_examples >= len(hf_dataset[split]):
                print(f"WARNING: max_examples ({max_examples}) is greater than the number of examples in the {split} set ({len(hf_dataset[split])}).")
                continue

            new_frame = None
            split_frame = hf_dataset[split].to_pandas()
            labels = split_frame["label"].unique()
            max_examples_per_label = max_examples // len(labels)
            for label in labels:
                current_label_sample_size = max_examples_per_label if len(split_frame[split_frame["label"] == label]) > max_examples_per_label else len(split_frame[split_frame["label"] == label])
                label_samples = split_frame[split_frame["label"] == label].sample(current_label_sample_size)
                if new_frame is None:
                    new_frame = label_samples
                else:
                    new_frame = pd.concat([new_frame, label_samples])

            new_frame = new_frame.sample(frac=1)
            new_frame = new_frame.drop(columns=["__index_level_0__"]) if "__index_level_0__" in new_frame.columns else new_frame
            hf_dataset[split] = Dataset.from_pandas(new_frame)

    # Split the test set into a production traffic set from which edits will be made, and a holdout set
    if ENABLE_EDITS:
        original_test_set = hf_dataset["test"].to_pandas().drop(columns=["__index_level_0__"])
        edit_set = original_test_set.sample(frac=0.5)
        test_set = original_test_set.drop(edit_set.index)
        hf_dataset["test"] = Dataset.from_pandas(test_set)
        hf_dataset["prod"] = Dataset.from_pandas(edit_set)

    return hf_dataset


def load_scotus_dataset():
    train_set = pd.read_csv("datasets/scotus_train.csv")
    test_set = pd.read_csv("datasets/scotus_test.csv")
    full_dataset = DatasetDict()
    full_dataset["train"] = Dataset.from_pandas(train_set)
    full_dataset["test"] = Dataset.from_pandas(test_set)
    return full_dataset

def load_wilds_dataset(dataset_name):
    if dataset_name == "wilds_civil_comments":
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for text, label, group in dataset.get_subset("train"):
            train_dict["text"].append(text)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for text, label, group in dataset.get_subset("test"):
            test_dict["text"].append(text)
            test_dict["label"].append(label.item())
            test_dict["group"].append(group.tolist())

        full_dataset = DatasetDict()
        full_dataset["train"] = Dataset.from_pandas(pd.DataFrame(train_dict))
        full_dataset["test"] = Dataset.from_pandas(pd.DataFrame(test_dict))
        return full_dataset
    elif dataset_name == "wilds_amazon":
        dataset = get_dataset(dataset="amazon", download=True)
        train_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for content, label, group in dataset.get_subset("train"):
            train_dict["text"].append(content)
            train_dict["label"].append(label.item())
            train_dict["group"].append(group.tolist())

        test_dict = {
            "text": [],
            "label": [],
            "group": []
        }
        for content, label, group in dataset.get_subset("test"):
            test_dict["text"].append(content)
            test_dict["label"].append(label.item())
            test_dict["group"].append(group.tolist())

        full_dataset = DatasetDict()
        full_dataset["train"] = Dataset.from_pandas(pd.DataFrame(train_dict))
        full_dataset["test"] = Dataset.from_pandas(pd.DataFrame(test_dict))
        return full_dataset
    else:
        raise Exception("Invalid WILDS dataset")


def get_retriever(icl_method, data, dataset_name, index_split='train', test_split='test'):
    dataset_ice_nums = {
        "sst2": 16,
        "ag_news": 6,
        "toxigenic": 6,
        "disaster_tweets": 32,
        "wilds_civil_comments": 16,
        "wilds_amazon": 16,
        "scotus": 4
    }
    ice_num = dataset_ice_nums[dataset_name]

    if icl_method == "topk":
        return TopkRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    elif icl_method == "mdl":
        return MDLRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    elif icl_method == "random":
        return RandomRetriever(dataset_reader=data, ice_num=ice_num, index_split=index_split, test_split=test_split)
    elif icl_method == "kne":
        return IndexIDMap(IndexFlatIP(768))
    else:
        raise Exception("Invalid ICL method")


def get_prompt_template(dataset_name):
    tp_dict = None
    if dataset_name == "sst2":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>1</E>"
        }
    elif dataset_name == "ag_news":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>:1</E>",
            2: "</text>:2</E>",
            3: "</text>:3</E>",
        }
    elif dataset_name == "toxigen":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>:1</E>",
        }
    elif dataset_name == "disaster_tweets":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>:1</E>",
        }
    elif dataset_name == "wilds_civil_comments":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>:1</E>",
        }
    elif dataset_name == "wilds_amazon":
        tp_dict = {
            0: "</text>:0</E>",
            1: "</text>:1</E>",
            2: "</text>:2</E>",
            3: "</text>:3</E>",
            4: "</text>:4</E>",
        }
    elif dataset_name == "scotus":
        tp_dict = {
            0: "\n</text> - Catagory=0</E>",
            1: "\n</text> - Catagory=1</E>",
            2: "\n</text> - Catagory=2</E>",
            3: "\n</text> - Catagory=3</E>",
            4: "\n</text> - Catagory=4</E>",
            5: "\n</text> - Catagory=5</E>",
            6: "\n</text> - Catagory=6</E>",
            7: "\n</text> - Catagory=7</E>",
            8: "\n</text> - Catagory=8</E>",
            9: "\n</text> - Catagory=9</E>",
            10: "\n</text> - Catagory=10</E>",
        }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
    return template


def get_judgment(model, tokenizer, template, device, exemplars, input_text, dataset_name):
    try:
        formatted_exemplars = []
        for i in range(len(exemplars)):
            if exemplars[i]["text"] == "" or exemplars[i]["text"] == None:
                continue
            formatted_exemplars.append({
                "label": exemplars[i]["label"],
                "text": (" ".join((exemplars[i]["text"].split()[:50] if len(exemplars[i]["text"].split()) >= 50 else exemplars[i]["text"]))).replace("\n", " ")
            })

        instructions = json.load(open("prompts/instructions.json"))[dataset_name]
        formatted_instructions = f"Task: {instructions}\n"
        prompt_lines = [formatted_instructions] + ["\n" + template.generate_ice_item(entry, entry["label"]).replace("\n", " ") for entry in reversed(formatted_exemplars)]
        formatted_input_text = " ".join(input_text.split()[:200] if len(input_text.split()) >= 200 else input_text)
        prompt_lines.append(formatted_input_text.replace("\n", " ") + " - Catagory=")
        prompts = "\n".join(prompt_lines)
        tokenized_prompt = tokenizer.encode(prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                        tokenized_prompt,
                        max_new_tokens=1,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id)

        return int(tokenizer.decode(outputs.sequences[:, -1]))
    except:
        return -1


def get_edit_exemplars(dataset, edit_retriever, input_sequence_embedding, exemplar_count, exemplars):
    # Get edit pool exemplars - filter out -1 indices
    edit_distances, edit_exemplar_indices = edit_retriever.index.search(input_sequence_embedding, k=exemplar_count)
    edit_exemplar_indices = [int(index) for index in edit_exemplar_indices[0] if index != -1]
    edit_exemplars = [dataset["edits"][index] for index in edit_exemplar_indices]

    # Backfill with exemplars from the original dataset
    if len(edit_exemplars) < exemplar_count:
        exemplar_index = 0
        while exemplar_index < 4:
            edit_exemplars.append(exemplars[exemplar_index])
            exemplar_index += 1

    return edit_exemplars


def evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, edit_retriever=None, embedding_model=None):
    template = get_prompt_template(dataset_name)
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name)
    edit_retriever = get_retriever("kne", data_reader, dataset_name) if edit_retriever is None else edit_retriever
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device) if embedding_model is None else embedding_model
    exemplar_count = 4
    original_judgments = []
    num_successfull_edits = 0
    prev_edit_accuracies = []

    for entry in tqdm(dataset[eval_set], desc=f"Evaluating {dataset_name} with {model_name} using {icl_method}"):
        input_text = entry["text"]
        input_label = entry["label"]

        # TODO: Update retrievers interface to return sequences for a given string
        exemplar_distances = exemplar_indices = None
        exemplar_indices = None
        retriever_response = exemplar_retriever.get_exemplars(input_text, exemplar_count)
        if isinstance(retriever_response, tuple):
            exemplar_distances, exemplar_indices = retriever_response
        else:
            exemplar_indices = retriever_response

        exemplars = [exemplar_retriever.dataset_reader.dataset["train"][int(index)] for index in exemplar_indices[0]]
        judgment = get_judgment(model, tokenizer, template, device, exemplars, input_text, dataset_name)
        original_judgments.append(judgment)
        if judgment == -1:
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {input_text}")

        # Perform an edit if the model made a mistake. Add the current input text along with the
        # correct label to the edit dataset. Also encode the input text and add it to the edit
        # retriever's index. Lastly, evaluate whether adding the current input to the prompt
        # along with other edits results in a correct judgment.
        enable_edits = False
        if eval_set == "prod" and judgment != input_label and enable_edits:
            if "edits" in dataset:
                dataset["edits"].append(entry)
            else:
                dataset["edits"] = [entry]

            input_sequence_embedding = embedding_model.encode([input_text], convert_to_numpy=True)
            edit_retriever.add_with_ids(input_sequence_embedding, np.array([len(dataset["edits"]) - 1]))
            edit_exemplars = get_edit_exemplars(dataset, edit_retriever, input_sequence_embedding, exemplar_count, exemplars)
            edit_judgment = get_judgment(model, tokenizer, template, device, edit_exemplars, input_text)
            if edit_judgment == input_label:
                num_successfull_edits += 1

            # Record accuracy on the holdout test set
            holdout_set_perf = evaluate_icl_method(
                experiment_id, model_name, model, tokenizer, dataset_name, dataset,
                icl_method, "test", edit_retriever, embedding_model)
            holdout_accuracy = holdout_set_perf["accuracy"]
            prev_edit_accuracies.append(holdout_accuracy)

            # TODO: Evaluate accuracy on all previous edits and the holdour set.
            if len(dataset["edits"]) > 0:
                prev_edits_perf = evaluate_icl_method(
                    experiment_id, model_name, model, tokenizer, dataset_name, dataset,
                    icl_method, "edits", edit_retriever, embedding_model)
                holdout_accuracy = prev_edits_perf["accuracy"]
                prev_edit_accuracies.append(holdout_accuracy)


    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, num_successfull_edits)


def generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, num_successfull_edits):
    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    formatted_model_name = model_name.replace("/", "-")
    report_dict = classification_report([entry["label"] for entry in dataset[eval_set]], original_judgments, output_dict=True)
    num_edits = len(dataset["edits"]) if "edits" in dataset else -1
    edit_success_rate = num_successfull_edits / num_edits if num_edits > 0 else -1
    icl_report = {
        "dataset": dataset_name,
        "split": eval_set,
        "dataset size": len(dataset[eval_set]),
        "icl_method": icl_method,
        "model": formatted_model_name,
        "accuracy": report_dict["accuracy"],
        "avg precision": report_dict["macro avg"]["precision"],
        "avg recall": report_dict["macro avg"]["recall"],
        "avg f1": report_dict["macro avg"]["f1-score"],
        "num edits": num_edits,
        "num successfull edits": num_successfull_edits,
        "edit success rate": edit_success_rate
    }
    output_file_name = f"set={dataset_name}_split={eval_set}_method={icl_method}_model={formatted_model_name}"

    if eval_set == "prod":
        json.dump(icl_report, open(f"results/{experiment_id}/{output_file_name}_report.json", "w+"), indent=4)
        print(f"Classification Results: {formatted_model_name} {dataset_name} {icl_method}")
        print(classification_report(data_reader.references, original_judgments))

    return icl_report


def main():
    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = [
        "scotus"
        # "wilds_amazon",
        # "wilds_civil_comments",
        # "ag_news"
    ]
    baseline_icl_methods = [
        "topk",
        "random",
        "mdl"
    ]
    model_names = [
        "decapoda-research/llama-65b-hf",
        "decapoda-research/llama-30b-hf",
        "decapoda-research/llama-7b-hf",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-410m"
    ]
    reports = []

    for model_name in model_names:
        print(f"Loading model {model_name}...")
        tokenizer = LlamaTokenizer.from_pretrained(model_name) if "llama" in model_name else AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
        for dataset_name in dataset_names:
            print(f"Loading dataset {dataset_name}...")
            dataset = get_formatted_dataset(dataset_name, max_examples=None)
            for icl_method in baseline_icl_methods:
                for evaluation_set in ["validation", "test"]:
                    reports.append(evaluate_icl_method(
                        experiment_id,
                        model_name,
                        model,
                        tokenizer,
                        dataset_name,
                        dataset,
                        icl_method,
                        evaluation_set))

                    all_reports = pd.DataFrame(reports)
                    print(all_reports)
                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)


if __name__ == "__main__":
    main()