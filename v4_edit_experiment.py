from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from openicl import DatasetReader
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
import os

from data_util import generate_icl_report, get_formatted_dataset
from icl_util import generate_prompt, get_prompt_template, get_retriever, get_exemplars


def get_judgment(model, tokenizer, prompt, device, input_entry, dataset_name):
    if model.config.architectures[0].endswith("ForQuestionAnswering"):
        with torch.no_grad():
            # input_sequence = tokenizer(input_entry["text"], return_tensors="pt").to(device)
            # outputs = model(**input_sequence)
            # return outputs.logits.argmax(axis=1)
            question = input_entry["question"]
            context = input_entry["text"]
            question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device="cuda:0")
            qa_response = question_answerer(question=question, context=context)
            return qa_response["answer"]

    if type(model).__name__.endswith("ForSequenceClassification"):
        with torch.no_grad():
            input_sequence = tokenizer(input_entry["text"], return_tensors="pt", truncation=True).to(device)
            outputs = model(**input_sequence)
            return int(outputs.logits.argmax(axis=1))

    is_qa_task = dataset_name.startswith("squad")
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            tokenized_prompt, max_new_tokens=50 if is_qa_task else 50, length_penalty=0, early_stopping=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id
        )

        generation = tokenizer.decode(outputs["sequences"][0][len(tokenized_prompt[0]):]).split("\n")[0].replace("</s>", "").strip()
    try:
        # generation = generation.replace("</s>", "") if "vicuna" in model.name_or_path else generation
        # return generation if is_qa_task else int(generation.strip()[0])
        if is_qa_task:
            return generation

        leading_token = generation.strip()[0]
        if leading_token == "0" or leading_token == "1":
            return int(leading_token)

        final_tokens = generation.replace("</s>", "").replace("<s>", "")[-2:]
        if final_tokens[1] == "0" or final_tokens[1] == "1":
            return int(final_tokens[1])
        elif final_tokens[0] == "0" or final_tokens[0] == "1":
            return int(final_tokens[0])

        split_tokens = [word.replace(".", "") for word in generation.split()]
        if split_tokens[0].lower() in ["positive", "negative"]:
            return 0 if split_tokens[0].lower() == "negative" else 1
        if split_tokens[-1].lower() in ["positive", "negative"]:
            return 0 if split_tokens[-1].lower() == "negative" else 1

        return -1
    except:
        print(f"Error: {generation} - unable to convert to int")
        return -1


def should_get_exemplars(model, eval_set_name):
    if model.config.architectures[0].endswith("ForCausalLM"):
        return True
    if eval_set_name.endswith("+adaptive"):
        return True
    return False


def evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, adaptive_model_name=None, num_shots=None):
    should_retrieve_exemplars = should_get_exemplars(model, eval_set)
    icl_method = icl_method if should_retrieve_exemplars else None
    template = get_prompt_template(dataset_name) if should_retrieve_exemplars else None
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name) if should_retrieve_exemplars else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_judgments = []
    num_successfull_edits = 0
    inference_logs = []
    num_failed_generations = 0

    is_adaptive_set = eval_set.endswith("+adaptive")
    adaptive_tokenizer = None
    adaptive_model = None
    if is_adaptive_set:
        adaptive_tokenizer, adaptive_model = get_model_objects(adaptive_model_name)

    description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
    print(f"{description} and {adaptive_model_name} for style transfer" if is_adaptive_set else description)
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")], desc=description):
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            exemplars, mean_exemplar_distance = get_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots) if should_retrieve_exemplars else None
        if is_adaptive_set:
            entry["original_text"] = entry["text"]
            entry["style_prompt"], entry["text"] = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars)

        prompt = generate_prompt(model_name, template, exemplars, entry["text"], dataset_name) if should_retrieve_exemplars else None
        judgment = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = {}
        inference_log["input"] = entry["text"]
        if is_adaptive_set:
            inference_log["original_input"] = entry["original_text"]
            inference_log["style prompt"] = entry["style_prompt"]
            inference_log["mean exemplar distance"] = mean_exemplar_distance
        if dataset_name.startswith("squad"):
            inference_log["question"] = entry["question"]
        inference_log["judgment"] = judgment
        if should_retrieve_exemplars:
            inference_log["prompt"] = prompt

        inference_log["label"] = entry["label"]
        inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_model_name, num_shots)
    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, adaptive_model_name, num_shots, num_failed_generations)


def save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_model_name, num_shots):
    current_logs = pd.DataFrame(inference_logs)
    model_name = model_name.replace("/", "-")
    adaptive_model_name = adaptive_model_name.replace("/", "-") if adaptive_model_name is not None else None
    current_logs.to_csv(f"results/{experiment_id}/{model_name}-{dataset_name}-{eval_set}-{icl_method}-{adaptive_model_name}-{num_shots}-inference-logs.csv", index=False)
    if eval_set != "test+adaptive":
        return

    combined_inference_log_file_name = f"{model_name.replace('/', '-')}-{dataset_name}-combined-inference-logs.csv"
    saved_logs_names = os.listdir(f"results/{experiment_id}")
    combined_inference_log = pd.read_csv(f"results/{experiment_id}/{combined_inference_log_file_name}") if combined_inference_log_file_name in saved_logs_names else None
    if combined_inference_log is None:
        test_baseline_log_file_name = [name for name in os.listdir(f"results/{experiment_id}") if name.startswith(f"{model_name.replace('/', '-')}-{dataset_name}-test-None")][0]
        test_baseline_log = pd.read_csv(f"results/{experiment_id}/{test_baseline_log_file_name}")

        combined_inference_log = pd.DataFrame()
        combined_inference_log["label"] = test_baseline_log["label"]
        combined_inference_log["original input"] = test_baseline_log["input"]
        combined_inference_log["original judgment"] = test_baseline_log["judgment"]

    for saved_log_name in saved_logs_names:
        if not saved_log_name.startswith(f"{model_name.replace('/', '-')}-{dataset_name}-{eval_set}"):
            continue

        prev_log = pd.read_csv(f"results/{experiment_id}/{saved_log_name}")
        column_name_prefix = f"{adaptive_model_name}-{num_shots}"
        combined_inference_log[f"{column_name_prefix} Judgment"] = prev_log["judgment"]
        combined_inference_log[f"{column_name_prefix} Input"] = prev_log["input"]
        combined_inference_log[f"{column_name_prefix} Prompt"] = prev_log["style prompt"]
    combined_inference_log.to_csv(f"results/{experiment_id}/{combined_inference_log_file_name}")



def get_transferred_input(adaptive_tokenizer, adaptive_model, input_entry, exemplars):
    style_input = input_entry["text"].replace("\n", " ")
    style_transfer_exemplars = "".join([f'"{exemplar["text"].strip()[:len(style_input)]}"\n' for exemplar in exemplars])
    task_prompt = f"""Paraphrase the input text into the exact writing style of the following examples while keeping the same semantic meaning. Keep all facts and information.
Examples:
{style_transfer_exemplars}
Now rewriter the current input text into the same style as the examples. Only return the new text.
Input Text: "{style_input}\""""
    input_prompts = f"User: {task_prompt}\nAssistant:" if "vicuna" in adaptive_model.config.name_or_path else task_prompt
    tokenized_prompt = adaptive_tokenizer.encode(input_prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = adaptive_model.generate(
            tokenized_prompt,
            max_new_tokens=300,
            length_penalty=0,
            repetition_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
        )

    generation = adaptive_tokenizer.decode(outputs["sequences"][0][len(tokenized_prompt[0]):]).replace("\n", " ").replace("</s>", "").strip()
    if "###" in generation:
        generation = generation.split("###")[0]
    if " Text:" in generation:
        generation = generation.split(" Text:")[1].strip()
    if "</s>" in generation:
        generation = generation.split("</s>")[0]
    if "<s>" in generation:
        generation = generation.replace("<s>", " ").strip()
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]
    if "<|endoftext|>" in generation:
        generation = generation.split("<|endoftext|>")[0]
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]

    # print(f"Generation: {generation}")
    input_text = generation
    return input_prompts, input_text


def get_model_objects(model_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_qa_model = model_config.architectures[0].endswith("ForQuestionAnswering")
    is_llm = model_config.architectures[0].endswith("ForCausalLM")
    is_llama_based_model = is_llm and "llama" in model_name or "vicuna" in model_name
    tokenizer = LlamaTokenizer.from_pretrained(model_name) if is_llama_based_model else AutoTokenizer.from_pretrained(model_name)
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_llm:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto").eval()
    elif is_qa_model:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval().to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval().to(device)
    return tokenizer, model


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--icl_method", type=str, default=None)
    parser.add_argument("--adaptive_model", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = (
        args.dataset.split(",")
        if args.dataset is not None
        else [
            "squadshifts_reddit",
            "squadshifts_amazon",
            "imdb_rotten_tomatoes",
            "rotten_tomatoes_imdb",
            "civil_toxigen",
            "scotus",
            "ag_news",
            "wilds_amazon",
            "wilds_civil_comments",
        ]
    )
    icl_methods = (
        args.icl_method.split(",")
        if args.icl_method is not None
        else [
            "random",
            "topk",
            "mdl"
        ]
    )
    splits = args.splits.split(",") if args.splits is not None else ["validation", "test", "test+adaptive"]
    adaptive_model_names = (
        args.adaptive_model.split(",")
        if args.adaptive_model is not None
        else [
            "TheBloke/vicuna-13B-1.1-HF",
            "TheBloke/vicuna-7B-1.1-HF",
            # "tiiuae/falcon-7b-instruct",
        ]
    )
    model_names = (
        args.model.split(",")
        if args.model is not None
        else [
            "csarron/bert-base-uncased-squad-v1",
            "TheBloke/vicuna-13B-1.1-HF",
            "decapoda-research/llama-65b-hf",
            "decapoda-research/llama-30b-hf",
            "decapoda-research/llama-7b-hf",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-410m",
            "tomh/toxigen_roberta",
        ]
    )
    # Also evaluate models used for sytle transfer
    model_names =  model_names + adaptive_model_names

    print("--------------------------------------------------")
    print("Running experiment with the following parameters:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset Names: {dataset_names}")
    print(f"Evalaution Splits: {splits}")
    print(f"ICL Methods: {icl_methods}")
    print(f"Task Model Names: {model_names}")
    print(f"Style Model Names: {adaptive_model_names}")
    print(f"Max Examples: {args.max_examples}")
    print("--------------------------------------------------\n")

    reports = []
    for model_name in model_names:
        print(f"Loading model {model_name}...")
        tokenizer, model = get_model_objects(model_name)

        for dataset_name in dataset_names:
            print(f"Loading dataset {dataset_name}...")
            dataset = get_formatted_dataset(dataset_name, max_examples=args.max_examples)

            for icl_method in icl_methods:
                for evaluation_set in splits:
                    # if is llm, incrmeent the shots for every split

                    # if not an llm, only increment the shots for the test+adaptive split

                    if evaluation_set == "test+adaptive":
                        for adaptive_model_name in adaptive_model_names:
                            for num_shots in [8]:
                                reports.append(evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, adaptive_model_name, num_shots))
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports)
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                    else:
                        is_llm = model.config.architectures[0].endswith("ForCausalLM")
                        if is_llm:
                            for num_shots in [8]:
                                reports.append(evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, num_shots=num_shots))
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports)
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                        else:
                            reports.append(evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set))
                            all_reports = pd.DataFrame(reports).drop_duplicates()
                            print(all_reports)
                            all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)


if __name__ == "__main__":
    main()
