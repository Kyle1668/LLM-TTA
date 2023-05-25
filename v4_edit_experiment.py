from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, pipeline
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


def get_judgment(model, tokenizer, template, device, exemplars, input_entry, dataset_name):
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

    if model.config.architectures[0].endswith("ForSequenceClassification"):
        with torch.no_grad():
            input_sequence = tokenizer(input_entry["text"], return_tensors="pt", truncation=True).to(device)
            outputs = model(**input_sequence)
            return int(outputs.logits.argmax(axis=1))

    is_qa_task = dataset_name.startswith("squad")
    prompts = generate_prompt(model.name_or_path, template, exemplars, input_entry["text"], dataset_name)
    tokenized_prompt = tokenizer.encode(prompts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            tokenized_prompt,
            max_new_tokens=50 if is_qa_task else 3,
            length_penalty=0,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id)

        generation = ""
        for score_tuple in outputs["scores"]:
            token_index = score_tuple.argmax(dim=1)
            token = tokenizer.decode(token_index)
            if token == "\n":
                break
            generation += token

    try:
        return generation if is_qa_task else int(generation)
    except:
        print(f"Error: {generation} - unable to convert to int")
        return -1


def evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, edit_retriever=None, embedding_model=None):
    template = get_prompt_template(dataset_name)
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name)
    edit_retriever = get_retriever("kne", data_reader, dataset_name) if edit_retriever is None else edit_retriever
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device) if embedding_model is None else embedding_model
    original_judgments = []
    num_successfull_edits = 0
    prev_edit_accuracies = []
    inference_logs = []
    is_adaptive_set = eval_set.endswith("+adaptive")
    adaptive_tokenizer = None
    adaptive_model = None
    if is_adaptive_set:
        adaptive_tokenizer = LlamaTokenizer.from_pretrained("TheBloke/vicuna-13B-1.1-HF")
        adaptive_model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-13B-1.1-HF", device_map="auto", torch_dtype=torch.float16).eval()

    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")], desc=f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"):
        exemplars = get_exemplars(entry["text"], dataset_name, exemplar_retriever)
        if is_adaptive_set:
            entry["original_text"] = entry["text"]
            entry["text"] = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars)

        judgment = get_judgment(model, tokenizer, template, device, exemplars, entry, dataset_name)
        original_judgments.append(judgment)
        if judgment == -1:
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = {}
        inference_log["input"] = entry["text"]
        if is_adaptive_set:
            inference_log["original_input"] = entry["original_text"]
        if dataset_name.startswith("squad"):
            inference_log["question"] = entry["question"]
        inference_log["judgment"] = judgment
        inference_log["label"] = entry["label"]
        inference_logs.append(inference_log)

        # Perform an edit if the model made a mistake. Add the current input text along with the
        # correct label to the edit dataset. Also encode the input text and add it to the edit
        # retriever's index. Lastly, evaluate whether adding the current input to the prompt
        # along with other edits results in a correct judgment.
        # if eval_set == "prod" and judgment != input_label and False:
        #     if "edits" in dataset:
        #         dataset["edits"].append(entry)
        #     else:
        #         dataset["edits"] = [entry]

        #     input_sequence_embedding = embedding_model.encode([input_text], convert_to_numpy=True)
        #     edit_retriever.add_with_ids(input_sequence_embedding, np.array([len(dataset["edits"]) - 1]))
        #     edit_exemplars = get_edit_exemplars(dataset, edit_retriever, input_sequence_embedding, exemplar_count, exemplars)
        #     edit_judgment = get_judgment(model, tokenizer, template, device, edit_exemplars, input_text)
        #     if edit_judgment == input_label:
        #         num_successfull_edits += 1

        #     # Record accuracy on the holdout test set
        #     holdout_set_perf = evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "test", edit_retriever, embedding_model)
        #     holdout_accuracy = holdout_set_perf["accuracy"]
        #     prev_edit_accuracies.append(holdout_accuracy)

        #     # TODO: Evaluate accuracy on all previous edits and the holdour set.
        #     if len(dataset["edits"]) > 0:
        #         prev_edits_perf = evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "edits", edit_retriever, embedding_model)
        #         holdout_accuracy = prev_edits_perf["accuracy"]
        #         prev_edit_accuracies.append(holdout_accuracy)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")
    pd.DataFrame(inference_logs).to_csv(f"results/{experiment_id}/{model_name.replace('/', '-')}-{dataset_name}-{eval_set}-{icl_method}-inference-logs.csv")
    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, num_successfull_edits)


def get_transferred_input(adaptive_tokenizer, adaptive_model, input_entry, exemplars):
    input_token_len = len(adaptive_tokenizer.encode(input_entry["text"]))
    style_transfer_exemplars = "".join([f'"{exemplar["text"][:input_token_len * 5].strip()}"\n' for exemplar in exemplars])
    style_input = input_entry["text"].replace("\n", " ")
    task_prompt = f"""Paraphrase the input text into the exact writing style of the following examples while keeping the same semantic meaning. Keep all facts and information.
Examples:
{style_transfer_exemplars}
Input Text: "{style_input}\""""
    input_prompts = f"User: {task_prompt}\nAssistant:"
    tokenized_prompt = adaptive_tokenizer.encode(input_prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = adaptive_model.generate(
            tokenized_prompt,
            max_new_tokens=300,
            length_penalty=0,
            early_stopping=True,
            do_sample=True,
            temperature=0.2,
            # output_scores=True,
            return_dict_in_generate=True,
            # pad_token_id=adaptive_tokenizer.eos_token_id,
        )

    generation = adaptive_tokenizer.decode(outputs["sequences"][0]).split("\nAssistant:")[1].replace("\n", " ").replace("</s>", "").strip()
    if "###" in generation:
        generation = generation.split("###")[0]
    if "</s>" in generation:
        generation = generation.split("</s>")[0]
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]

    print(f"Generation: {generation}")
    input_text = generation
    return input_text


def get_model_objects(model_name):
    is_qa_model = AutoConfig.from_pretrained(model_name).architectures[0].endswith("ForQuestionAnswering")
    is_llm = AutoConfig.from_pretrained(model_name).architectures[0].endswith("ForCausalLM")
    is_llama_based_model = is_llm and "llama" in model_name or "vicuna" in model_name
    tokenizer = LlamaTokenizer.from_pretrained(model_name) if is_llama_based_model else AutoTokenizer.from_pretrained(model_name)
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_llm:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
    elif is_qa_model:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.float16).eval().to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).eval().to(device)
    return tokenizer, model


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--icl_method", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    experiment_id = f"edit_experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dataset_names = args.dataset.split(",") if args.dataset is not None else [
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
    icl_methods = args.icl_method.split(",") if args.icl_method is not None else [
        "random",
        "topk",
        # "mdl"
    ]
    splits = args.splits.split(",") if args.splits is not None else ["validation", "test", "test+adaptive"]
    model_names = args.model.split(",") if args.model is not None else [
        "csarron/bert-base-uncased-squad-v1",
        "TheBloke/vicuna-13B-1.1-HF",
        "decapoda-research/llama-65b-hf",
        "decapoda-research/llama-30b-hf",
        "decapoda-research/llama-7b-hf",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-410m",
        "tomh/toxigen_roberta"
    ]

    print("--------------------------------------------------")
    print("Running experiment with the following parameters:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset Names: {dataset_names}")
    print(f"Evalaution Splits: {splits}")
    print(f"ICL Methods: {icl_methods}")
    print(f"Model Names: {model_names}")
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
                    reports.append(evaluate_icl_method(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set))
                    all_reports = pd.DataFrame(reports)
                    print(all_reports)
                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)


if __name__ == "__main__":
    main()
