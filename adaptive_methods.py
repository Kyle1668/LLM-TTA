from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline
from openicl import DatasetReader
from torch.optim import AdamW
from itertools import chain
from umap import UMAP
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import plotly.express as px
import pandas as pd
import numpy as np
import hashlib
import torch
import time
import ast
import os
import re
tqdm.pandas()

from util_data import generate_evaluation_Report, get_num_labels
from util_modeling import get_model_objects, is_large_language_model, is_language_model, is_openai_model
from util_icl import generate_prompt, get_prompt_template, get_retriever, get_static_exemplars, get_dynamic_exemplars

# Distributed inference
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# TODO: Return logits for LLMs and QA
def get_judgment(model, tokenizer, prompt, device, input_entry, dataset_name):
    if input_entry["text"] == None:
        return -1, None

    if model.config.architectures[0].endswith("ForQuestionAnswering"):
        with torch.no_grad():
            question = input_entry["question"]
            context = input_entry["text"]
            question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device="cuda:0")
            qa_response = question_answerer(question=question, context=context)
            return qa_response["answer"]

    if type(model).__name__.endswith("ForSequenceClassification"):
        is_nli_task = dataset_name.startswith("boss_nli")
        with torch.no_grad():
            input_texts = input_entry["text"] + [input_entry["original_text"]] if "original_text" in input_entry else input_entry["text"]
            input_sequence = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True).to(model.device)
            outputs = model(**input_sequence)
            logits = outputs.logits
            mean_logits = logits.mean(dim=0)
            predicted_class = int(mean_logits.argmax())

            if is_nli_task:
                nl_judgment = model.config.id2label[predicted_class].lower()
                token_label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                return token_label_map[nl_judgment], logits

            # Calculate inference metrics
            class_probabilities = torch.softmax(mean_logits, dim=0)
            entropy = -torch.sum(class_probabilities * torch.log(class_probabilities)).detach().item()
            all_probs = torch.softmax(logits, dim=1)
            all_entropies = [-torch.sum(prob_dist * torch.log(prob_dist)).detach().item() for prob_dist in all_probs]
            inference_metadata = {
                "entropy": entropy,
                "mean probs": class_probabilities.detach().cpu().tolist(),
                "all probs": all_probs.detach().cpu().tolist(),
                "all entropies": all_entropies,
            }

            return predicted_class, inference_metadata

    try:
        generations = None
        inference_metadata = {}
        is_openai = is_openai_model(model.name_or_path)

        generations = []
        if is_openai:
            generations = [model.generate(model_input_prompt, max_new_tokens=100) for model_input_prompt in prompt]
        else:
            for input_text in [input_entry["text"]] if isinstance(input_entry["text"], str) else input_entry["text"]:
                if model.config.architectures[0].startswith("T5"):
                    tokenized_prompt = tokenizer.encode(input_text, return_tensors="pt", max_length=512).to(model.device)
                else:
                    formatted_prompt = wrap_classification_prompt_keywords(prompt[0], model.name_or_path)
                    truncation_length = tokenizer.model_max_length if tokenizer.model_max_length <= 10000 else 10000
                    tokenized_prompt = tokenizer.encode(formatted_prompt, return_tensors="pt", max_length=truncation_length).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(tokenized_prompt, max_new_tokens=100, length_penalty=0, early_stopping=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
                    start_decoding_index = len(tokenized_prompt[0]) if is_large_language_model(model.name_or_path) else 0
                    generation = tokenizer.decode(outputs["sequences"][0][start_decoding_index:], skip_special_tokens=True).split("\n")[0].replace("</s>", "").strip()
                    generations.append(generation)

        predicted_classes = []
        for generation in generations:
            predicted_classes.append(parse_generation_to_label(dataset_name, generation))

        inference_metadata["generations"] = generations
        inference_metadata["predicted_classes"] = predicted_classes
        majority_class = max(set(predicted_classes), key=predicted_classes.count)
        return majority_class, inference_metadata
    except Exception as e:
        print(f"Error for input {input_entry['text']} ---- Error: {e}")
        return -1


def parse_generation_to_label(dataset_name, generation):
    is_qa_task = dataset_name.startswith("squad")
    if is_qa_task:
        return generation

    leading_token = generation.strip()[0]
    final_tokens = generation.replace("</s>", "").replace("<s>", "")[-2:]
    if leading_token == "{" and final_tokens == "}":
        generation = generation.replace("{", "").replace("}", "").strip()

    possible_int_labels = [str(label) for label in range(get_num_labels(dataset_name))]
    if leading_token in possible_int_labels:
        return int(leading_token)

    if final_tokens[1] in possible_int_labels or final_tokens[1] in possible_int_labels:
        return int(final_tokens[1])
    elif final_tokens[0] == "0" or final_tokens[0] == "1":
        return int(final_tokens[0])

    split_tokens = [word.replace(".", "") for word in generation.split()]
    verbalizers = {
        "negative": 0,
        "positive": 1,
        "neutral": 2,
    }
    if split_tokens[0].lower() in verbalizers:
        return verbalizers[split_tokens[0].lower()]
    if split_tokens[-1].lower() in verbalizers:
        return verbalizers[split_tokens[0].lower()]

    extracted_integer = re.findall(r"\d+", generation)
    if len(extracted_integer) == 1:
        return int(extracted_integer[0])

    print(f"WARNING: Could not extract judgment from: {generation}")
    return -1


def should_get_exemplars(model, is_adaptive_set):
    return model.config.architectures[0].endswith("ForCausalLM") or is_adaptive_set


def evaluate_without_adaptation(rank, world_size, experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, num_shots=None):
    if dist.is_initialized():
        dist.barrier()

    should_retrieve_exemplars = should_get_exemplars(model, eval_set)
    icl_method = icl_method if should_retrieve_exemplars else None
    template = get_prompt_template(dataset_name) if should_retrieve_exemplars else None
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_judgments = []
    inference_logs = []
    num_failed_generations = 0
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name) if should_retrieve_exemplars else None

    sampler = DistributedSampler(dataset[eval_set.replace("+adaptive", "")]) if dist.is_initialized() else None
    data_loader = DataLoader(dataset[eval_set.replace("+adaptive", "")], sampler=sampler)
    if rank == 0:
        description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
        data_loader = tqdm(data_loader, desc=description)

    for entry in data_loader:
        entry["text"] = entry["text"][0] if isinstance(entry["text"], list) else entry["text"]
        entry["label"] = entry["label"].item() if isinstance(entry["label"], torch.Tensor) else entry["label"]
        start_time = time.perf_counter()
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            if icl_method == "static":
                exemplars = get_static_exemplars(dataset_name, num_shots)
            else:
                distance_goal = "NA" if not icl_method.startswith("topk") else icl_method if icl_method == "topk" else icl_method.split("_")[1]
                exemplars, mean_exemplar_distance = get_dynamic_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots, distance_goal) if should_retrieve_exemplars else None

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        inference = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        inference_metadata = inference[1] if isinstance(inference, tuple) else None
        judgment = inference[0] if isinstance(inference, tuple) else inference
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = inference_metadata if inference_metadata is not None else {}
        inference_log["latency"] = time.perf_counter() - start_time
        inference_log["input"] = entry["text"]
        if dataset_name.startswith("squad"):
            inference_log["question"] = entry["question"]
        inference_log["judgment"] = judgment
        if should_retrieve_exemplars:
            inference_log["prompt"] = prompt

        inference_log["label"] = entry["label"]
        inference_logs.append(inference_log)

    if rank == 0 and not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    distributed_inference_logs = None
    if rank == 0:
        distributed_inference_logs = [[] for i in range(world_size)]

    if dist.is_initialized():
        dist.gather_object(inference_logs, distributed_inference_logs)

    if rank == 0:
        eval_inference_logs = list(chain(*distributed_inference_logs)) if dist.is_initialized() else inference_logs
        save_inference_log(eval_inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, "No Adaptation", num_shots)
        dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
        inference_log_frame = pd.DataFrame(eval_inference_logs)
        return generate_evaluation_Report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, "No Adaptation", num_shots, num_failed_generations)


def get_outcome_type(original_judgment, styled_jdugment, label):
    if original_judgment == styled_jdugment and original_judgment != label:
        return "Unfixed Mistake"
    if original_judgment == styled_jdugment and original_judgment == label:
        return "Unchanged Correct"
    if original_judgment != styled_jdugment and original_judgment == label:
        return "New Mistake"
    if original_judgment != styled_jdugment and styled_jdugment == label:
        return "New Correct"
    return "NA"


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
        updated_cache_frame = cache_miss_frame if cache_frame is None else pd.concat([cache_frame, cache_miss_frame])
        updated_cache_frame.to_csv(cache_path, index=False)
    except Exception as e:
        print(f"Error writing cached rewrites: {e}")


def evaluate_style_transfer(rank, world_size, experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, adaptive_method_name=None, num_shots=None, trim_exemplars=False, temperature=0, transfer_prompt=None):
    if dist.is_initialized():
        dist.barrier()

    is_adaptive_set = adaptive_method_name is not None and adaptive_method_name != "No Adaptation"
    should_retrieve_exemplars = should_get_exemplars(model, evaluate_style_transfer)
    icl_method = icl_method if should_retrieve_exemplars else None
    template = get_prompt_template(dataset_name) if should_retrieve_exemplars else None
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_judgments = []
    inference_logs = []
    num_failed_generations = 0
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name) if should_retrieve_exemplars else None

    adaptive_tokenizer = None
    adaptive_model = None
    if is_adaptive_set:
        adaptive_tokenizer, adaptive_model = get_model_objects(adaptive_method_name, -1)

    sampler = DistributedSampler(dataset[eval_set.replace("+adaptive", "")]) if dist.is_initialized() else None
    data_loader = DataLoader(dataset[eval_set.replace("+adaptive", "")], sampler=sampler)
    if rank == 0:
        description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
        print(f"{description} and {adaptive_method_name} for style transfer" if is_adaptive_set else description)
        if not dist.is_initialized():
            data_loader = tqdm(data_loader, desc=description)

    for entry in data_loader:
        if dist.is_initialized():
            print(f"\nRank: {rank} | {len(inference_logs) + 1}/{len(data_loader)}")

        entry["text"] = entry["text"][0] if isinstance(entry["text"], list) else entry["text"]
        entry["label"] = entry["label"].item() if isinstance(entry["label"], torch.Tensor) else entry["label"]
        start_time = time.perf_counter()
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            if icl_method == "static":
                exemplars = get_static_exemplars(dataset_name, num_shots)
            else:
                distance_goal = "NA" if not icl_method.startswith("topk") else icl_method if icl_method == "topk" else icl_method.split("_")[1]
                exemplars, mean_exemplar_distance = get_dynamic_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots, distance_goal) if should_retrieve_exemplars else None

        if is_adaptive_set:
            entry["original_text"] = entry["text"]
            if dataset_name == "boss_nli":
                entry["text"] = entry["Premise"]
                entry["style_prompt"], styled_premise = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars, temperature, transfer_prompt)
                entry["text"] = entry["Hypothesis"]
                entry["style_prompt"], styled_hypothesis = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars, temperature, transfer_prompt)
                entry["text"] = f"{styled_premise} / {styled_hypothesis}"
            else:
                # cached_rewrites = get_cached_rewritess(dataset_name, eval_set, adaptive_method_name, icl_method, num_shots, temperature, entry)
                cached_rewrites = None
                if cached_rewrites == None:
                    entry["style_prompt"], entry["text"] = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars, temperature, transfer_prompt)
                else:
                    entry["style_prompt"], entry["text"] = cached_rewrites

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        inference = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        inference_metadata = inference[1] if isinstance(inference, tuple) else None
        judgment = inference[0] if isinstance(inference, tuple) else inference
        judgment = judgment[0] if isinstance(judgment, tuple) else judgment
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = inference_metadata if inference_metadata is not None else {}
        inference_log["latency"] = time.perf_counter() - start_time
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


    distributed_inference_logs = None
    if rank == 0:
        distributed_inference_logs = [[] for i in range(world_size)]

    if dist.is_initialized():
        dist.gather_object(inference_logs, distributed_inference_logs)
    if rank == 0:
        if not os.path.exists(f"results/{experiment_id}"):
            os.makedirs(f"results/{experiment_id}")

        eval_inference_logs = list(chain(*distributed_inference_logs)) if dist.is_initialized() else inference_logs
        dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
        save_inference_log(eval_inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots, trim_exemplars)

        # Save new mistakes_lods
        inference_log_frame = save_baseline_logs(experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots, eval_inference_logs)

        eval_reports = []
        for inference_method in ["ensemble", "entropy threshold half", "entropy threshold best", "entropy threshold+lowest", "lowest entropy", "single rewrite"]:
            if "entropy" not in inference_log_frame.columns and "entropy" in inference_method:
                print(f"Skipping {inference_method} because entropy was not calculated")
                continue

            eval_reports.append(generate_evaluation_Report(
                experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, adaptive_method_name, num_shots, num_failed_generations, trim_exemplars, temperature, inference_method
            ))

        return inference_log_frame, eval_reports


    # --------------

    # distributed_inference_logs = None
    # if rank == 0:
    #     distributed_inference_logs = [[] for i in range(world_size)]

    # dist.gather_object(inference_logs, distributed_inference_logs)

    # if rank == 0:
    #     distributed_inference_logs = list(chain(*distributed_inference_logs))

    #     save_inference_log(distributed_inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, "No Adaptation", num_shots)
    #     dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    #     inference_log_frame = pd.DataFrame(distributed_inference_logs)
    #     return generate_evaluation_Report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, "No Adaptation", num_shots, num_failed_generations)

def save_baseline_logs(experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots, inference_logs):
    # Save logs frame
    experiment_directory = f"results/{experiment_id}"
    experiment_run_prefix = f"{model_name.replace('/', '-')}-{dataset_name}-{icl_method}-{eval_set}-{adaptive_method_name.replace('/', '-')}-{num_shots}"
    no_adapt_logs = get_baseline_inference_log_frame(experiment_id, model_name, dataset_name, icl_method, eval_set)
    inference_log_frame = pd.DataFrame(inference_logs)
    inference_log_frame["original judgment"] = no_adapt_logs["judgment"]
    if "entropy" in inference_log_frame.columns:
        inference_log_frame["original entropy"] = no_adapt_logs["entropy"]
        inference_log_frame["entropy decrease"] =  inference_log_frame["original entropy"] - inference_log_frame["entropy"]
        inference_log_frame["entropy decreased"] = inference_log_frame["entropy"] < inference_log_frame["original entropy"]
    inference_log_frame["outcome"] = inference_log_frame.apply(lambda row: get_outcome_type(row["original judgment"], row["judgment"], row["label"]), axis=1)
    inference_log_frame.to_csv(f"{experiment_directory}/{experiment_run_prefix}-style_inference_log.csv", index=False)

    if "entropy" not in inference_log_frame.columns:
        return inference_log_frame

    # Save summary frame
    outcome_summary_frame = inference_log_frame.groupby("outcome").describe()
    outcome_summary_frame.to_csv(f"{experiment_directory}/{experiment_run_prefix}-style_inference_outcome_summary.csv")
    entropy_change_table = inference_log_frame.value_counts(["outcome", "entropy decreased"])
    entropy_change_table.to_csv(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_change_table.csv")

    # Save entropy plots
    entropy_plot = px.scatter(inference_log_frame, y="entropy", color="outcome", title=f"Entropy by Outcome: {experiment_run_prefix}")
    entropy_plot.write_image(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_plot.png")
    entropy_plot.write_html(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_plot.html")
    entropy_plot_log = px.scatter(inference_log_frame, y="entropy", color="outcome", title=f"Entropy by Outcome: {experiment_run_prefix}", log_y=True)
    entropy_plot_log.write_image(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_plot_log.png")
    entropy_plot_log.write_html(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_plot_log.html")
    entropy_delta_plot = px.scatter(inference_log_frame, y="entropy decrease", color="outcome", title=f"Entropy Decrease by Outcome: {experiment_run_prefix}")
    entropy_delta_plot.write_image(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_delta_plot.png")
    entropy_delta_plot.write_html(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_delta_plot.html")
    entropy_delta_plot_log = px.scatter(inference_log_frame, y="entropy decrease", color="outcome", title=f"Entropy Decrease by Outcome: {experiment_run_prefix}", log_y=True)
    entropy_delta_plot_log.write_image(f"{experiment_directory}/{experiment_run_prefix}-style_inference_entropy_delta_plot_log.png")
    entropy_delta_plot_log.write_html(f"{experiment_directory}/{experiment_run_prefix}-style_inference_eentropy_delta_plot_log.html")

    # Save embedding plots
    # embedding_tokenizer, embedding_model = get_model_objects("princeton-nlp/sup-simcse-roberta-large", num_labels=-1)
    # def get_embedding(text):
    #     input_ids = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(embedding_model.device)
    #     embedding = embedding_model(input_ids).pooler_output[0].cpu().detach().numpy()
    #     return embedding

    # inference_log_frame["original_embedding"] = inference_log_frame.progress_apply(lambda row: get_embedding(row["original_input"]), axis=1)
    # inference_log_frame["rewritten_embedding"] = inference_log_frame.progress_apply(lambda row: get_embedding(row["input"]), axis=1)
    # umap_2d = UMAP(n_components=2, init='random', random_state=0)
    # all_embedding_projections = umap_2d.fit_transform(pd.concat([inference_log_frame["original_embedding"], inference_log_frame["rewritten_embedding"]]).tolist())
    # inference_log_frame["original_projection"] = pd.Series(all_embedding_projections[len(inference_log_frame):].tolist())
    # inference_log_frame["rewritten_projection"] = pd.Series(all_embedding_projections[:len(inference_log_frame)].tolist())

    return inference_log_frame


def get_baseline_inference_log_frame(experiment_id, model_name, dataset_name, icl_method, eval_set):
    compare_file_name_prefix = None
    if is_large_language_model(model_name):
        if dataset_name.startswith("boss_"):
            set_name = dataset_name.split("-")[0]
            compare_file_name_prefix = f'{model_name.replace("/", "-")}-{set_name}-{eval_set}-{icl_method}-No Adaptation'
        else:
            compare_file_name_prefix = f'{model_name.replace("/", "-")}-{dataset_name}-{eval_set}-{icl_method}-No Adaptation'

    else:
        if dataset_name.startswith("boss_"):
            set_name = dataset_name.split("-")[0]
            compare_file_name_prefix = f'{model_name.replace("/", "-")}-{set_name}-{eval_set}-static-No Adaptation'
        else:
            compare_file_name_prefix = f'{model_name.replace("/", "-")}-{dataset_name}-{eval_set}-static-No Adaptation'

    no_adapt_logs_filename = [file_name for file_name in os.listdir(f"results/{experiment_id}") if compare_file_name_prefix in file_name][0]
    return pd.read_csv(f"results/{experiment_id}/{no_adapt_logs_filename}")


def save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots, trim_exemplars="NA"):
    current_logs = pd.DataFrame(inference_logs)
    model_name = model_name.replace("/", "-")
    adaptive_method_name = adaptive_method_name.replace("/", "-") if adaptive_method_name is not None else None
    current_logs.to_csv(
        f"results/{experiment_id}/{model_name}-{dataset_name}-{eval_set}-{icl_method}-{adaptive_method_name}-{num_shots}-TrimExemplars={trim_exemplars}-inference-logs.csv", index=False
    )
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
        combined_inference_log["trim_exemplars"] = trim_exemplars

    for saved_log_name in saved_logs_names:
        if not saved_log_name.startswith(f"{model_name.replace('/', '-')}-{dataset_name}-{eval_set}"):
            continue

        prev_log = pd.read_csv(f"results/{experiment_id}/{saved_log_name}")
        column_name_prefix = f"{adaptive_method_name}-{num_shots}"
        combined_inference_log[f"{column_name_prefix} Judgment"] = prev_log["judgment"]
        combined_inference_log[f"{column_name_prefix} Input"] = prev_log["input"]
        combined_inference_log[f"{column_name_prefix} Prompt"] = prev_log["style prompt"]

    combined_inference_log.to_csv(f"results/{experiment_id}/{combined_inference_log_file_name}")


def wrap_rewrite_prompt_keywords(prompt, model_name):
    if "vicuna" in model_name:
        return f"User: {prompt} Assistant:"
    elif "xgen-7b-8k-inst" in model_name:
        return f"### Human: {prompt.replace('###', '---').strip()}\n###"
    elif "oasst" in model_name:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>"
    elif "StableBeluga" in model_name:
        system_message = prompt.split("### Input Text ###")[0]
        user_message = "### Input Text ###" + prompt.split("### Input Text ###")[1]
        return f"### System:\n{system_message}### User:\n{user_message}\n\n### Assistant:\n"
    else:
        return prompt


def wrap_classification_prompt_keywords(prompt, model_name):
    user_message = prompt.split("\n")[-1]
    system_message = prompt.split(user_message)[0]

    if "vicuna" in model_name:
        return f"User: {prompt} Assistant:"
    elif "StableBeluga" in model_name:
        return f"### System:\n{system_message}### User:\n{user_message}\n\n### Assistant:\n"
    else:
        return prompt


def get_transferred_input(adaptive_tokenizer, adaptive_model, input_entry, exemplars, trim_exemplars, temperature, transfer_prompt):
    style_input = input_entry["text"].replace("\n", " ")
    is_openai = is_openai_model(adaptive_model.name_or_path)
    num_example_tokens = adaptive_tokenizer(style_input, return_tensors="pt")["input_ids"].shape[1] if adaptive_tokenizer is not None else len(style_input)

    input_prompts = None
    if is_large_language_model(adaptive_model.name_or_path):
        style_transfer_exemplars = None
        if is_openai:
            style_transfer_exemplars = "".join(['- "' + exemplar["text"].strip().replace("\n", "")[:500] + '"\n' for exemplar in exemplars])
        elif trim_exemplars:
            style_transfer_exemplars = "".join([f'- "{adaptive_tokenizer.decode(adaptive_tokenizer.encode(exemplar["text"].strip())[:int(500 / len(exemplars))])}"\n' for exemplar in exemplars])
        else:
            style_transfer_exemplars = "".join(['- "' + exemplar["text"].strip().replace("\n", "") + '"\n' for exemplar in exemplars])

        task_prompt = None
        with open(f"prompts/{transfer_prompt}.txt", "r") as style_transfer_prompt_file:
            prompt_template = style_transfer_prompt_file.read()
            prompt_template = prompt_template.replace("<style_transfer_exemplars>", style_transfer_exemplars)
            prompt_template = prompt_template.replace("<style_input>", style_input)
            prompt_template = prompt_template.replace("<s>", "")
            task_prompt = prompt_template

        input_prompts = wrap_rewrite_prompt_keywords(task_prompt, adaptive_model.config.name_or_path)
    else:
        input_prompts = style_input

    # Try reading from the cache. If the cache doesn't exist, generate a new rewrite
    cached_rewrites = get_cached_rewrites(adaptive_model, temperature, input_prompts)
    if cached_rewrites is not None:
        return input_prompts, cached_rewrites

    tokenized_prompt = input_prompts if adaptive_tokenizer is None else adaptive_tokenizer.encode(input_prompts, return_tensors="pt").to(adaptive_model.device)
    try:
        with torch.no_grad():
            outputs = adaptive_model.generate(
                tokenized_prompt,
                temperature=temperature,
                max_new_tokens=num_example_tokens * 5,
                early_stopping=True,
                return_dict_in_generate=True,
                num_return_sequences=4,
                num_beam_groups=4,
                num_beams=4,
                top_p=0.95,
                top_k=0,
                repetition_penalty=10.0,
                diversity_penalty=1.0,
                no_repeat_ngram_size=2,
            )
    except torch.cuda.OutOfMemoryError as generation_error:
        print(generation_error)
        print(f"Ran out of memory when generating an input for the following prompt: {input_prompts}")
        return input_prompts, None

    formatted_generated_sequences = []
    if is_openai:
        return input_prompts, [outputs["choices"][0]["message"]["content"]]

    for output in outputs[0]:
        if isinstance(output, str):
            formatted_generated_sequences.append(output)
            continue

        generation = None
        if is_large_language_model(adaptive_model.name_or_path):
            generation = adaptive_tokenizer.decode(output[len(tokenized_prompt[0]) :])
        else:
            generation = adaptive_tokenizer.decode(output, skip_special_tokens=True)

        parsed_generation = parse_generation(style_input, generation)
        formatted_generated_sequences.append(parsed_generation)

    # Write rewrites to cache
    write_cached_rewrites(adaptive_model, temperature, input_prompts, formatted_generated_sequences)

    print(f"\n\nOriginal Input: {input_entry['text']}")
    print("Rewrites:\n- " + "\n- ".join(formatted_generated_sequences))
    return input_prompts, formatted_generated_sequences


def parse_generation(style_input, generation):
    generation = generation.replace("\n", " ").replace("</s>", "").replace("```", "").strip()

    if "###" in generation:
        generation = generation.split("###")[0]
    if "</s>" in generation:
        generation = generation.split("</s>")[0]
    if "<s>" in generation:
        generation = generation.replace("<s>", " ").strip()
    if "<unk>" in generation:
        generation = generation.replace("<unk>", " ").strip()
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]
    if "<|endoftext|>" in generation:
        generation = generation.split("<|endoftext|>")[0]
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]
    if "Input Text:" in generation:
        generation = generation.split("Input Text:")[0].strip()
    if '"  Assistant: ' in generation:
        generation = generation.split('"  Assistant: ')[0]
        if generation[0] == '"':
            generation = generation[1:]
        if generation[-1] == '"':
            generation = generation[:-1]
    if "<end task example>" in generation:
        generation = generation.split("<end task example>")[0].strip()
    if generation.startswith('"') and generation.endswith('"'):
        generation = generation[1:-1]
    if generation.startswith("Assistant:"):
        generation = generation.split("Assistant:")[1].strip()
    if generation.startswith("Paraphrased:"):
        generation = generation.split("Paraphrased:")[1].strip()
    if generation.startswith('"'):
        generation = generation.split('"')[1]

    if generation.strip() == "":
        print("Generation was empty")

    return generation


# TODO: Add support for LLM inference
def evaluate_test_time_augmentation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, eval_set, icl_method, aug_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_tokenizer, paraphrase_model = get_model_objects("humarin/chatgpt_paraphraser_on_T5_base", num_labels=-1)
    aug = naw.ContextualWordEmbsAug(action="substitute", device="cuda")
    inference_logs = []

    print(f"Evaluating {dataset_name} with {model_name} using TTA baseline")
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")]):
        start_time = time.perf_counter()
        original_text_input = entry["text"]

        augmented_inputs = None
        if dataset_name == "boss_nli":
            premises = aug.augment(entry["Premise"], n=4) if aug_method == "replace" else get_paraphrase_augmentations(entry["Premise"], paraphrase_tokenizer, paraphrase_model, device)
            hypothesis = aug.augment(entry["Hypothesis"], n=4) if aug_method == "replace" else get_paraphrase_augmentations(entry["Hypothesis"], paraphrase_tokenizer, paraphrase_model, device)
            augmented_inputs = [f"{p} / {h}" for (p, h) in zip(premises, hypothesis)]
        else:
            augmented_inputs = get_augmentations(aug_method, device, paraphrase_tokenizer, paraphrase_model, aug, original_text_input)

        logits = []
        judgments = []
        tta_inputs = [original_text_input] + augmented_inputs
        for aug_input in tta_inputs:
            input_entry = entry.copy()
            input_entry["text"] = aug_input
            aug_judgment, aug_logits = get_judgment(model, tokenizer, aug_input, device, input_entry, dataset_name)
            logits.append(aug_logits)
            judgments.append(aug_judgment)

        final_judgment = torch.stack(logits).mean(dim=0).argmax().detach().item()
        inference_log = {}
        inference_log["latency"] = time.perf_counter() - start_time
        inference_log["input"] = entry["text"]
        inference_log["label"] = entry["label"]
        inference_log["judgment"] = final_judgment
        inference_log["original_input"] = entry["text"]
        inference_log["style prompt"] = ", ".join(augmented_inputs)
        inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, f"test_time_aug_{aug_method}", None)
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    inference_log_frame = pd.DataFrame(inference_logs)
    return generate_evaluation_Report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, f"Test-Time Augmentation - {aug_method}")


def get_augmentations(aug_method, device, paraphrase_tokenizer, paraphrase_model, aug, original_text_input):
    if aug_method == "replace":
        return aug.augment(original_text_input, n=4)
    if aug_method == "paraphrase":
        return get_paraphrase_augmentations(original_text_input, paraphrase_tokenizer, paraphrase_model, device)
    # if aug_method == "rewrite":

    return aug.augment(original_text_input, n=4) if aug_method == "replace" else get_paraphrase_augmentations(original_text_input, paraphrase_tokenizer, paraphrase_model, device)


# TODO: Add support for LLM inference
def evaluate_memo(experiment_id, task_model_name, task_model, task_tokenizer, dataset_name, dataset, eval_set, icl_method, aug_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_tokenizer, paraphrase_model = get_model_objects("humarin/chatgpt_paraphraser_on_T5_base", num_labels=-1)
    optimizer = AdamW(task_model.parameters(), lr=0.000001, weight_decay=0.01)
    aug = naw.ContextualWordEmbsAug(action="substitute", device="cuda")

    inference_logs = []
    entropies = []
    print(f"Evaluating {dataset_name} with {task_model_name} using MEMO baseline")
    for entry in tqdm(dataset[eval_set]):
        start_time = time.perf_counter()
        task_model.train()
        optimizer.zero_grad()

        # Get the augmentations for the current input and compute the marginal
        # entropy. Then backpropagate the marginal entropy before predicting.
        original_text_input = entry["text"]
        augmentations = None
        if dataset_name == "boss_nli":
            premises = aug.augment(entry["Premise"], n=4) if aug_method == "replace" else get_paraphrase_augmentations(entry["Premise"], paraphrase_tokenizer, paraphrase_model, device)
            hypothesis = aug.augment(entry["Hypothesis"], n=4) if aug_method == "replace" else get_paraphrase_augmentations(entry["Hypothesis"], paraphrase_tokenizer, paraphrase_model, device)
            augmentations = [f"{p} / {h}" for (p, h) in zip(premises, hypothesis)]
        else:
            augmentations = aug.augment(original_text_input, n=4) if aug_method == "replace" else get_paraphrase_augmentations(original_text_input, paraphrase_tokenizer, paraphrase_model, device)

        aug_tokens = task_tokenizer(augmentations, return_tensors="pt", padding="longest").to(device)
        aug_logits = task_model(**aug_tokens).logits
        aug_probs = aug_logits.softmax(dim=1)
        marginal_probs = aug_probs.mean(dim=0)
        marginal_entropy = -torch.sum(marginal_probs * torch.log(marginal_probs))
        marginal_entropy.backward()
        entropies.append(marginal_entropy.item())
        optimizer.step()

        # Make the prediciton for the current original input with the new model weights
        with torch.no_grad():
            task_model.eval()
            input_tokens = task_tokenizer(original_text_input, return_tensors="pt").to(device)
            input_logits = task_model(**input_tokens).logits
            final_judgment = torch.argmax(input_logits).detach().item()

            inference_log = {}
            inference_log["latency"] = time.perf_counter() - start_time
            inference_log["input"] = entry["text"]
            inference_log["label"] = entry["label"]
            inference_log["judgment"] = final_judgment
            inference_log["original_input"] = entry["text"]
            inference_log["style prompt"] = ", ".join(augmentations)
            inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, task_model_name, dataset_name, icl_method, eval_set, f"memo_{aug_method}", None)
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    inference_log_frame = pd.DataFrame(inference_logs)
    return generate_evaluation_Report(experiment_id, task_model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, f"MEMO - {aug_method}")


def get_paraphrase_augmentations(
    question,
    paraphrase_tokenizer,
    paraphrase_model,
    device,
    num_return_sequences=4,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128,
):
    input_ids = paraphrase_tokenizer(
        f"paraphrase: {question}",
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = paraphrase_model.generate(
        input_ids,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_return_sequences,
        num_beam_groups=num_return_sequences,
        max_length=max_length,
        diversity_penalty=diversity_penalty,
    )

    res = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def evaluate_fine_tuning(experiment_id, task_model_name, task_model, task_tokenizer, dataset_name, dataset, eval_set, icl_method):
    device = task_model.device
    optimizer = AdamW(task_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    task_dataset = GenericDataset(dataset[eval_set])
    data_loader = DataLoader(task_dataset, batch_size=8)
    start_funetuning_time = time.perf_counter()

    print(f"Fine-Tuning {task_model_name} on {dataset_name}")
    is_lm = is_language_model(task_model_name)
    for batch_inputs, batch_labels in tqdm(data_loader):
        task_model.train()
        optimizer.zero_grad()
        tokenized_batch = task_tokenizer(batch_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        labels = task_tokenizer([str(label) for label in batch_labels.tolist()], return_tensors="pt", padding=True, truncation=True, max_length=512) if is_lm else batch_labels
        labels = labels.input_ids if is_lm else labels
        labels = labels.to(task_model.device)

        loss = task_model(**tokenized_batch, labels=labels).loss
        loss.backward()
        optimizer.step()

    task_model.eval()
    optimizer.zero_grad()
    inference_logs = []

    fine_tuning_latency = (time.perf_counter() - start_funetuning_time) / len(dataset[eval_set])
    print(f"Evaluating {dataset_name}-{eval_set} with {task_model_name} using fine-tuning baseline")
    for entry in tqdm(dataset[eval_set]):
        with torch.no_grad():
            # eval_text = entry["text"]
            # tokenized_sample = task_tokenizer(eval_text, return_tensors="pt").to(device)
            # logits = task_model(**tokenized_sample).logits
            # eval_prediciton = torch.argmax(logits, dim=1).cpu().item()
            eval_prediciton = get_judgment(task_model, task_tokenizer, entry["text"], task_model.device, entry, dataset_name)
            inference_log = {}
            inference_log["latency"] = fine_tuning_latency
            inference_log["input"] = entry["text"]
            inference_log["label"] = entry["label"]
            inference_log["judgment"] = eval_prediciton
            inference_log["original_input"] = entry["text"]
            inference_log["style prompt"] = ""
            inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, task_model_name, dataset_name, icl_method, eval_set, "fine_tuning", None)
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    inference_log_frame = pd.DataFrame(inference_logs)
    return generate_evaluation_Report(experiment_id, task_model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, "Fine-Tuning")


class GenericDataset(Dataset):
    def __init__(self, in_dataset):
        self.dataset = in_dataset

    def __getitem__(self, index):
        return self.dataset["text"][index], self.dataset["label"][index]

    def __len__(self):
        return len(self.dataset["text"])
