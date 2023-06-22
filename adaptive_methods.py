from torch.utils.data import DataLoader, Dataset
from transformers import pipeline
from openicl import DatasetReader
from torch.optim import AdamW
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import pandas as pd
import torch
import time
import os
import re

from util_data import generate_evaluation_Report, get_num_labels
from util_modeling import get_model_objects, is_large_language_model, is_language_model
from util_icl import generate_prompt, get_prompt_template, get_retriever, get_static_exemplars, get_dynamic_exemplars


# TODO: Return logits for LLMs and QA
def get_judgment(model, tokenizer, prompt, device, input_entry, dataset_name):
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
            input_sequence = tokenizer(input_entry["text"], return_tensors="pt", truncation=True).to(device)
            outputs = model(**input_sequence)
            logits = outputs.logits
            predicted_class = int(logits.argmax(axis=1))

            if is_nli_task:
                nl_judgment = model.config.id2label[predicted_class].lower()
                token_label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
                return token_label_map[nl_judgment], logits

            return predicted_class, logits

    try:
        tokenized_prompt = None
        if model.config.architectures[0].startswith("T5"):
            tokenized_prompt = tokenizer.encode(input_entry["text"], return_tensors="pt", max_length=512).to(device)
        else:
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt", max_length=tokenizer.model_max_length).to(device)

        with torch.no_grad():
            outputs = model.generate(tokenized_prompt, max_new_tokens=100, length_penalty=0, early_stopping=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
            start_decoding_index = len(tokenized_prompt[0]) if is_large_language_model(model.name_or_path) else 0
            generation = tokenizer.decode(outputs["sequences"][0][start_decoding_index:], skip_special_tokens=True).split("\n")[0].replace("</s>", "").strip()

        is_qa_task = dataset_name.startswith("squad")
        if is_qa_task:
            return generation

        leading_token = generation.strip()[0]
        possible_int_labels = [str(label) for label in range(get_num_labels(dataset_name))]
        if leading_token in possible_int_labels:
            return int(leading_token)

        final_tokens = generation.replace("</s>", "").replace("<s>", "")[-2:]
        if final_tokens[1] in possible_int_labels or final_tokens[1] in possible_int_labels:
            return int(final_tokens[1])
        elif final_tokens[0] == "0" or final_tokens[0] == "1":
            return int(final_tokens[0])

        split_tokens = [word.replace(".", "") for word in generation.split()]
        if split_tokens[0].lower() in ["positive", "negative"]:
            return 0 if split_tokens[0].lower() == "negative" else 1
        if split_tokens[-1].lower() in ["positive", "negative"]:
            return 0 if split_tokens[-1].lower() == "negative" else 1

        extracted_integer = re.findall(r"\d+", generation)
        if len(extracted_integer) == 1:
            return int(extracted_integer[0])

        return -1
    except Exception as e:
        print(f"Error for input {input_entry} ---- {e}")
        return -1


def should_get_exemplars(model, is_adaptive_set):
    return model.config.architectures[0].endswith("ForCausalLM") or is_adaptive_set


def evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, num_shots=None):
    should_retrieve_exemplars = should_get_exemplars(model, eval_set)
    icl_method = icl_method if should_retrieve_exemplars else None
    template = get_prompt_template(dataset_name) if should_retrieve_exemplars else None
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_judgments = []
    inference_logs = []
    num_failed_generations = 0
    exemplar_retriever = get_retriever(icl_method, data_reader, dataset_name) if should_retrieve_exemplars else None

    description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")]):
        start_time = time.perf_counter()
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            if icl_method == "static":
                exemplars = get_static_exemplars(dataset_name, num_shots)
            else:
                distance_goal = "NA" if not icl_method.startswith("topk") else icl_method if icl_method == "topk" else icl_method.split("_")[1]
                exemplars, mean_exemplar_distance = get_dynamic_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots, distance_goal) if should_retrieve_exemplars else None

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        judgment = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        judgment = judgment[0] if isinstance(judgment, tuple) else judgment
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = {}
        inference_log["latency"] = time.perf_counter() - start_time
        inference_log["input"] = entry["text"]
        if dataset_name.startswith("squad"):
            inference_log["question"] = entry["question"]
        inference_log["judgment"] = judgment
        if should_retrieve_exemplars:
            inference_log["prompt"] = prompt

        inference_log["label"] = entry["label"]
        inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, "No Adaptation", num_shots)
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    inference_log_frame = pd.DataFrame(inference_logs)
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


def evaluate_style_transfer(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, adaptive_method_name=None, num_shots=None, trim_exemplars=False):
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

    description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
    print(f"{description} and {adaptive_method_name} for style transfer" if is_adaptive_set else description)
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")]):
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
                entry["style_prompt"], styled_premise = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars)
                entry["text"] = entry["Hypothesis"]
                entry["style_prompt"], styled_hypothesis = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars)
                entry["text"] = f"{styled_premise} / {styled_hypothesis}"
            else:
                entry["style_prompt"], entry["text"] = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars, trim_exemplars)

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        judgment = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        judgment = judgment[0] if isinstance(judgment, tuple) else judgment
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = {}
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

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots, trim_exemplars)

    # Save new mistakes_lods

    no_adapt_logs = get_baseline_inference_log_frame(experiment_id, model_name, dataset_name, icl_method, eval_set)
    inference_log_frame = pd.DataFrame(inference_logs)
    inference_log_frame["original judgment"] = no_adapt_logs["judgment"]
    inference_log_frame["outcome"] = inference_log_frame.apply(lambda row: get_outcome_type(row["original judgment"], row["judgment"], row["label"]), axis=1)

    return inference_log_frame, generate_evaluation_Report(
        experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, inference_log_frame, adaptive_method_name, num_shots, num_failed_generations, trim_exemplars
    )


def get_baseline_inference_log_frame(experiment_id, model_name, dataset_name, icl_method, eval_set):
    compare_file_name_prefix = None
    if is_large_language_model(model_name):
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


def get_transferred_input(adaptive_tokenizer, adaptive_model, input_entry, exemplars, trim_exemplars):
    style_input = input_entry["text"].replace("\n", " ")
    num_example_tokens = adaptive_tokenizer(style_input, return_tensors="pt")["input_ids"].shape[1]
    style_transfer_exemplars = None
    if trim_exemplars:
        style_transfer_exemplars = "".join([f'"{adaptive_tokenizer.decode(adaptive_tokenizer.encode(exemplar["text"].strip())[:num_example_tokens * 2])}"\n' for exemplar in exemplars])
    else:
        style_transfer_exemplars = "".join([f'"{exemplar["text"].strip()}"\n' for exemplar in exemplars])

    task_prompt = f"""Paraphrase the input text into the exact writing style of the following examples while keeping the same semantic meaning. Keep all facts and information.
Examples:
{style_transfer_exemplars}
Now paraphrase the new domain input text into the same style as the old domain examples. Only return the paraphrased text for the below input text. Make sure to keep all facts, information, and meaning.

Input Text: "{style_input}\"\n""".replace(
        "<s>", ""
    )
    input_prompts = f"User: {task_prompt}\nAssistant:" if "vicuna" in adaptive_model.config.name_or_path else task_prompt

    tokenized_prompt = adaptive_tokenizer.encode(input_prompts, return_tensors="pt").to("cuda")
    try:
        with torch.no_grad():
            outputs = adaptive_model.generate(
                tokenized_prompt,
                do_sample=False,
                max_new_tokens=num_example_tokens * 3,
                early_stopping=True,
                return_dict_in_generate=True,
            )
    except torch.cuda.OutOfMemoryError as generation_error:
        print(generation_error)
        print(f"Ran out of memory when generating an input for the following prompt: {input_prompts}")
        return input_prompts, ""

    generation = adaptive_tokenizer.decode(outputs["sequences"][0][len(tokenized_prompt[0]) :]).replace("\n", " ").replace("</s>", "").strip()
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
    if "Input Text:" in generation:
        generation = generation.split("Input Text:")[0].strip()

    return input_prompts, generation


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
            augmented_inputs = aug.augment(original_text_input, n=4) if aug_method == "replace" else get_paraphrase_augmentations(original_text_input, paraphrase_tokenizer, paraphrase_model, device)

        logits = []
        judgments = []
        tta_inputs = [original_text_input] + augmented_inputs
        for aug_input in tta_inputs:
            input_entry = entry.copy()
            input_entry["text"] = aug_input
            aug_judgment, aug_logits = get_judgment(model, tokenizer, input_entry["text"], device, entry, dataset_name)
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


# TODO: Add support for LLM inference
def evaluate_memo(experiment_id, task_model_name, task_model, task_tokenizer, dataset_name, dataset, eval_set, icl_method, aug_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_tokenizer, paraphrase_model = get_model_objects("humarin/chatgpt_paraphraser_on_T5_base", num_labels=-1)
    optimizer = AdamW(task_model.parameters(), lr=0.0000003)
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
    num_beams=5,
    num_beam_groups=5,
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
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
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
