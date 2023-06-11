from torch.utils.data import DataLoader, Dataset
from transformers import pipeline
from openicl import DatasetReader
from torch.optim import AdamW
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import pandas as pd
import torch
import os

from util_data import generate_icl_report
from util_modeling import get_model_objects
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

    is_qa_task = dataset_name.startswith("squad")
    tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            tokenized_prompt, max_new_tokens=50 if is_qa_task else 50, length_penalty=0, early_stopping=True, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id
        )

        generation = tokenizer.decode(outputs["sequences"][0][len(tokenized_prompt[0]) :]).split("\n")[0].replace("</s>", "").strip()
    try:
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
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            if icl_method == "static":
                exemplars = get_static_exemplars(dataset_name)
            else:
                exemplars, mean_exemplar_distance = get_dynamic_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots) if should_retrieve_exemplars else None

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        judgment = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        judgment = judgment[0] if isinstance(judgment, tuple) else judgment
        original_judgments.append(judgment)
        if judgment == -1:
            num_failed_generations += 1
            print(f"Warning: {model_name} failed to generate a judgment for the following input: {entry['text']}")

        inference_log = {}
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
    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, "No Adaptation", num_shots, num_failed_generations)


def evaluate_style_transfer(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, eval_set, adaptive_method_name=None, num_shots=None):
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
        adaptive_tokenizer, adaptive_model = get_model_objects(adaptive_method_name)

    description = f"Evaluating {dataset_name}-{eval_set} with {model_name} using {icl_method}"
    print(f"{description} and {adaptive_method_name} for style transfer" if is_adaptive_set else description)
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")]):
        exemplars = mean_exemplar_distance = None
        if should_retrieve_exemplars:
            if icl_method == "static":
                exemplars = get_static_exemplars(dataset_name)
            else:
                exemplars, mean_exemplar_distance = get_dynamic_exemplars(entry["text"], dataset_name, exemplar_retriever, num_shots) if should_retrieve_exemplars else None

        if is_adaptive_set:
            entry["original_text"] = entry["text"]
            entry["style_prompt"], entry["text"] = get_transferred_input(adaptive_tokenizer, adaptive_model, entry, exemplars)

        prompt = generate_prompt(model_name, template, exemplars, entry, dataset_name) if should_retrieve_exemplars else None
        judgment = get_judgment(model, tokenizer, prompt, device, entry, dataset_name)
        judgment = judgment[0] if isinstance(judgment, tuple) else judgment
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

    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots)
    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, adaptive_method_name, num_shots, num_failed_generations)


def save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, adaptive_method_name, num_shots):
    current_logs = pd.DataFrame(inference_logs)
    model_name = model_name.replace("/", "-")
    adaptive_method_name = adaptive_method_name.replace("/", "-") if adaptive_method_name is not None else None
    current_logs.to_csv(f"results/{experiment_id}/{model_name}-{dataset_name}-{eval_set}-{icl_method}-{adaptive_method_name}-{num_shots}-inference-logs.csv", index=False)
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
        column_name_prefix = f"{adaptive_method_name}-{num_shots}"
        combined_inference_log[f"{column_name_prefix} Judgment"] = prev_log["judgment"]
        combined_inference_log[f"{column_name_prefix} Input"] = prev_log["input"]
        combined_inference_log[f"{column_name_prefix} Prompt"] = prev_log["style prompt"]
    combined_inference_log.to_csv(f"results/{experiment_id}/{combined_inference_log_file_name}")


def get_transferred_input(adaptive_tokenizer, adaptive_model, input_entry, exemplars):
    style_input = input_entry["text"].replace("\n", " ")
    # style_transfer_exemplars = "".join([f'"{exemplar["text"].strip()[:len(style_input)]}"\n' for exemplar in exemplars])
    style_transfer_exemplars = "".join([f'"{exemplar["text"].strip()}"\n' for exemplar in exemplars])
    task_prompt = f"""Paraphrase the input text into the exact writing style of the following examples while keeping the same semantic meaning. Keep all facts and information.
Examples:
{style_transfer_exemplars}
Now paraphrase the current input text into the same style as the examples. Only return the paraphrased text for the below input text. MAke sure to keep all facts, information, and meaning.

Input Text: "{style_input}\""""
    input_prompts = f"User: {task_prompt}\nAssistant:" if "vicuna" in adaptive_model.config.name_or_path else task_prompt
    tokenized_prompt = adaptive_tokenizer.encode(input_prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = adaptive_model.generate(
            tokenized_prompt,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=300,
            length_penalty=0,
            repetition_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
        )

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

    # print(f"Generation: {generation}")
    return input_prompts, generation


# TODO: Add support for LLM inference
def evaluate_test_time_augmentation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, eval_set, icl_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_logs = []
    aug = naw.ContextualWordEmbsAug(action="substitute", device="cuda")

    print(f"Evaluating {dataset_name} with {model_name} using TTA baseline")
    for entry in tqdm(dataset[eval_set.replace("+adaptive", "")]):
        original_text_input = entry["text"]
        augmented_inputs = aug.augment(original_text_input, n=4)
        logits = []
        judgments = []
        tta_inputs = [original_text_input] + augmented_inputs
        for aug_input in tta_inputs:
            input_entry = entry.copy()
            input_entry["text"] = aug_input
            aug_judgment, aug_logits = get_judgment(model, tokenizer, None, device, entry, dataset_name)
            logits.append(aug_logits)
            judgments.append(aug_judgment)

        final_judgment = torch.stack(logits).mean(dim=0).argmax().detach().item()
        inference_log = {}
        inference_log["input"] = entry["text"]
        inference_log["label"] = entry["label"]
        inference_log["judgment"] = final_judgment
        inference_log["original_input"] = entry["text"]
        inference_log["style prompt"] = ", ".join(augmented_inputs)
        inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, model_name, dataset_name, icl_method, eval_set, "test_time_aug", None)
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    original_judgments = [log["judgment"] for log in inference_logs]
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    return generate_icl_report(experiment_id, model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, "Test-Time Augmentation")


# TODO: Add support for LLM inference
def evaluate_memo(experiment_id, task_model_name, task_model, task_tokenizer, dataset_name, dataset, eval_set, icl_method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paraphrase_tokenizer, paraphrase_model = get_model_objects("humarin/chatgpt_paraphraser_on_T5_base")
    optimizer = AdamW(task_model.parameters(), lr=0.0000003)

    inference_logs = []
    entropies = []
    print(f"Evaluating {dataset_name} with {task_model_name} using MEMO baseline")
    for entry in tqdm(dataset[eval_set]):
        task_model.train()
        optimizer.zero_grad()

        # Get the augmentations for the current input and compute the marginal
        # entropy. Then backpropagate the marginal entropy before predicting.
        original_text_input = entry["text"]
        augmentations = get_paraphrase_augmentations(original_text_input, paraphrase_tokenizer, paraphrase_model, device)
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
            inference_log["input"] = entry["text"]
            inference_log["label"] = entry["label"]
            inference_log["judgment"] = final_judgment
            inference_log["original_input"] = entry["text"]
            inference_log["style prompt"] = ", ".join(augmentations)
            inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, task_model_name, dataset_name, icl_method, eval_set, "memo", None)
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    original_judgments = [log["judgment"] for log in inference_logs]
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    return generate_icl_report(experiment_id, task_model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, "MEMO")


def get_paraphrase_augmentations(
    question,
    paraphrase_tokenizer,
    paraphrase_model,
    device,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
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
    data_loader = DataLoader(task_dataset, batch_size=16)

    print(f"Fine-Tuning {task_model_name} on {dataset_name}")
    for batch_inputs, batch_labels in tqdm(data_loader):
        task_model.train()
        optimizer.zero_grad()
        tokenized_batch = task_tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt").to(device)
        labels = batch_labels.to(device)

        logits = task_model(**tokenized_batch).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    task_model.eval()
    optimizer.zero_grad()
    inference_logs = []

    print(f"Evaluating {dataset_name}-{eval_set} with {task_model_name} using fine-tuning baseline")
    for entry in tqdm(dataset[eval_set]):
        with torch.no_grad():
            eval_text = entry["text"]
            eval_label = entry["label"]
            tokenized_sample = task_tokenizer(eval_text, return_tensors="pt").to(device)
            logits = task_model(**tokenized_sample).logits
            eval_prediciton = torch.argmax(logits, dim=1).cpu().numpy()

            inference_log = {}
            inference_log["input"] = entry["text"]
            inference_log["label"] = entry["label"]
            inference_log["judgment"] = eval_prediciton
            inference_log["original_input"] = entry["text"]
            inference_log["style prompt"] = ""
            inference_logs.append(inference_log)

    if not os.path.exists(f"results/{experiment_id}"):
        os.makedirs(f"results/{experiment_id}")

    save_inference_log(inference_logs, experiment_id, task_model_name, dataset_name, icl_method, eval_set, "fine_tuning", None)
    data_reader = DatasetReader(dataset, input_columns=["text"], output_column="label")
    original_judgments = [log["judgment"] for log in inference_logs]
    dataset_name = f"{dataset_name}-{eval_set}" if dataset_name.startswith("boss_") else dataset_name
    return generate_icl_report(experiment_id, task_model_name, dataset_name, icl_method, eval_set, dataset, data_reader, original_judgments, "Fine-Tuning")


class GenericDataset(Dataset):
    def __init__(self, in_dataset):
        self.dataset = in_dataset

    def __getitem__(self, index):
        return self.dataset["text"][index], self.dataset["label"][index]

    def __len__(self):
        return len(self.dataset["text"])
