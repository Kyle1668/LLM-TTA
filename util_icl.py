from faiss import IndexIDMap, IndexFlatIP
from openicl import PromptTemplate, TopkRetriever, MDLRetriever, RandomRetriever, VotekRetriever
from util_data import get_num_labels
import numpy as np
import json


def get_num_shots(dataset_name):
    dataset_ice_nums = {
        "sst2": 16,
        "adv_sst2": 4,
        "squad": 4,
        "ag_news": 6,
        "ag_news_twitter": 6,
        "toxigenic": 6,
        "disaster_tweets": 32,
        "wilds_civil_comments": 16,
        "civil_toxigen": 16,
        "rotten_tomatoes_imdb": 4,
        "imdb_rotten_tomatoes": 8,
        "wilds_amazon": 16,
        "boss_sentiment": 6,
        "boss_toxicity": 6,
        "boss_nli": 6,
        "scotus": 4
    }
    dataset_name = "squad" if dataset_name.startswith("squad") else dataset_name
    return dataset_ice_nums[dataset_name]


def get_retriever(icl_method, data, dataset_name, index_split="train"):
    icl_method = icl_method.replace("_furthest", "").replace("_centroid", "").replace("_nearest", "")
    if icl_method == "topk":
        return TopkRetriever(dataset_reader=data, ice_num=get_num_shots(dataset_name), index_split=index_split)
    elif icl_method == "mdl":
        return MDLRetriever(dataset_reader=data, ice_num=get_num_shots(dataset_name), index_split=index_split)
    elif icl_method == "random":
        return RandomRetriever(dataset_reader=data, ice_num=get_num_shots(dataset_name), index_split=index_split)
    elif icl_method == "votek":
        return VotekRetriever(dataset_reader=data, ice_num=get_num_shots(dataset_name), index_split=index_split)
    elif icl_method == "kne":
        return IndexIDMap(IndexFlatIP(768))
    elif icl_method == "static":
        return None
    else:
        raise Exception("Invalid ICL method")


def get_prompt_template(dataset_name):
    num_labels = get_num_labels(dataset_name)

    tp_dict = {}
    dataset_name = "squad" if dataset_name.startswith("squad") else dataset_name
    for i in range(num_labels):
        tp_dict[i] = f"\n</text> - Label={i}</E>"

    template = PromptTemplate(tp_dict, {"text": "</text>"}, ice_token="</E>")
    return template


def generate_qa_prompt(exemplars, input_entry):
    formatted_exemplars = [f"\nContext: {exemplar['text']}\nQuestion: {exemplar['question']}\nAnswer: {exemplar['label']}\n" for exemplar in exemplars]
    with open("prompts/squad.txt", encoding="utf-8") as f:
        prompt = f.read()
        prompt = prompt.replace("<exemplars>", "".join(formatted_exemplars))
        prompt = prompt.replace("<context>", input_entry["text"])
        prompt = prompt.replace("<question>", input_entry["question"])

    return prompt


def generate_classification_prompt(input_text, exemplars, template, dataset_name):
    if input_text is None:
        return None
    # style_transfer_exemplars = None
    #     if trim_exemplars:
    #         style_transfer_exemplars = "".join([f'- "{adaptive_tokenizer.decode(adaptive_tokenizer.encode(exemplar["text"].strip())[:int(1500 / len(exemplars))])}"\n' for exemplar in exemplars])
    #     else:
    #         style_transfer_exemplars = "".join(['- "' + exemplar["text"].strip().replace("\n", "") + '"\n' for exemplar in exemplars])

    input_text = input_text if isinstance(input_text, list) else [input_text]
    prompts = []
    for input in input_text:
        formatted_exemplars = []
        max_words_per_exemplar = 1600 // len(exemplars) if len(exemplars) > 0 else 0
        for i in range(len(exemplars)):
            if exemplars[i]["text"] == "" or exemplars[i]["text"] == None:
                continue
            formatted_exemplars.append(
                {"label": exemplars[i]["label"], "text": (" ".join(exemplars[i]["text"].split()[:max_words_per_exemplar]) if len(exemplars[i]["text"].split()) >= max_words_per_exemplar else exemplars[i]["text"]).replace("\n", " ").lstrip()}
            )

        instructions = json.load(open("prompts/instructions.json", encoding="utf-8"))[dataset_name]
        prompt_lines = [f"Task: {instructions}"]
        for exemplar in reversed(formatted_exemplars):
            exemplar_line = f'\n"{exemplar["text"].strip()}" - Label={exemplar["label"]}'
            prompt_lines.append(exemplar_line)

        # prompt_lines = [formatted_instructions] + ["\n" + template.generate_ice_item(entry, entry["label"]).replace("\n", " ").lstrip() for entry in reversed(formatted_exemplars)]
        prompt_lines.append('\nWhat is the label for the following text? You must decide which label the text is."')
        formatted_input_text = " ".join(input.split()[:500]) if len(input.split()) >= 500 else input

        prompt_lines.append('\n"' + formatted_input_text.replace("\n", " ").strip() + '" - Label=')
        prompt = "\n".join(prompt_lines).replace("</s>", " ")
        prompts.append(prompt)

    return prompts


def generate_prompt(model_name, template, exemplars, input_entry, dataset_name):
    prompt = None
    if dataset_name.startswith("squad"):
        prompt = generate_qa_prompt(exemplars, input_entry)
    else:
        prompt = generate_classification_prompt(input_entry["text"], exemplars, template, dataset_name)

    supported_chat_prompts = {"TheBloke/vicuna-13B-1.1-HF": f"User: {prompt}\nAssistant:", "TheBloke/vicuna-7B-1.1-HF": f"User: {prompt}\nAssistant:"}
    return supported_chat_prompts[model_name] if model_name in supported_chat_prompts else prompt


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


def get_static_exemplars(dataset_name, num_shots):
    exemplars_file_path = "prompts/exemplars.json"
    exemplars_file = json.load(open(exemplars_file_path, encoding="utf-8"))
    return exemplars_file[dataset_name][:num_shots]


def get_dynamic_exemplars(input_text, dataset_name, exemplar_retriever, exemplar_count=None, distance_goal="NA"):
    exemplar_count = get_num_shots(dataset_name) if exemplar_count is None else exemplar_count
    exemplar_distances = exemplar_indices = None
    exemplar_indices = None
    retriever_response = exemplar_retriever.get_exemplars(input_text, exemplar_count, distance_goal)
    if isinstance(retriever_response, tuple):
        exemplar_distances, exemplar_indices = retriever_response
    else:
        exemplar_indices = retriever_response

    mean_distance = np.mean(exemplar_distances[0]) if exemplar_distances is not None else None
    return ([exemplar_retriever.dataset_reader.dataset["train"][int(index)] for index in exemplar_indices[0]], mean_distance)
