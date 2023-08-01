
import os
import torch
import argparse
import torch.nn.functional as F
import nlpaug.augmenter.word as naw
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
tqdm.pandas()

from util_modeling import get_model_objects
from util_data import get_formatted_dataset
from adaptive_methods import get_paraphrase_augmentations


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(tokenizer, model, left_text, right_text):
    encoded_input = tokenizer(left_text, right_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])


def get_cosine_similarity(sentence_tokenizer, sentence_model, left_text, right_text):
    left_embedding = get_embeddings(sentence_tokenizer, sentence_model, left_text, right_text)
    right_embedding = get_embeddings(sentence_tokenizer, sentence_model, right_text, left_text)
    return F.cosine_similarity(left_embedding, right_embedding).item()


def get_augmentation(paraphrase_tokenizer, paraphrase_model, sentence_tokenizer, sentence_model, word_augmenter, random_deleter, current_text):
    paraphrases = get_paraphrase_augmentations(
        current_text,
        paraphrase_tokenizer,
        paraphrase_model,
        paraphrase_model.device,
        temperature=1.0,
        repetition_penalty=100.0,
        diversity_penalty=100.0,
        no_repeat_ngram_size=10)

    corrupted_paraphrases = [word_augmenter.augment(random_deleter.augment(aug)) for aug in paraphrases]
    corrupted_paraphrases = [aug[0] for aug in corrupted_paraphrases if len(aug) > 0]
    corrupted_cosines = [get_cosine_similarity(sentence_tokenizer, sentence_model, current_text, aug) for aug in corrupted_paraphrases]
    corrupted_aug_cosine_pairs = list(zip(corrupted_paraphrases, corrupted_cosines))
    most_corrupted = max(enumerate(corrupted_aug_cosine_pairs), key=lambda x: x[1])[1][0]
    return most_corrupted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    formatted_dataset = get_formatted_dataset(args.dataset, args.max_examples)["train"].to_pandas()
    formatted_dataset.rename(columns={"label": "class","text": "label"}, inplace=True)

    paraphrase_tokenizer, paraphrase_model = get_model_objects("humarin/chatgpt_paraphraser_on_T5_base", num_labels=-1)
    hf_model_path = "sentence-transformers/all-mpnet-base-v2"
    sentence_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    sentence_model = AutoModel.from_pretrained(hf_model_path)
    word_replacer = naw.ContextualWordEmbsAug(device="cuda", action="substitute")
    random_deleter = naw.RandomWordAug(action="delete", aug_p=0.30)

    formatted_dataset["text"] = formatted_dataset.progress_apply(lambda row: get_augmentation(
        paraphrase_tokenizer,
        paraphrase_model,
        sentence_tokenizer,
        sentence_model,
        word_replacer,
        random_deleter,
        row["label"]), axis=1)

    print(formatted_dataset.head())
    corruped_datasets_path = "./datasets/corruped"
    if not os.path.exists(corruped_datasets_path):
        os.makedirs(corruped_datasets_path)

    formatted_dataset = formatted_dataset[["text", "label", "class"]]
    formatted_dataset.to_csv(f"{corruped_datasets_path}/{args.dataset}{args.max_examples if args.max_examples is not None else ''}.csv", index=False)


if __name__ == "__main__":
    main()
