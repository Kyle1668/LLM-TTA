from argparse import ArgumentParser
from datetime import datetime
import pandas as pd

from adaptive_methods import (
    evaluate_test_time_augmentation,
    evaluate_style_transfer,
    evaluate_fine_tuning,
    evaluate_memo,
    evaluate_without_adaptation
)
from util_data import get_formatted_dataset
from util_modeling import get_model_objects


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
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
    icl_methods = args.icl_method.split(",") if args.icl_method is not None else ["static", "topk", "mdl"]
    splits = args.splits.split(",") if args.splits is not None else None
    adaptive_model_names = (
        args.adaptive_model.split(",")
        if args.adaptive_model is not None
        else [
            "TheBloke/vicuna-13B-1.1-HF",
            "TheBloke/vicuna-7B-1.1-HF",
            # "tiiuae/falcon-40b-instruct",
            # "tiiuae/falcon-7b-instruct",
        ]
    )
    baselines = args.baseline.split(",") if args.baseline is not None else ["test_time_augmentation", "fine-tuning", "memo"]
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
    model_names = model_names + adaptive_model_names
    adaptive_methods = ["No Adaptation"] + baselines + adaptive_model_names
    # adaptive_methods = ["No Adaptation"] + adaptive_model_names

    print("--------------------------------------------------")
    print("Running experiment with the following parameters:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset Names: {dataset_names}")
    # print(f"Evalaution Splits: {splits}")
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
            splits = splits if splits is not None else [split for split in dataset.keys() if split != "train"]

            for icl_method in icl_methods:
                for evaluation_set in splits:
                    if evaluation_set not in ["validation"]:
                        for adaptive_method in adaptive_methods:
                            if adaptive_method == "No Adaptation":
                                reports.append(evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, None))
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                            elif adaptive_method == "test_time_augmentation":
                                tta_report = evaluate_test_time_augmentation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method)
                                reports.append(tta_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                            elif adaptive_method == "memo":
                                memo_report = evaluate_memo(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method)
                                reports.append(memo_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])

                                # Now evaluate on the in-distribution set to assess potential catastrophic forgetting
                                forgetting_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "validation")
                                reports.append(forgetting_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])

                                # Since MEMO updates the model's parameters, we need to reload the model so
                                # as to not affect the next experiment
                                tokenizer, model = get_model_objects(model_name)
                            elif adaptive_method == "fine-tuning":
                                # dataset_name = f"{dataset_name}_{evaluation_set}" if dataset_name.startswith("boss_") else dataset_name
                                ft_report = evaluate_fine_tuning(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method)
                                reports.append(ft_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])

                                # Now evaluate on the in-distribution set to assess potential catastrophic forgetting
                                forgetting_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "validation")
                                reports.append(forgetting_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])

                                # Since fine-tuning the model further updates the model's parameters, we
                                # need to reload the model so as to not affect the next experiment
                                tokenizer, model = get_model_objects(model_name)
                            else:
                                for num_shots in [4]:
                                    reports.append(evaluate_style_transfer(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, adaptive_method, num_shots))
                                    all_reports = pd.DataFrame(reports).drop_duplicates()
                                    print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])
                                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                    else:
                        is_llm = model.config.architectures[0].endswith("ForCausalLM")
                        if is_llm:
                            for num_shots in [4]:
                                reports.append(evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, num_shots=num_shots))
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                        else:
                            reports.append(evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set))
                            all_reports = pd.DataFrame(reports).drop_duplicates()
                            print(all_reports[["dataset", "split", "dataset size", "accuracy", "avg f1"]])
                            all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)


if __name__ == "__main__":
    main()
