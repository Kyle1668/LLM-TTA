from datetime import datetime
import pandas as pd
import argparse
import random
import numpy as np
import torch
import wandb
import time
import json
import os

from adaptive_methods import evaluate_test_time_augmentation, evaluate_style_transfer, evaluate_fine_tuning, evaluate_memo, evaluate_without_adaptation
from util_data import get_formatted_dataset, get_num_labels
from util_modeling import get_model_objects, is_large_language_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--icl_method", type=str, default=None)
    parser.add_argument("--temperature", type=str, default=None)
    parser.add_argument("--num_shots", type=str, default=None)
    parser.add_argument("--adaptive_model", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--skip_eval_styling", action="store_true")
    parser.add_argument("--skip_style_model_eval", action="store_true")
    parser.add_argument("--transfer_prompt", type=str, default="domain_transfer_no_aug_tasks_v4")
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create expeirment directory
    experiment_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.dataset}_{args.model.replace('/', '-')}"
    time.sleep(5)
    if not os.path.exists("results"):
        os.mkdir("results")
    os.mkdir(f"results/{experiment_id}")
    json.dump(vars(args), open(f"results/{experiment_id}/args.json", "w", encoding="utf-8"), indent=4)

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
    icl_methods = args.icl_method.split(",") if args.icl_method is not None else ["random", "topk_nearest"]
    domain_transfer_temperatures = [float(char) for char in args.temperature.split(",")] if args.temperature is not None else [0.0, 0.7]
    num_shots = [int(char) for char in args.num_shots.split(",")] if args.num_shots is not None else [16, 8]
    splits = args.splits.split(",") if args.splits is not None else None
    adaptive_model_names = (
        args.adaptive_model.split(",")
        if args.adaptive_model is not None
        else [
            "Salesforce/xgen-7b-8k-inst",
            "TheBloke/vicuna-7B-1.1-HF",
            # "TheBloke/vicuna-13B-1.1-HF",
            # "tiiuae/falcon-7b",
        ]
    )
    baselines = args.baseline.split(",") if args.baseline is not None else [] if args.baseline == "skip" else ["fine-tuning", "test_time_augmentation", "memo"]

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

    # Evalaute with baselines
    adaptive_methods = ["No Adaptation"] + [method for method in baselines if method != "skip"] + ([] if args.skip_eval_styling else adaptive_model_names)

    print("--------------------------------------------------")
    print("Running experiment with the following parameters:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset Names: {dataset_names}")
    print(f"ICL Methods: {icl_methods}")
    print(f"Task Model Names: {model_names}")
    print(f"Style Model Names: {adaptive_model_names}")
    print(f"Max Examples: {args.max_examples}")
    print(args)
    print("--------------------------------------------------\n")

    wandb_enabled = args.use_wandb
    wandb_run = None
    if wandb_enabled:
        wandb_config = {
            "dataset_names": dataset_names,
            "icl_methods": icl_methods,
            "task_model_names": model_names,
            "style_model_names": adaptive_model_names,
            "max_examples": args.max_examples,
            "baslines": baselines,
            "adaptive_methods": adaptive_methods,
        }
        project_name = "In-Context Domain Transfer Improves Out-of-Domain Robustness"
        wandb_run = wandb.init(project=project_name, name=experiment_id, config=wandb_config)

    reports = []
    for dataset_name in dataset_names:
        print(f"Loading dataset {dataset_name}...")
        dataset = get_formatted_dataset(dataset_name, max_examples=args.max_examples)
        splits = splits if splits is not None else [split for split in dataset.keys() if split != "train"]

        for model_name in model_names:
            print(f"Loading model {model_name}...")
            num_labels = get_num_labels(dataset_name)
            tokenizer, model = get_model_objects(model_name, num_labels)
            adaptive_tokenizer = adaptive_model = None
            is_llm = is_large_language_model(model_name)

            for evaluation_set in splits:
                for icl_method in icl_methods if is_llm else ["static"]:
                    # Evaluate style model on the task
                    if not args.skip_style_model_eval:
                        for adaptive_model_name in adaptive_model_names:
                            for style_icl_method in icl_methods:
                                for shots in num_shots:
                                    if adaptive_model is None:
                                        adaptive_tokenizer, adaptive_model = get_model_objects(adaptive_model_name, num_labels)

                                    current_report = evaluate_without_adaptation(
                                        experiment_id, adaptive_model_name, adaptive_model, adaptive_tokenizer, dataset_name, dataset, style_icl_method, evaluation_set, shots
                                    )
                                    reports.append(current_report)
                                    all_reports = pd.DataFrame(reports).drop_duplicates()
                                    print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                    if wandb_enabled:
                                        wandb.log(current_report)
                                        wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                                    adaptive_tokenizer = None
                                    adaptive_model = None

                    if evaluation_set not in ["validation"]:
                        for adaptive_method in adaptive_methods:
                            if adaptive_method == "No Adaptation":
                                # Evaluate the task model
                                current_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, evaluation_set, None)
                                reports.append(current_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                if wandb_enabled:
                                    wandb.log(current_report)
                                    wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                            elif adaptive_method == "test_time_augmentation":
                                for aug_method in ["paraphrase", "replace"]:
                                    tta_report = evaluate_test_time_augmentation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method, aug_method)
                                    reports.append(tta_report)
                                    all_reports = pd.DataFrame(reports).drop_duplicates()
                                    print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                    if wandb_enabled:
                                        wandb.log(tta_report)
                                        wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                            elif adaptive_method == "memo":
                                for aug_method in ["paraphrase", "replace"]:
                                    memo_report = evaluate_memo(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method, aug_method)

                                    reports.append(memo_report)
                                    all_reports = pd.DataFrame(reports).drop_duplicates()
                                    print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                    if wandb_enabled:
                                        wandb.log(memo_report)
                                        wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                                    # Now evaluate on the in-distribution set to assess potential catastrophic forgetting
                                    forgetting_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "validation")
                                    reports.append(forgetting_report)
                                    all_reports = pd.DataFrame(reports).drop_duplicates()
                                    print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                    all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                    if wandb_enabled:
                                        wandb.log(forgetting_report)
                                        wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                                    # Since MEMO updates the model's parameters, we need to reload the model so
                                    # as to not affect the next experiment
                                    tokenizer, model = get_model_objects(model_name, num_labels)
                            elif adaptive_method == "fine-tuning":
                                ft_report = evaluate_fine_tuning(experiment_id, model_name, model, tokenizer, dataset_name, dataset, evaluation_set, icl_method)
                                reports.append(ft_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                if wandb_enabled:
                                    wandb.log(ft_report)
                                    wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                                # Now evaluate on the in-distribution set to assess potential catastrophic forgetting
                                forgetting_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, "validation")
                                reports.append(forgetting_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                if wandb_enabled:
                                    wandb.log(forgetting_report)
                                    wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})

                                # Since fine-tuning the model further updates the model's parameters, we
                                # need to reload the model so as to not affect the next experiment
                                tokenizer, model = get_model_objects(model_name, num_labels, training=True)
                            else:
                                for style_icl_method in icl_methods:
                                    for shots in num_shots:
                                        print(f"Evaluating style transfer with {shots} shots")
                                        for trim_exemplars in [True]:
                                            for temperature in domain_transfer_temperatures:
                                                style_inference_log_frame, current_report = evaluate_style_transfer(
                                                    experiment_id,
                                                    model_name,
                                                    model,
                                                    tokenizer,
                                                    dataset_name,
                                                    dataset,
                                                    style_icl_method,
                                                    evaluation_set,
                                                    adaptive_method,
                                                    shots,
                                                    trim_exemplars,
                                                    temperature,
                                                    args.transfer_prompt,
                                                )
                                                reports.append(current_report)
                                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                                print(
                                                    all_reports[
                                                        ["dataset", "split", "task model", "icl_method", "exemplar count", "trim exemplars", "style transfer model", "dataset size", "accuracy", "avg f1"]
                                                    ]
                                                )
                                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                                if wandb_enabled:
                                                    wandb.log(current_report)
                                                    wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})
                                                    wandb_run.log({f"{evaluation_set}_{adaptive_method}_{style_icl_method}_style_logs": wandb.Table(dataframe=style_inference_log_frame)})
                    else:
                        if is_llm:
                            for num_shots in num_shots:
                                current_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, "static", evaluation_set, num_shots=num_shots)
                                reports.append(current_report)
                                all_reports = pd.DataFrame(reports).drop_duplicates()
                                print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                                all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                                if wandb_enabled:
                                    wandb.log(current_report)
                                    wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})
                        else:
                            current_report = evaluate_without_adaptation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, "static", evaluation_set)
                            reports.append(current_report)
                            all_reports = pd.DataFrame(reports).drop_duplicates()
                            print(all_reports[["dataset", "split", "task model", "icl_method", "exemplar count", "style transfer model", "dataset size", "accuracy", "avg f1"]])
                            all_reports.to_csv(f"results/{experiment_id}/reports.csv", index=False)
                            if wandb_enabled:
                                wandb.log(current_report)
                                wandb_run.log({"reports": wandb.Table(dataframe=all_reports)})


if __name__ == "__main__":
    main()
