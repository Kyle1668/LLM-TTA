# Introduction

**WIP: This codebase is under development for an ongoing research project.**

This codebase evaluates the effectiveness of pairing a task-specific model with an LLM for improved distributional robustness. In this setup, the LLM rewrites an OOD example into the style of the original distribution, which the task model was trained on. This technique can improve performance depending on the task and the selected models. This codebase provides the pipeline for almost any HuggingFace BERT or T5 model with an LLM for rewriting.

## Commands
`--seed`: Set the random seed for reproducibility.

`--dataset`: The HuggingFace or custom supported dataset to evaluate on.

`--model`: The HuggingFace `AutoModelForSequenceClassification` model trained on the original distribution.

`--splits`: The splits of the dataset to evaluate on. If not specified, all splits will be evaluated.

`--baseline`: The baselines to evaluate. If not specified, all baselines will be evaluated. Supported baselines are: `fine-tuning`, `test_time_augmentation`, `memo`, `skip`.

`--icl_method`: The ICL method to use for style transfer. If not specified, all ICL methods will be evaluated. Supported ICL methods are: `random`, `topk_nearest`.

`--temperature`: The temperature to use for style transfer. If not specified, 0.0 and 0.7 are evaluated.

`--num_shots`: The number of shots to use for style transfer. If not specified, 32, 16, and 8 are evaluated.

`--adaptive_model`: The HuggingFace LLM that will rewrite each example. Multiple models can be specified by separating them with a comma.

`--max_examples`: The maximum number of examples to evaluate on. If not specified, all examples will be evaluated.

`--use_wandb`: Whether to use wandb to log results.

`--skip_eval_styling`: Whether to skip evaluating the style transfer methods.

`--skip_style_model_eval`: Whether to skip evaluating the style transfer models.

`--evaluate_id_adaptation`: Whether to evaluate the style transfer models on the in-dsitribution test set.

`--transfer_prompt`: The transfer prompt to use for style transfer.

## Quick Start

In this example, we will determine how well XGen-7b is at rewriting inputs for BERT in an IMDB --> Rotten Tomatoes shift. Here we're also evaluating performance across different numbers of in-context examples and the temperature of the rewriting model.

`python evaluate_styling.py --use_wandb --dataset=imdb_rotten_tomatoes --model=lvwerra/bert-imdb --adaptive_model=Salesforce/xgen-7b-8k-inst --temperature=0.0,0.3,0.7 --num_shots=32,8 --skip_style_model_eval --baselines=skip`

You can also evaluate how well the rewriting LLM and baseline methods perform on the task by committing the `--skip_style_model_eval` and  `--baselines=skip` flags. The results will be reported in the `./results` directory.

## Baseline Methods

### Supervised Fine-Tuning

We train the task model on the entire labeled OOD evaluation dataset for a single epoch and then evaluate it. The model will have thus seen each OOD example once and updated its gradient based on the supervised loss. This method can be thought of as an upper bound on domain generalization at the expense of catastrophic forgetting and the requirement for labeled examples. We measure catastrophic forgetting, the extent that performance on the in-distribution evaluation set regresses with the use of parametric adaption techniques, in the appendix.

### Test-Time Augmentation

We follow the approach suggested in (Lu et al., 2022), where the final prediction is derived from the mean probability distribution of the task model’s inference on the original input as well as a set of augmentations of the input. We evaluate this technique separately using two different augmentation strategies. We first use the suggested word replacement, where a single word is replaced with a synonym. We also evaluate using full-text paraphrases from a T5 model fine-tuned on ChatGPT outputs (Vladimir Vorobev, 2023). We used four augmented examples and report
the results using both augmentation methods
separately.

### Parametric Unsupervised Test-Time Adaptation (MEMO)
A representative method for unsupervised domain adaptation
where the source model’s weights are unfrozen is MEMO (Zhang et al., 2021). This method uses the entropy of the mean probability distribution across augmentations of the current input as an unsupervised loss signal. For each text example, derive the entropy of the mean predicted class probability distribution for a set of augmentations on the input. The entropy is then used as a loss to backpropagate through the network. After performing this gradient step, make the final prediction on the original test input. Intuition suggests that minimizing entropy leads to more confident predictions, which often improves performance. We use the same two augmentation strategies from (Lu et al., 2022) and (Vladimir Vorobev, 2023) described previously.
