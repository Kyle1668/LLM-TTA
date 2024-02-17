# Improving Black-box Robustness with In-Context Rewriting

This repo contains the code to replicate the results in [our paper](https://arxiv.org/abs/2402.08225) and extend our study to additional models and datasets. We use Make to abstract most of the commands to replicate our results.

## Installation

We've tested our experiments using Python 3.10. Conda is recommended. 

```bash
conda create -n llm-tta python=3.10
conda activate llm-tta
```

You can install the pip packages and datasets using the following command.

```bash
make install_depends
```

Generating augmentation is slow. We use a caching mechanism to leverage previously generated augmentations for each test input to speed up experiments. You can fill the cache using the following command. The cache wil otherwise automatically begin populating otherwise. 

```bash
make download_rewrites_cache
```

And clear the cache with:
```bash
make clear_rewrites_cache
```

## Run Experiments

You can iterate through the main results using the following commands.

```bash
make main_results_async # split across all GPUs (Recommended)
make main_results_sync # Single-GPU setup
```

`Makefile` contains multiple other preset experiment configurations and shows examples of various command line arguments. # Improving Black-box Robustness with In-Context Rewriting\\

## 