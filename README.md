# Improving Black-box Robustness with In-Context Rewriting

This repo contains the code to replicate the results in [our paper](https://arxiv.org/abs/2402.08225) and extend our study to additional models and datasets. We use Make to abstract most of the commands to replicate our results. Don't hestiate to reach out to Kyle O'Brien with questions.

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

`Makefile` contains multiple other preset experiment configurations and shows examples of various command line arguments.

# Acknowledgments

We are grateful to EleutherAI for permitting access to their compute resources for initial experiments. The welcome and open research community on the EleutherAI Discord was especially helpful for the literature review, debugging PyTorch issues, and information necessary to conduct the parameter count ablation experiment (Appendix A.2). In particular, we would like to thank Stella Biderman, Nora Belrose, and Hailey Schoelkopf.

Lydia O’Brien provided copy editing and feedback on figure design and engaged in extensive discussions that
shaped the direction of this project.

M. Ghassemi’s work is supported in part by Quanta Computing and the Gordon and Betty Moore Foundation. The research of J. Mendez-Mendez is funded by an MIT-IBM Distinguished Postdoctoral Fellowship.