# Introduction

## WIP: This codebase is under development for an ongoing research project.

This codebases evaluates the effectivness of pairing a task-specific model with a LLM for improved distributiional robustness. In this setup, the LLM rewrites an OOD example into thr style of the original distribution which the task model was trained on. This techique can improve performance depending on the task and thr selected models. This codebase provides the pipeline for almost any HuggingFace BERT or T5 model with an LLM for rewriting.

## Quick Start

To 