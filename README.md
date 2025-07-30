# Moral Summarization

Code for the *News is More than a Collection of Facts: Moral Frame Preserving News Summarization* COLM2025 paper. Currently available as a [pre-print](https://arxiv.org/abs/2504.00657).

## Start-up

1. Install the package in developer mode `pip install -e .`. This will also install the dependencies.
1. Clone the [EMONA dataset](https://github.com/yuanyuanlei-nlp/EMONA_dataset) in the root folder.
1. Download the models (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) to your preferred destination. The location of the models can be passed as argument to the scripts.

## Important note

The naming convention of the prompting methods is different from the one used in the paper. Here is the mapping:
- vanilla --> Plain
- simple --> Direct
- cot --> CoT
- oracle --> Oracle
- class --> Class

## Usage

1. You can train the moral-laden classifier with `python training.py --hf-model-folder <location-of-models> --config-file token_labeling.yaml`.
1. You can perform hyperparameters search and cross-validation with `python hyperparameter_testing.py --hf-model-folder <location-of-models> --config-file token_labeling.yaml`.
1. You can create the prompts to generate the summaries with `python generate_prompts.py --prompt-dir <destination-dir>`
1. You can generate the summaries (with the prompts generated above) with `python prompting.py --config-file peft_config.yaml --hf-model-folder <location-of-models> --prompt-dir <destination-dir>`
1. You can run the automated evaluations with `python evaluate.py --llama --QaFactEval --results-dir <destination-dir>`.