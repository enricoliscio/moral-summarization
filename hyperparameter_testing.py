import torch
import os

from moral_summarization.model import (
    LlamaModelForSequenceClassification,
    LlamaModelForTokenClassification
)
from moral_summarization.args import load_config, get_config_combinations
from moral_summarization.dataset import get_folded_dataset
from moral_summarization.utils import dump_yaml


# Parse command line arguments and config file
config = load_config(training=True)

torch.manual_seed(0)

# Define the hyperparameters to be tuned
hyperparameters = {
    'lora.r': [32, 64],
    'training.learning_rate': [0.0002, 0.0005],
    'training.per_device_train_batch_size': [8, 16],
    'training.num_train_epochs': [3, 4],
}

configs = get_config_combinations(config, hyperparameters)

# make folder for saving all the iterations
if not os.path.exists(config['training']['output_dir']):
    os.makedirs(config['training']['output_dir'])

print(f"Testing {len(configs)} configurations")

# Train and evaluate the model for each configuration
for i, config in enumerate(configs):
    print("Testing configuration", i)

    # Make a folder for saving the output of this iteration
    iteration_output_dir = os.path.join(config['training']['output_dir'], str(i))
    if not os.path.exists(iteration_output_dir):
        os.makedirs(iteration_output_dir)

    # Dump this iteration of the config file to yaml
    dump_yaml(os.path.join(iteration_output_dir, 'config.yaml'), config)

    config['training']['output_dir'] = iteration_output_dir

    if config['task'] == 'article_token_classification':
        max_length = 1866
    else:
        max_length = 512

    for i, dataset in enumerate(get_folded_dataset(config, n_folds=3, max_length=max_length)):
        if config['task'] == 'sequence_classification':
            llama_model = LlamaModelForSequenceClassification(config)
        elif config['task'] == 'token_classification' or config['task'] == 'article_token_classification':
            llama_model = LlamaModelForTokenClassification(config)

        llama_model.train(dataset)
        llama_model.predict_on_dataset(
            dataset['val'],
            predictions_file=f'predictions_fold_{i}.csv',
            metrics_file=f'metrics_fold_{i}.json',
            max_length=max_length
        )
