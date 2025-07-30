import torch

from moral_summarization.model import (
    LlamaModelForSequenceClassification,
    LlamaModelForTokenClassification
)
from moral_summarization.args import load_config
from moral_summarization.dataset import get_dataset


# Parse command line arguments and config file
config = load_config(training=True)

torch.manual_seed(0)

if config['task'] == 'sequence_classification':
    llama_model = LlamaModelForSequenceClassification(config)
elif config['task'] == 'token_classification':
    llama_model = LlamaModelForTokenClassification(config)

dataset = get_dataset(config, max_length=512, join_train_val=True)

llama_model.train(dataset)

llama_model.predict_on_dataset(dataset['test'], max_length=512)

llama_model.save_model()