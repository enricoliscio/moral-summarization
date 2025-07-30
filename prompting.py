import os
import torch
from tqdm import tqdm

from moral_summarization.utils import *
from moral_summarization.args import load_config
from moral_summarization.model import LlamaModelForSequenceCompletion


# Parse command line arguments and config file
config = load_config(inference=True)

torch.manual_seed(0)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_model = LlamaModelForSequenceCompletion(model_name, config)

for dataset_folder in os.listdir(config['inference']['prompt_dir']):
    dataset_path = os.path.join(config['inference']['prompt_dir'], dataset_folder)
    for article_folder in tqdm(os.listdir(dataset_path), desc=f"Generating responses for {dataset_folder}"):
        article_path = os.path.join(dataset_path, article_folder)
        for file_path in os.listdir(article_path):
            prompt_path = os.path.join(article_path, file_path)
            if 'prompt' in file_path and os.path.isfile(prompt_path):
                prompt = read_from_file(prompt_path)

                if 'vanilla' in file_path:
                    system_content = "You are a news summarizer assistant."
                else:
                    system_content = "You are a news summarizer assistant and a moral expert."

                response, conversation = llama_model.get_response(prompt, system_content)

                response_path = os.path.join(article_path, file_path.replace('prompt', 'response'))
                write_to_file(response_path, response[-1]['content'])
                if config['verbose']:
                    print(f"Generated response for {article_folder}/{file_path}")

        if config['inference']['testing']:
            break
