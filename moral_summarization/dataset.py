import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from datasets import Dataset, DatasetDict

from moral_summarization.data_utils import *
from moral_summarization.utils import get_tokenizer


# Map the moral labels to 1 and non-moral to 0
moral2label = {label: 1 for label in moral_labels}
moral2label['non-moral'] = 0
moral2label['non-event'] = 0


def has_moral_annotations(annotation):
    if set(moral_labels).intersection(set(annotation['tokens_label_list'])):
        return True
    else:
        return False


def make_sentence_df(task, annotation, dataset, article_name):
    if task == 'sequence_classification':
        return pd.DataFrame({
            'dataset' : [dataset],
            'article': [article_name],
            'sentence_id': [annotation['sentence_id']],
            'text': [annotation['sentence_text']],
            'label': [1 if has_moral_annotations(annotation) else 0],
        })
    elif task == 'token_classification':
        return pd.DataFrame({
            'dataset' : [dataset],
            'article': [article_name],
            'sentence_id': [annotation['sentence_id']],
            'tokens': [annotation['tokens_list']],
            'labels': [[moral2label[ann] for ann in annotation['tokens_label_list']]],
        })
    

def extend_sentences_df(task, annotations, dataset, article_name, sentences_df):
    if task == 'sequence_classification' or task == 'token_classification':
        for annotation in annotations['sentences']:
            # Skip sentences with no annotations for token classification
            #if task == 'token_classification' and not has_moral_annotations(annotation):
            #    continue

            sentence_df = make_sentence_df(task, annotation, dataset, article_name)
            sentences_df = pd.concat([sentences_df, sentence_df], ignore_index=True)

    elif task == 'article_token_classification':
        article_tokens, article_labels = [], []
        for annotation in annotations['sentences']:
            article_tokens += annotation['tokens_list']
            article_labels += [moral2label[ann] for ann in annotation['tokens_label_list']]
    
        # if len(article_tokens) > 2000:
        #     return sentences_df

        article_tokens_df = pd.DataFrame({
            'dataset' : [dataset],
            'article': [article_name],
            'tokens': [article_tokens],
            'labels': [article_labels],
        })
        sentences_df = pd.concat([sentences_df, article_tokens_df], ignore_index=True)

    return sentences_df


def get_article_type(name):
    split_art = name.split('_')
    if split_art[0] == 'basil':
        return '_'.join([split_art[0], split_art[-1]])
    else:
        return '_'.join(split_art[:-1]) 


def make_article_df(article_name):
    article_type = get_article_type(article_name)
    return pd.DataFrame({
        'article': [article_name],
        'type': [article_type]
    })


def get_annotations_df(task, size):
    if task == 'sequence_classification':
        sentences_df = pd.DataFrame(columns=['dataset', 'article', 'sentence_id', 'text', 'label'])
    elif task == 'token_classification':
        sentences_df = pd.DataFrame(columns=['dataset', 'article', 'sentence_id', 'tokens', 'labels'])
    elif task == 'article_token_classification':
        sentences_df = pd.DataFrame(columns=['dataset', 'article', 'tokens', 'labels'])
    articles_df = pd.DataFrame(columns=['article', 'type'])

    for dataset in EMONA_datasets:
        dataset_path = os.path.join(EMONA_dataset_path, article_folders[dataset])
        article_names = [os.path.splitext(f)[0] for f in os.listdir(dataset_path)]
        for article_name in article_names:
            article_df = make_article_df(article_name)
            articles_df = pd.concat([articles_df, article_df], ignore_index=True)
            annotations = load_annotations(article_name, dataset)
            sentences_df = extend_sentences_df(task, annotations, dataset, article_name, sentences_df)

    return sentences_df[:size], articles_df[:size]


def split_dataset(sentences_df, articles_df, train_size=0.85, test_size=0.15, val_size=40, random_state=235):
    """
    Split the dataset into train, validation, and test sets,
    stratifying by the article type.
    """
    df_train, df_test = train_test_split(articles_df, train_size=train_size, test_size=test_size, random_state=random_state, stratify=articles_df['type'])
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=random_state, stratify=df_train['type'])

    train_sent_df = sentences_df.loc[sentences_df['article'].isin(df_train['article'])]
    val_sent_df = sentences_df.loc[sentences_df['article'].isin(df_val['article'])]
    test_sent_df = sentences_df.loc[sentences_df['article'].isin(df_test['article'])]

    return train_sent_df.sample(frac=1), val_sent_df, test_sent_df


def make_Dataset(task='sequence_classification', size=-1, join_train_val=False):
    # Get the annotations and articles DataFrames
    sentences_df, articles_df = get_annotations_df(task, size)

    # Split the dataset into train, validation, and test sets
    dataset_train, dataset_val, dataset_test = split_dataset(sentences_df, articles_df)

    if join_train_val:
        dataset_train = pd.concat([dataset_train, dataset_val], ignore_index=True)

    # Combine them into a single DatasetDict
    return DatasetDict({
        'train': Dataset.from_pandas(dataset_train.reset_index(drop=True)),
        'val': Dataset.from_pandas(dataset_val.reset_index(drop=True)),
        'test': Dataset.from_pandas(dataset_test.reset_index(drop=True))
    })


def tokenize_dataset(dataset, tokenizer, task, max_length=4096, label_all_tokens=True):
    def tokenize_sequence_classification(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)

    def tokenize_token_classification(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'],
            truncation=True,
            max_length=max_length,
            is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f'labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    if task == 'sequence_classification':
        tokenized_dataset = dataset.map(tokenize_sequence_classification, batched=True)
    elif task == 'token_classification' or task == 'article_token_classification':
        tokenized_dataset = dataset.map(tokenize_token_classification, batched=True)

    tokenized_dataset.set_format('torch')

    return tokenized_dataset


def get_dataset(config, max_length=4096, label_all_tokens=True, join_train_val=False):
    dataset = make_Dataset(config['task'], config['dataset_size'], join_train_val=join_train_val)

    model_name = config['pretrained_model_name'] if 'pretrained_model_name' in config else config['base_model_name']
    model_path = os.path.join(config['hf_model_folder'], model_name)
    tokenizer = get_tokenizer(model_path)

    return tokenize_dataset(dataset, tokenizer, config['task'], max_length, label_all_tokens)


def get_folded_dataset(config, max_length=4096, label_all_tokens=True, n_folds=5):
    model_name = config['pretrained_model_name'] if 'pretrained_model_name' in config else config['base_model_name']
    model_path = os.path.join(config['hf_model_folder'], model_name)
    tokenizer = get_tokenizer(model_path)

    # Get the annotations and articles DataFrames
    sentences_df, articles_df = get_annotations_df(config['task'], config['dataset_size'])

    df_train, df_test = train_test_split(articles_df, train_size=0.85, test_size=0.15, random_state=235, stratify=articles_df['type'])

    train_sent_df = sentences_df.loc[sentences_df['article'].isin(df_train['article'])]
    test_sent_df = sentences_df.loc[sentences_df['article'].isin(df_test['article'])]

    # Split the train set into folds
    kf = KFold(n_splits=n_folds, random_state=235, shuffle=True)
    folded_dataset = []

    for train_index, val_index in kf.split(train_sent_df):
        split_train_set_df = train_sent_df.iloc[train_index]
        split_val_sent_df = train_sent_df.iloc[val_index]

        # Combine them into a single DatasetDict
        dataset = DatasetDict({
            'train': Dataset.from_pandas(split_train_set_df.reset_index(drop=True)),
            'val': Dataset.from_pandas(split_val_sent_df.reset_index(drop=True)),
            'test': Dataset.from_pandas(test_sent_df.reset_index(drop=True))
        })

        dataset = tokenize_dataset(dataset, tokenizer, config['task'], max_length, label_all_tokens)
        folded_dataset.append(dataset)

    return folded_dataset
