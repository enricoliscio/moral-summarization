import os

from .utils import *
from .data_utils import *
from .utils import *


vanilla_intro = 'You have to summarize the following article.'
moral_intro = 'You have to summarize the following text preserving the moral framing that the author gave to it.'
here_is_the_news_article = 'Here is the news article:\n\n'
cot_instructions = '(1) First, you identify all the single words that are morally framed. Identify this step as "STEP 1:" and report each word in a new line starting with *\n(2) Finally, you write a summary of the news article. '
preserve_moral_words = 'Please preserve as many morally framed words as possible in your summary. '
closing = 'The summary has to be returned after a "SUMMARY:" token and ending with a "END OF SUMMARY." token. The summary should be no longer than 200 words.'


def load_predictions_df(predictions_path):
    literal_eval_columns = ['predicted_words', 'labeled_words']
    converters = {column: literal_eval for column in literal_eval_columns}
    return pd.read_csv(predictions_path, converters=converters)


def get_clean_article_text(article_path):
    article_text = read_from_file(article_path)
    return article_text.replace('\n', ' ').replace('  ', ' ')


def make_article_prompt(article_text):
    return here_is_the_news_article + article_text


def load_moral_annotations(dataset, article_file):
    annotations_path = os.path.join(EMONA_dataset_path, annotation_folders[dataset], f'{article_file}.json')
    annotations = load_json(annotations_path)
    moral_annotations = get_moral_annotations(annotations)
    return [remove_punctuation(annotation['token']) for annotation in moral_annotations]


def load_predicted_moral_words(predictions_df, article_file):
    predicted_words = []
    # loop through the sentences of this article and get the predicted words
    for _, row in predictions_df[predictions_df['article'] == article_file].iterrows():
        predicted_words.extend(row['predicted_words'])

    return predicted_words


def make_moral_words_list(moral_words, deduplicate=True):
    intro = 'The author used the following morally framed words in the article:\n'

    if deduplicate:
        moral_words = list(dict.fromkeys(moral_words))

    # print the words in a list with a bullet point
    moral_list = '\n'.join([f'* {word}' for word in moral_words])

    return intro + moral_list


def dump_prompt(prompt, destination_folder, prompt_type):
    destination_file = os.path.join(destination_folder, f'{prompt_type}_prompt.txt')
    write_to_file(destination_file, prompt)


def dump_vanilla_prompt(article_prompt, destination_folder):
    prompt = vanilla_intro + '\n\n' + article_prompt + '\n\n' + closing
    dump_prompt(prompt, destination_folder, 'vanilla')


def dump_simple_prompt(article_prompt, destination_folder):
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + closing
    dump_prompt(prompt, destination_folder, 'simple')


def dump_cot_prompt(article_prompt, destination_folder):
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + cot_instructions \
          + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'cot')


def dump_oracle_prompt(article_prompt, destination_folder, dataset, article_file, deduplicate=True):
    moral_annotations = load_moral_annotations(dataset, article_file)
    moral_list = make_moral_words_list(moral_annotations, deduplicate)
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + moral_list + '\n\n' \
         + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'oracle')


def dump_class_prompt(article_prompt, destination_folder, predicions_df, article_file, deduplicate=True):
    # If the article is not in the test set, do not generate a prompt
    if article_file not in predicions_df['article'].to_list():
        return

    moral_predictions = load_predicted_moral_words(predicions_df, article_file)
    moral_list = make_moral_words_list(moral_predictions, deduplicate)
    prompt = moral_intro + article_prompt + '\n\n' + moral_list + '\n\n' \
         + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'class')


def dump_prompts(article_name, dataset, predicions_df, prompt_folder='results/test_prompts', deduplicate=True):
    article_path = os.path.join(EMONA_dataset_path, article_folders[dataset], f'{article_name}.txt')
    article_text = get_clean_article_text(article_path)
    article_prompt = make_article_prompt(article_text)

    destination_folder = os.path.join(prompt_folder, dataset, article_name)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dump_vanilla_prompt(article_prompt, destination_folder)
    dump_simple_prompt(article_prompt, destination_folder)
    dump_cot_prompt(article_prompt, destination_folder)
    dump_oracle_prompt(article_prompt, destination_folder, dataset, article_name, deduplicate)
    dump_class_prompt(article_prompt, destination_folder, predicions_df, article_name, deduplicate)
