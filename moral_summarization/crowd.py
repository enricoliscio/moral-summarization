import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.stats import wilcoxon, spearmanr

from moral_summarization.utils import load_pickle
from moral_summarization.data_utils import load_annotations


methods = ['vanilla', 'simple', 'cot', 'oracle', 'class']


def get_article_scores(scores_df):
    methods_means = [f'{method}_mean' for method in methods]
    article_scores = scores_df[methods_means]
    article_scores = article_scores.groupby('article').mean()
    article_scores.columns = methods

    return article_scores


def print_means(scores_df):
    for method in methods:
        print(f"{method}: {scores_df[method].mean():.2f}")
        print("=====================================")


def print_std(scores_df):
    for method in methods:
        print(f"{method}: {scores_df[method].std():.2f}")
        print("=====================================")


def get_article_counts(results):
    unique1, counts1 = np.unique(results['summid1'].values, return_counts=True)
    unique2, counts2 = np.unique(results['summid2'].values, return_counts=True)

    # concatenate the two lists
    article_names = np.concatenate((unique1, unique2))
    article_counts = np.concatenate((counts1, counts2))

    counts = pd.DataFrame({'article': article_names, 'count': article_counts})
    counts.set_index('article', inplace=True)

    return counts


def dump_latex_scores(scores, path, index=False):
    scores.to_csv(path, sep=' ', index=index)


def get_wilcoxon_from_df_columns(df, column_1, column_2):
    a = df[column_1].to_numpy().astype(float)
    b = df[column_2].to_numpy().astype(float)
    return round(wilcoxon(a, b).pvalue, 3)


def pairwise_wilcoxon(scores):
    wilcoxon_matrix = np.zeros((5, 5))
    for i, j in combinations(range(5), 2):
        wilcoxon_matrix[i, j] = get_wilcoxon_from_df_columns(scores, methods[i], methods[j])
        wilcoxon_matrix[j, i] = wilcoxon_matrix[i, j]

    return pd.DataFrame(wilcoxon_matrix, index=methods, columns=methods)


def get_scores_per_highlights(scores_df, length_ranges):
    scores_groups = pd.DataFrame(index=methods, columns=range(len(length_ranges)))
    scores_groups = scores_groups.map(lambda x: [])
    for index, row in scores_df.iterrows():
        for method in methods:
            length = len(scores_df.loc[index, method])
            for group, (min_length, max_length) in length_ranges.items():
                if min_length <= length < max_length:
                    scores_groups.loc[method, group].append(row[f'{method}_mean'])

    # add a row at the bottom with the length of the lists
    for method in methods:
        for group in scores_groups.columns:
            scores_groups.loc['length', group] = len(scores_groups.loc[method, group])

    means_df = scores_groups.copy()
    means_df = means_df.drop('length')

    for method in methods:
        for group in scores_groups.columns:
            means_df.loc['length', group] = len(scores_groups.loc[method, group])
            means_df.loc[method, group] = round(np.mean(means_df.loc[method, group]), 2)

    return means_df


def plot_scores_per_highlight(scores, length_ranges):
    for method in methods:
       plt.plot(scores.columns, scores.loc[method, :])
    plt.legend(methods)
    plt.xticks(scores.columns, [f'{length_ranges[i][0]}-{length_ranges[i][1]}' for i in scores.columns])
    plt.show()


def get_spearman_correlation(file_path, article_scores):
    automated_results = load_pickle(file_path)

    spearman_df = pd.DataFrame(columns=methods, index=automated_results.keys())

    for metric, results_df in automated_results.items():
        results = results_df.loc['Meta-Llama-3-70B-Instruct'].loc[['allsides', 'mpqa', 'basil']]
        results = results.droplevel(0)

        # sort moral_div and article_scores by index
        results = results.sort_index()
        article_scores = article_scores.sort_index()

        for method in methods:
            df = pd.DataFrame(index=results.index, columns=[metric, 'scores'])
            df[metric] = results[method]
            df['scores'] = article_scores[method]
            df = df.dropna()

            a = df[metric].to_numpy().astype(float)
            b = df['scores'].to_numpy().astype(float)
            spearman_df.loc[metric, method] = round(spearmanr(a, b).statistic, 2)

    return spearman_df


def create_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def get_indices_overlapping_sentences(highlight, sentences, ngram_size=3):
    # Split the highlight into words
    highlight_words = re.findall(r'\w+', highlight)

    # Create trigrams (3-grams) from the highlight words
    if len(highlight_words) < ngram_size:
        highlight_trigrams = highlight
    else:
        highlight_trigrams = create_ngrams(highlight_words, ngram_size)

    # Find all occurrences of the highlight trigrams in each sentence
    overlapping_sentences_indices = set()
    for idx, sentence in enumerate(sentences):
        for trigram in highlight_trigrams:
            if re.search(re.escape(trigram), sentence):
                overlapping_sentences_indices.add(idx)
                break  # No need to check further trigrams for this sentence

    overlapping_sentences_indices = list(overlapping_sentences_indices)
    if len(highlight_words) < ngram_size and len(overlapping_sentences_indices) > 1:
        return [overlapping_sentences_indices[0]]
    else:
        return overlapping_sentences_indices


def get_quoted_parts_in_highlight(highlight, sentences):
    # Merge sentences into one text chunk
    text = ' '.join(sentences)

    # Regex to find quoted spans (straight and curly quotes)
    quote_regex = r'(["“”])(.*?)(\1)'

    quoted_parts = []
    highlight_start = text.find(highlight)

    if highlight_start == -1:
        return []  # If the highlight is not found, return an empty list

    highlight_end = highlight_start + len(highlight)

    # Search for quoted spans
    for match in re.finditer(quote_regex, text):
        quote_start, quote_end = match.span()

        # Check if the quoted span overlaps with the highlight span
        if not (highlight_end <= quote_start or highlight_start >= quote_end):
            # Find the intersection between the highlight and the quoted span
            overlap_start = max(highlight_start, quote_start)
            overlap_end = min(highlight_end, quote_end)
            overlap_text = text[overlap_start:overlap_end]

            if overlap_text.strip():
                quoted_parts.append(overlap_text)

    return quoted_parts


def filter_scores_preserve_moral_words(scores_df, results_class, summaries, filter_quotes=False):
    filtered_scores = pd.DataFrame(index=scores_df.index, columns=methods)
    for index, row in filtered_scores.iterrows():
        for method in methods:
            filtered_scores.loc[index, method] = []

    if filter_quotes:
        quotes_with_highlights = pd.DataFrame(index=scores_df.index, columns=['highlights_with_quotes'])
        for index, row in quotes_with_highlights.iterrows():
            quotes_with_highlights.loc[index, 'highlights_with_quotes'] = []

    for index, row in scores_df.iterrows():
        prolific_id = index[0]
        article = index[1]

        article_pred = results_class[results_class['article'] == article]
        annotations = load_annotations(article, article.split('_')[0])
        sentences = [sentence['sentence_text'] for sentence in annotations['sentences']]

        # for each highlight in row['highlights'], check with which sentence it overlaps
        for highlight_id, highlight in enumerate(row['highlights']):
            ids_with_highlight = get_indices_overlapping_sentences(highlight, sentences)
            if len(ids_with_highlight) == 0: # happens only a couple of times due to some qutoe marks or hyphens
                continue

            # select subset of sentences that contain the highlight
            preds_with_highlight = article_pred[article_pred['sentence_id'].isin(ids_with_highlight)]

            if filter_quotes:
                quoted_parts = get_quoted_parts_in_highlight(
                    highlight,
                    [sentences[idx] for idx in ids_with_highlight]
                )
                if len(quoted_parts) > 0:
                    quotes_with_highlights.loc[(prolific_id, article), 'highlights_with_quotes'].append(highlight)

            # check if any of the words in the labeled words are in the highlight
            moral_words_in_highlight = []
            for idx, pred_with_highlight in preds_with_highlight.iterrows():
                labeled_words = pred_with_highlight['labeled_words']
                for word in labeled_words:
                    if filter_quotes:
                        if any(word in quoted_part for quoted_part in quoted_parts):
                            moral_words_in_highlight.append(word)
                    else:
                        if word in highlight:
                            moral_words_in_highlight.append(word)

            for method in methods:
                if any(word in summaries.loc[article, method] for word in moral_words_in_highlight):
                    filtered_scores.loc[(prolific_id, article), method].append(row[method][highlight_id])

    # Add means columns like in scores_df
    for index, row in filtered_scores.iterrows():
        for method in methods:
            filtered_scores.loc[index, f"{method}_mean"] = np.mean(row[method]) if len(row[method]) > 0 else np.nan

    if filter_quotes:
        return filtered_scores, quotes_with_highlights
    else:
        return filtered_scores