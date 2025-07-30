import numpy as np
from sklearn.metrics import classification_report
import evaluate


def get_performance_metrics(df_test):
    y_test = df_test.label.astype(int)
    y_pred = df_test.predictions.round()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, output_dict=True, zero_division=0.0))


def sequence_classification_metrics(eval_pred):
    predictions, labels = eval_pred

    # it's a classification task, take the argmax
    predictions = np.argmax(predictions, axis=1)

    return classification_report(
        labels, predictions, output_dict=True, zero_division=0.0)


def seqeval_metrics(predictions, labels, label_list=['non-moral', 'moral']):
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def token_classification_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    return seqeval_metrics(predictions, labels)


def evaluate_on_df(df, task):
    if task == 'sequence_classification':
        return classification_report(
        df.label, df.predictions, output_dict=True, zero_division=0.0)
    elif task == 'token_classification':
        return seqeval_metrics(df.predictions, df.labels)
