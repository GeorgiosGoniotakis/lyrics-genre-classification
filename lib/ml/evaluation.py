"""
Contains all the methods which produce detailed statistics upon the completion
of an experiment.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix


def produce_metrics(y, y_pred):
    """Produces a report containing the accuracy, f1-score, precision and recall metrics.

    Args:
        y: The true classification
        y_pred: The predicted classification
    """
    return "Accuracy: {}, F1 Score: {}, Precision: {}, Recall: {}".format(accuracy_score(y, y_pred),
                                                                          f1_score(y, y_pred, average="macro"),
                                                                          precision_score(y, y_pred, average="macro"),
                                                                          recall_score(y, y_pred, average="macro"))


def produce_classification_report(y, y_pred):
    """Produces a classification report.

    Args:
        y: The true classification
        y_pred: The predicted classification
    """
    return classification_report(y, y_pred)


def produce_confusion_matrix(y, y_pred):
    """Produces a confusion matrix.

    Args:
        y: True target labels
        y_pred: Predicted targets
    """
    return confusion_matrix(y, y_pred)


def produce_detailed_report(y, y_pred):
    """Leverages all the methods of this collection to produce
    a detailed report using only one call.

    Args:
        title: The tile of the experiment (can contain the parameters, etc)
        y: True target labels
        y_pred: The predicted targets
    """
    metrics_report = produce_metrics(y, y_pred)
    class_report = produce_classification_report(y, y_pred)
    conf_report = produce_confusion_matrix(y, y_pred)
    return metrics_report, class_report, conf_report


def store_report(report, filepath):
    with open(filepath, "w") as f:
        f.write(report)
