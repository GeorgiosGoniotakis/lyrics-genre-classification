# -*- coding: utf-8 -*-

"""This experiment has been individually packaged to enable
easier experimentation on online solutions like Kaggle and
Google Collaboratory. The current implements a custom
fully-connected architecture."""

import numpy as np
import pandas as pd
from keras import utils
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing import text
from sklearn.preprocessing import LabelEncoder

# Parameters and definitions
RANDOM_SEED = 0
VAL_SET_SIZE = 0.2

np.random.seed(RANDOM_SEED)

"""### File Paths"""

DATA = "../../data/380000_final.csv"
EMB_FILE_PATH = "../../emb/glove.840B.300d.txt"

"""### Helper Methods"""


def load_data():
    """Loads the training and testing sets into the memory.
    """
    return pd.read_csv(DATA)


"""### Data Wrangling"""

df = load_data()

df.dropna(inplace=True)
df.drop(df[(df.genre == "Not Available") | (df.genre == "Other")].index, inplace=True)

train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

"""### Data Preparation"""

params = {
    "max_words" : list(np.arange(1000, 14000, 1000)), # 1000 - 13000
    "epochs": list(np.arange(5, 11, 1)), # 1 - 10
    "n_outputs": [512],
    "batch_size": [64, 128]
}

train_posts = train['lyrics']
train_tags = train['genre']

valid_posts = validate["lyrics"]
valid_tags = validate["genre"]

test_posts = test['lyrics']
test_tags = test['genre']

"""### Evaluation"""

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
    return confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))


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


def export_report(data, model, title, filepath):
    global best_acc, best_f1

    x_train, x_val, x_test, y_train, y_val, y_test = data

    y_pred_train = model.predict(x_train, verbose=1)
    y_pred_train = (y_pred_train == y_pred_train.max(axis=1, keepdims=True)).astype(int)

    y_pred_val = model.predict(x_val, verbose=1)
    y_pred_val = (y_pred_val == y_pred_val.max(axis=1, keepdims=True)).astype(int)

    y_pred_test = model.predict(x_test, verbose=1)
    y_pred_test = (y_pred_test == y_pred_test.max(axis=1, keepdims=True)).astype(int)

    cur_test_acc = accuracy_score(y_test, y_pred_test)
    cur_test_f1 = f1_score(y_test, y_pred_test, average="macro")

    if cur_test_acc > best_acc:
        best_acc = cur_test_acc
        store_report("acc\tfile\n" + str(best_acc) + "\t" + filepath, "best/acc.csv")

    if cur_test_f1 > best_f1:
        best_f1 = cur_test_f1
        store_report("f1\tfile\n" + str(best_f1) + "\t" + filepath + ".csv", "best/f1.csv")

    metrics_report_train, class_report_train, conf_report_train = produce_detailed_report(y_train, y_pred_train)
    metrics_report_val, class_report_val, conf_report_val = produce_detailed_report(y_val, y_pred_val)
    metrics_report_test, class_report_test, conf_report_test = produce_detailed_report(y_test, y_pred_test)

    report = "Title: " + title + "\n\n" + "Results of Training Set\n" + "\n--------------------\n" + \
             str(metrics_report_train) + "\n\n" + str(class_report_train) + "\n\n" + str(conf_report_train) + \
             "\n\nResults of Validation Set\n" + "\n--------------------\n" + \
             str(metrics_report_val) + "\n\n" + str(class_report_val) + "\n\n" + str(conf_report_val) + \
             "\n\nResults of Test Set\n" + "\n--------------------\n" + \
             str(metrics_report_test) + "\n\n" + str(class_report_test) + "\n\n" + str(conf_report_test) + \
             "\n\n"

    print("Metrics for experiment have been stored to file: {}".format(filepath))
    store_report(report, "../../experiments/deep/" + filepath)


"""### Machine Learning"""


def build_model(data, w, e, o, b, run):
    x_train, x_val, x_test, y_train, y_val, y_test = data
    report_file = str(run) + "_advanced_fc_w" + str(w) + "_e" + str(e) + "_o" + str(o) + "_b" + str(b) + "_380"
    log_file = report_file + "_log.csv"
    title = "Fully Connected - Words: {}, Batch: {}, Epochs: {}, Outputs: {}".format(str(w), str(b), str(e), str(o))

    # Create a logger
    csv_logger = CSVLogger("../../logs/" + log_file, append=True, separator=';')

    # Build the model
    model = Sequential()
    model.add(Dense(o, input_shape=(w,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(x_train, y_train,
                        batch_size=b,
                        epochs=e,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[csv_logger,
                                   EarlyStopping(monitor='val_acc',
                                                 mode="max",
                                                 verbose=1,
                                                 patience=50,
                                                 min_delta=.2)])

    export_report(data, model, title, report_file)


"""### Run experiments"""

exp_id = 460
best_f1 = 100
best_acc = 100

for w in params["max_words"]:

    tokenize = text.Tokenizer(num_words=w, char_level=False)
    tokenize.fit_on_texts(train_posts)  # only fit on train

    x_train = tokenize.texts_to_matrix(train_posts)
    x_val = tokenize.texts_to_matrix(valid_posts)
    x_test = tokenize.texts_to_matrix(test_posts)

    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_val = encoder.transform(valid_tags)
    y_test = encoder.transform(test_tags)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_val = utils.to_categorical(y_val, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    for e in params["epochs"]:
        for o in params["n_outputs"]:
            for b in params["batch_size"]:
                build_model((x_train, x_val, x_test, y_train, y_val, y_test), w, e, o, b, exp_id)
                exp_id += 1
