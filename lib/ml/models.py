from lib.exceptions.MLModel import NoMLModelFound, WrongParameters
from lib.utils.timer import Timer
from lib.ml.evaluation import *
from lib.ml.exporter import *
from lib.ml.prepare import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Model:

    def __init__(self, kind, method, title, filepath):
        """Builds an experiment given a model and some
        metadata.

        Args:
            kind: The Machine Learning model
            method: The preparation method (tfidf, bow)
            title: The title of the experiment
            filepath: The file name of the report
        """
        self.__model = self.init_model(kind)
        self.__method = method
        self.__title = title
        self.__filepath = filepath
        self.__timer = None

    def init_model(self, kind):
        """Initializes a Machine Learning model.

        Args:
            kind: The name of the Machine Learning model

        Returns:
            The requested model
        """
        if kind == "SVM":
            return SVC()
        elif kind == "RandomForest":
            return RandomForestClassifier(n_estimators=500)
        elif kind == "LogisticRegression":
            return LogisticRegression()
        else:
            raise NoMLModelFound(kind)

    def train_predict(self, data):
        """It handles the training of the model. """
        self.validate_input()  # Validates the input
        self.__timer = Timer()  # Start a timer

        print("Training process has started for experiment: {}".format(self.__title))
        X_train, X_test, y_train, y_test = prepare_data(data, self.__method)
        self.__model.fit(X_train, y_train)

        print("Performing predictions for experiment: {}".format(self.__title))
        y_pred_train = self.__model.predict(X_train)
        y_pred_test = self.__model.predict(X_test)

        print("Calculating metrics for experiment: {}.".format(self.__title))

        metrics_report_train, class_report_train, conf_report_train = produce_detailed_report(y_train, y_pred_train)
        metrics_report_test, class_report_test, conf_report_test = produce_detailed_report(y_test, y_pred_test)

        report = "Title: " + self.__title + "\n\n" + "Results of Training Set\n" + "\n--------------------\n" + \
                 str(metrics_report_train) + "\n\n" + str(class_report_train) + "\n\n" + str(conf_report_train) + \
                 "\n\nResults of Test Set\n" + "\n--------------------\n" + \
                 str(metrics_report_test) + "\n\n" + str(class_report_test) + "\n\n" + str(conf_report_test) + \
                 "\n\nTotal training time: " + str(round(self.__timer.get_time(), 2)) + " secs or " \
                 + str(round(self.__timer.get_time() / 60, 2)) + " mins"

        print("Metrics for experiment have been stored to file: {}".format(self.__filepath))
        store_report("experiments/" + self.__filepath, report)

    def validate_input(self):
        if not (self.__title and self.__filepath and (self.__method == "tfidf" or self.__method == "bow")):
            raise WrongParameters([self.__title, self.__filepath, self.__method])

    def get_model(self):
        return self.__model
