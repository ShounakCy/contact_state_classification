import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from visdom import Visdom

from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict

from . import config as cfg
import tensorflow as tf

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#
# # Path to root of this project, contains lots of modules
# import sys
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.getcwd())


class CSClassifier:
    def __init__(self, experiment_dir, dataset_name):
        self.experiment_dir = experiment_dir
        self.dataset_name = dataset_name
        self.dataset_path = self.experiment_dir + "/csd_result/" + dataset_name + ".pkl"
        self.csd_dataset_plot_dir = self.experiment_dir + "/csd_result/plot/"
        os.makedirs(self.csd_dataset_plot_dir, exist_ok=True)
        self.csc_logger = logger

        # Dataset
        self.csd_data_df = None
        self.csd_data_dict = None
        self.X = []
        self.y = []
        self.X_df = None
        self.accuracy_score = None
        self.accuracy =[]

        # Classifier
        self.lb = None
        self.classifier = None

        # Dataset information
        self.all_classes = None
        self.num_classes = None

        # Train the classifier
        self.load_data()
        self.setup_classifier(cfg.params["use_pca"], cfg.params["use_lda"])

        self.get_dataset_information()

    def load_data(self):
        # load data to dict, because processing of dataframe takes too much time
        self.csd_data_df = pd.read_pickle(self.dataset_path)
        self.csd_data_dict = self.csd_data_df.to_dict()

    def get_traj_index_by_labels(self, label):
        traj_index_dict = dict()
        for key, value in self.csd_data_dict['label'].items():
            if label in value:
                traj_index_dict[key] = value

        labels = list(set(traj_index_dict.values()))

        return traj_index_dict, labels

    def merge_feature_by_labels(self, traj_index_dict=None, feature="dist", labels=None):
        feature_values = dict()
        for label in labels:
            feature_values[label] = []
            traj_index_list = [key for key, value in traj_index_dict.items() if value == label]
            for traj_index in traj_index_list:
                feature_values[label].append(self.csd_data_dict[feature][traj_index])
        return feature_values

    def setup_classifier(self, use_pca=False, use_lda=False):
        self.lb = preprocessing.LabelBinarizer()
        if cfg.params["classifier"] == "MVC_1":
            self.X, self.y = CSClassifier.extract_features_from_df_for_mvc(self.csd_data_df)
            print("MVC_Shape",self.X.shape)
        else:
            self.X, self.y = self.extract_features_from_df(self.csd_data_df)
            #print("KNN_X_Shape",self.X.shape)
            self.X = np.array(self.X)
        self.lb.fit(self.y)
        # self.y = self.lb.transform(self.y)
        num_labels = np.unique(self.y, axis=0).shape[0]
        if use_pca:
            self.pca()
        if use_lda:
            self.lda()
        if cfg.params["classifier"] == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=cfg.params["n_neighbors"])
            print("KNN_shape",self.X.shape)
            self.classifier.fit(self.X, self.y)

            if cfg.params["basic_visualization"]:
                self.basic_visualization()



        elif cfg.params["classifier"] == "MVC_1":
            # Time series concatenation
            # Concatenation of time series columns into a single long time series column via ColumnConcatenator and apply a classifier to the concatenated data,

            steps = [
                ("concatenate", ColumnConcatenator()),
                ("classify", TimeSeriesForestClassifier(n_estimators=100)),
            ]
            self.classifier = Pipeline(steps)
            print("MVC_1_shape", self.X.shape)
            self.classifier.fit(self.X, self.y)

            if cfg.params["mvc_visualization"]:
               self.mvc_visualization()

        elif cfg.params["classifier"] == "MVC_2":
            self.classifier = MrSEQLClassifier()
            self.classifier.fit(self.X, self.y)

            if cfg.params["basic_visualization"]:
                self.basic_visualization()

        else:
            return

    def cross_val_score(self, random_state=None):
        skf = StratifiedKFold(n_splits=cfg.params["n_splits"], shuffle=True, random_state=random_state)

        if cfg.params["classifier"] == "KNN":

            score = cross_val_score(self.classifier, self.X, self.y, cv=skf)
            self.accuracy_score = sum(score) / len(score)
            print("score", self.accuracy_score)
            return self.accuracy_score
        elif cfg.params["classifier"] == "SOM":
            return

        elif cfg.params["classifier"] == "MVC_1":

            for train_index, test_index in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                self.classifier.fit(X_train, y_train)
                pred_labels = self.classifier.predict(X_test)
                score = accuracy_score(y_test, pred_labels)
                self.accuracy.append(score)
                #return self.accuracy_score
            self.accuracy_score = sum(self.accuracy)/len(self.accuracy)
            print(self.accuracy_score)
            return self.accuracy_score



        else:
            return

    def fit(self):
        if cfg.params["classifier"] == "KNN":
            self.classifier.fit(self.X, self.y)
        elif cfg.params["classifier"] == "SOM":
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        else:
            return

    def get_dataset_information(self):
        self.all_classes = self.lb.classes_
        self.num_classes = len(self.all_classes)
        self.csc_logger.info("All classes from the dataset {} are {}: ", self.dataset_name, self.all_classes)

    def predict(self, input_data):
        result = self.classifier.predict(input_data)
        if cfg.params["classifier"] == "KNN":
            label = self.lb.inverse_transform(result)
            return result, label
        elif cfg.params["classifier"] == "SOM":
            return result, None

    def pca(self):
        if len(self.X.shape) > 2:
            return
        pca = PCA(n_components=cfg.params["n_components"], svd_solver='auto', whiten='true')
        pca.fit(self.X)
        self.X = pca.transform(self.X)
        print("X_pca",self.X.shape)
        print("variance_ratio:")
        print(pca.explained_variance_ratio_)
        print("variance:")
        print(pca.explained_variance_)

    def lda(self):
        if len(self.X.shape) > 2:
            return
        lda = LDA(n_components=cfg.params["n_components"])
        lda.fit(self.X, self.y)
        self.X = lda.transform(self.X)
        print("LDA variance ratio:", lda.explained_variance_ratio_)

    @staticmethod
    def extract_features_from_df(df):
        X = []
        y = []
        for index, row in df.iterrows():
            x = []
            for feature in cfg.params["simple_features"]:
                x = x + row[feature]
            for feature in cfg.params["complex_features"]:
                x = x + np.concatenate(row[feature]).ravel().tolist()
            X.append(x)
            y.append(row["label"])
        X = np.array(X)
        y = np.array(y)
        return X, y

    @staticmethod
    def extract_features_from_df_for_mvc(df):
        X = []
        y = []
        for index, row in df.iterrows():
            x = np.zeros((cfg.params["n_act"], 0))
            for feature in cfg.params["simple_features"]:
                x = np.hstack((x, np.array(row[feature]).reshape((cfg.params["n_act"], 1))))
            for feature in cfg.params["complex_features"]:
                x = np.hstack((x, np.array(row[feature])))
            X.append(x)
            y.append(row["label"])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def basic_visualization(self):
        # Plot the distances

        viz = Visdom()
        score = self.cross_val_score(42)
        assert viz.check_connection()
        try:
            viz.scatter(
                X=self.X,
                Y=[cfg.params["cs_index_map"][x] for x in self.y],
                opts=dict(
                    legend=list(cfg.params["cs_index_map"].keys()),
                    markersize=10,
                    title="After LDA with %d PC and Accuracy : %2f" %
                          (cfg.params["n_components"], score),

                    xlabel="DC1",

                    ylabel="DC2",
                    zlabel="DC3",
                )
            )
        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)

