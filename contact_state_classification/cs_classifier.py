import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn_som.som import SOM
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from visdom import Visdom

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.datasets import load_basic_motions
import contact_state_classification as csc
from . import config as cfg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier




class CSClassifier:
    def __init__(self, experiment_dir, dataset_name_list, test_set_name_list):
        self.experiment_dir = experiment_dir
        self.dataset_name_list = dataset_name_list
        self.dataset_path_list = [self.experiment_dir + "/csd_result/" + x + ".pkl" for x in dataset_name_list]
        self.test_set_list = [self.experiment_dir + "/csd_result/" + x + ".pkl" for x in test_set_name_list]
        self.csd_dataset_plot_dir = self.experiment_dir + "/csd_result/plot/"
        os.makedirs(self.csd_dataset_plot_dir, exist_ok=True)
        self.csc_logger = logger

        # Dataset
        self.csd_data_df = None
        self.csd_data_dict = None
        self.csd_test_data_df = None
        self.csd_test_data_dict = None
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
        for path in self.dataset_path_list:
            if self.csd_data_df is None:
                self.csd_data_df = pd.read_pickle(path)
            else:
                self.csd_data_df.append(pd.read_pickle(path))
        self.csd_data_dict = self.csd_data_df.to_dict()

        for path in self.test_set_list:
            if self.csd_test_data_df is None:
                self.csd_test_data_df = pd.read_pickle(path)
            else:
                self.csd_test_data_df.append(pd.read_pickle(path))
        self.csd_test_data_dict = self.csd_test_data_df.to_dict()

        self.lb = preprocessing.LabelBinarizer()
        if cfg.params["classifier"] == "SHP":
            self.X, self.y = csc.CSClassifier.extract_features_from_df_for_shapelet(self.csd_data_df)
        else:
            self.X, self.y = self.extract_features_from_df(self.csd_data_df)
        self.lb.fit(self.y)                 

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
        num_labels = np.unique(self.y, axis=0).shape[0]
        if use_pca:
            self.pca()
        if use_lda:
            self.lda()
           
        if cfg.params["classifier"] == "MVC_1":
            # Time series concatenation
            # Concatenation of time series columns into a single long time series column via ColumnConcatenator and apply a classifier to the concatenated data,
            #Time Series Forest Classifier
            steps = [
                ("concatenate", ColumnConcatenator()),
                ("classify", TimeSeriesForestClassifier(n_estimators=100)),
            ]
            self.classifier = Pipeline(steps)
            print("MVC_1_shape", self.X.shape)
            self.classifier.fit(self.X, self.y)

        if cfg.params["classifier"] == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=cfg.params["n_neighbors"])
            print("KNN_shape",self.X.shape)
            self.classifier.fit(self.X, self.y)

            #if cfg.params["basic_visualization"]:
             #   self.basic_visualization()
        
        if cfg.params["classifier"] == "DTW":
            self.classifier = KNeighborsTimeSeriesClassifier(n_neighbors=cfg.params["n_neighbors"], distance="dtw")
            self.classifier.fit(self.X, self.y)
         

        else:
            return

    def cross_val_score(self, random_state=None):
        skf = StratifiedKFold(n_splits=cfg.params["n_splits"], shuffle=True, random_state=random_state)

        if cfg.params["classifier"] == "KNN":


            if (cfg.params["use_test_set"]):
                X_test, y_test = self.extract_features_from_df(self.csd_test_data_df)
                pred_labels = self.classifier.predict(X_test)
                print("Correct classification rate:", accuracy_score(y_test, pred_labels))
            else:
                score = cross_val_score(self.classifier, self.X, self.y, cv=skf)
                print(sum(score) / len(score))


        elif cfg.params["classifier"] == "MVC_1":

           for train_index, test_index in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                
                self.classifier.fit(X_train, y_train)
                pred_labels = self.classifier.predict(X_test)
                print("Correct classification rate:", accuracy_score(y_test, pred_labels))
                
        
        elif cfg.params["classifier"] == "DTW":

            if (cfg.params["use_test_set"]):
                X_test, y_test = self.extract_features_from_df(self.csd_test_data_df)
                pred_labels = self.classifier.predict(X_test)
                print("Correct classification rate:", accuracy_score(y_test, pred_labels))
            else:
            
                for train_index, test_index in skf.split(self.X, self.y):
                    X_train, X_test = self.X[train_index], self.X[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]
                    pred_labels = self.classifier.predict(X_test)
                    print("Correct classification rate:", accuracy_score(y_test, pred_labels))
                    
                    




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
        self.csc_logger.info("All classes from the dataset {} are {}: ", self.dataset_name_list, self.all_classes)
    
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
    def extract_features_from_df_for_shapelet(df):
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

    def shapelet_visualization(self, shapelet_sizes):
        XX = self.X
        distances = self.classifier.transform(self.X)
        # Make predictions and calculate accuracy score
        pred_labels = self.classifier.predict(self.X)
        print("Correct classification rate:", accuracy_score(self.y, pred_labels))

        # Plot the different discovered shapelets
        plt.figure()
        for i, sz in enumerate(shapelet_sizes.keys()):
            plt.subplot(len(shapelet_sizes), 1, i + 1)
            plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
            for shp in self.classifier.shapelets_:
                if ts_size(shp) == sz:
                    plt.plot(shp.ravel())
            plt.xlim([0, max(shapelet_sizes.keys()) - 1])

        plt.tight_layout()
        plt.show()

        # The loss history is accessible via the `model_` that is a keras model
        plt.figure()
        plt.plot(np.arange(1, self.classifier.n_iter_ + 1), self.classifier.history_["loss"])
        plt.title("Evolution of cross-entropy loss during training")
        plt.xlabel("Epochs")
        plt.show()

        viz = Visdom()
        assert viz.check_connection()
        try:
            for i, sz in enumerate(shapelet_sizes.keys()):
                viz.scatter(
                    X=distances,
                    Y=[cfg.params["cs_index_map"][x] for x in pred_labels],
                    opts=dict(
                        legend=list(cfg.params["cs_index_map"].keys()),
                        markersize=5,
                        title="%d shapelets of size %d" % (shapelet_sizes[sz], sz),
                        xlabel="Distance to 1st Shapelet",
                        ylabel="Distance to 2nd Shapelet",
                        zlabel="Distance to 3rd Shapelet",
                    )
                )

        except BaseException as err:
            print('Skipped matplotlib example')
            print('Error message: ', err)

    def extract_df(self, X):
        columns_simple_features = ['act' + str(y) + ' ' + x for x in cfg.params["simple_features"] for y in
                                   range(0, cfg.params["n_act"])]
        columns_complex_features = ['act_' + str(y) + ' joint_' + str(z) + ' ' + x for x in
                                    cfg.params["complex_features"] for y in
                                    range(0, cfg.params["n_act"]) for z in range(self.csd_data_dict[x][0][0].shape[0])]
        X_df = pd.DataFrame(data=X, index=range(0, X.shape[0]),
                                 columns=columns_simple_features + columns_complex_features)
        y_df = pd.DataFrame(data=self.y, index=range(0, len(self.y)), columns=['label'])
        return X_df.join(y_df)

