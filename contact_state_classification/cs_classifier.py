import pandas as pd
import os
import numpy as np
from loguru import logger
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn_som.som import SOM
from . import config as cfg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt



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

        # Classifier
        self.lb = None
        self.classifier = None

        # Dataset information
        self.all_classes = None
        self.num_classes = None

        # Train the classifier
        self.load_data()
        self.setup_classifier(cfg.params["use_pca"], cfg.params["use_lda"] )

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
        self.X, self.y = self.extract_features_from_df(self.csd_data_df)
        self.X = np.array(self.X)
        columns_simple_features = ['act' + str(y) + ' ' + x for x in cfg.params["simple_features"] for y in
                                   range(0, cfg.params["n_act"])]
        columns_complex_features = ['act_' + str(y) + ' joint_' + str(z) + ' ' + x for x in
                                    cfg.params["complex_features"] for y in
                                    range(0, cfg.params["n_act"]) for z in range(self.csd_data_dict[x][0][0].shape[0])]
        self.X_df = pd.DataFrame(data=self.X, index=range(0, self.X.shape[0]),
                                 columns=columns_simple_features + columns_complex_features)
        y_df = pd.DataFrame(data=self.y, index=range(0, len(self.y)), columns=['label'])
        self.X_df = self.X_df.join(y_df)
        self.lb.fit(self.y)
        # self.y = self.lb.transform(self.y)
        num_labels = np.unique(self.y, axis=0).shape[0]
        if cfg.params["classifier"] == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=cfg.params["n_neighbors"])
            if use_pca:
                self.pca()
            if use_lda:
                self.lda()
            self.classifier.fit(self.X, self.y)
        elif cfg.params["classifier"] == "SOM":
            self.classifier = SOM(m=6, n=1, dim=self.X.shape[1])
            if use_pca:
                self.pca()
            if use_lda:
                self.lda()
            self.classifier.fit(self.X, epochs=10, shuffle=False)
        else:
            return

    def cross_val_score(self, random_state=None):
        if cfg.params["classifier"] == "KNN":
            skf = StratifiedKFold(n_splits=cfg.params["n_splits"], shuffle=True, random_state=random_state)
            score = cross_val_score(self.classifier, self.X, self.y, cv=skf)
            print(sum(score) / len(score))
        elif cfg.params["classifier"] == "SOM":
            return
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
        pca = PCA(n_components=cfg.params["n_components"], svd_solver='auto', whiten='true')
        pca.fit(self.X)
        self.X = pca.transform(self.X)
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_)


    def plot(self, use_pca=False, use_lda=False):
        plt.subplot(1, 1, 1)
        for color, label in zip('rgbck', ('CS1', 'CS2', 'CS3', 'CS5', 'CS6')):
            plt.scatter(self.X[self.y == label, 0], self.X[self.y == label, 1],
                        c=color, label='{}'.format(label), cmap="plasma")

        plt.title('Point Cloud after LDA Transformation with 2 DC',
                      fontsize=14)
        plt.xlabel("1st DC")
        plt.ylabel("2nd DC")
        plt.legend()
        plt.show()

    def lda(self):
        lda = LDA(n_components=2)
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
