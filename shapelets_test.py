import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
import contact_state_classification as csc
from contact_state_classification import config as cfg
import os


def example():
    # Set seed for determinism
    np.random.seed(0)

    # Load the Trace dataset
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

    # Normalize each of the timeseries in the Trace dataset
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

    # Get statistics of the dataset
    n_ts, ts_sz = X_train.shape[:2]
    n_classes = len(set(y_train))

    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=0.1,
                                                           r=1)

    # Define the model using parameters provided by the authors (except that we
    # use fewer iterations here)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=.01,
                                max_iter=200,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy score
    pred_labels = shp_clf.predict(X_test)
    print("Correct classification rate:", accuracy_score(y_test, pred_labels))

    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])

    plt.tight_layout()
    plt.show()

    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()

def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir, dataset_name=csc.config.path["dataset_name"])
    cs_classifier.pca(n_components=5)
    df = cs_classifier.csd_data_df
    X, y = csc.CSClassifier.extract_features_from_df_for_shapelet(df)

    # Normalize each of the timeseries in the Trace dataset
    # X_train = TimeSeriesScalerMinMax().fit_transform(X)

    # Get statistics of the dataset
    n_ts, ts_sz = X.shape[:2]
    n_classes = len(set(y))

    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=0.1,
                                                           r=1)

    # Define the model using parameters provided by the authors (except that we
    # use fewer iterations here)
    shp_clf = LearningShapelets(n_shapelets_per_size={2: 12},
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=.01,
                                max_iter=800,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(X, y)

    # Make predictions and calculate accuracy score
    pred_labels = shp_clf.predict(X)
    print("Correct classification rate:", accuracy_score(y, pred_labels))

    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])

    plt.tight_layout()
    plt.show()

    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()

if __name__ == "__main__":
    main()
