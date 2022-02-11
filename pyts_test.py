import contact_state_classification as csc
from contact_state_classification import config as cfg
import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import LearningShapelets
from pyts.datasets import load_gunpoint
from pyts.utils import windowed_view


def example_pyts():
    # Load the data set and fit the classifier
    X, _, y, _ = load_gunpoint(return_X_y=True)
    clf = LearningShapelets(random_state=42, tol=0.01)
    clf.fit(X, y)

    # Select two shapelets
    shapelets = np.asarray([clf.shapelets_[0, -9], clf.shapelets_[0, -12]])

    # Derive the distances between the time series and the shapelets
    shapelet_size = shapelets.shape[1]
    X_window = windowed_view(X, window_size=shapelet_size, window_step=1)
    X_dist = np.mean(
        (X_window[:, :, None] - shapelets[None, :]) ** 2, axis=3).min(axis=1)

    plt.figure(figsize=(14, 4))

    # Plot the two shapelets
    plt.subplot(1, 2, 1)
    plt.plot(shapelets[0])
    plt.plot(shapelets[1])
    plt.title('Two learned shapelets', fontsize=14)

    # Plot the distances
    plt.subplot(1, 2, 2)
    for color, label in zip('br', (1, 2)):
        plt.scatter(X_dist[y == label, 0], X_dist[y == label, 1],
                    c=color, label='Class {}'.format(label))
    plt.title('Distances between the time series and both shapelets',
              fontsize=14)
    plt.legend()
    plt.show()


def main_pyts():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir, dataset_name=csc.config.path["dataset_name"])
    df = cs_classifier.csd_data_df
    X, y = csc.CSClassifier.extract_features_from_df(df)
    clf = LearningShapelets(random_state=42, tol=0.01)
    clf.fit(X, y)

    # Select two shapelets
    shapelets = np.asarray([clf.shapelets_[0, 0], clf.shapelets_[0, 1]])

    # Derive the distances between the time series and the shapelets
    shapelet_size = shapelets.shape[1]
    X_window = windowed_view(X, window_size=shapelet_size, window_step=1)
    X_dist = np.mean(
        (X_window[:, :, None] - shapelets[None, :]) ** 2, axis=3).min(axis=1)

    plt.figure(figsize=(14, 4))

    # Plot the two shapelets
    plt.subplot(1, 2, 1)
    plt.plot(shapelets[0])
    plt.plot(shapelets[1])
    plt.title('Two learned shapelets', fontsize=14)

    # Plot the distances
    plt.subplot(1, 2, 2)
    for color, label in zip('rgbck', ('CS1', 'CS2', 'CS3', 'CS5', 'CS6')):
        plt.scatter(X_dist[y == label, 0], X_dist[y == label, 1],
                    c=color, label='Class {}'.format(label))
    plt.title('Distances between the time series and both shapelets',
              fontsize=14)
    plt.legend()
    plt.show()

    print(clf.predict(X))


if __name__ == "__main__":
    main_pyts()
