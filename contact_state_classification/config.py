# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "/home/shounak/Desktop/TUB/contact_state_classification/contact_state_classification",
    "dataset_name": "RoboticsProject2510_2"
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "use_pca": False,
    "use_lda": True,
    "simple_features": ["dist", "obs_ee_theta", "obs_ee_phi", ],
    "complex_features": ["error_q"],
    "n_splits": 8,
    "n_neighbors": 4,
    "n_components": 2,
    "classifier": "KNN"
}
