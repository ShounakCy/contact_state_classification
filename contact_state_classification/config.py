# This is where all file names and path-related parameters are stored.
path = {
    "experiment_dir": "/home/shounak/Desktop/TUB/contact_state_classification/contact_state_classification",
    "dataset": ["RoboticsProject2510_2"],
    "test_set": ["RoboticsProject2510_2"]
    #"test_set": ["RoboticsProject2510"]
}
# This is where all classifier configuration parameters are stored.
# Since different classifiers may be used, parameters may need to be nested.
params = {
    "n_act": 12,
    "use_pca": False,
    "use_lda": False,
    "use_test_set": False,
    "simple_features": ["dist"],
    "complex_features": [],
    "n_splits": 4,
    "n_neighbors": 4,
    "n_components": 2,
    "classifier": "SHP",
    "cs_index_map": {"CS1": 1, "CS2": 2, "CS3": 3, "CS5": 4, "CS6": 5},
    "basic_visualization": True,
    "mvc_visualization":False
}
