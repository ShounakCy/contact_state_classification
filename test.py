import pandas as pd
import os
import numpy as np
from loguru import logger
import sys
# Load dateloader
from scipy.stats import gaussian_kde
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import contact_state_classification as csc
import numpy as np
from contact_state_classification import config as cfg
import pandas as pd
# import seaborn as sns
import random
# import matplotlib.pyplot as plt
from visdom import Visdom


def main():
    experiment_dir = csc.config.path["experiment_dir"]
    cs_classifier = csc.CSClassifier(experiment_dir=experiment_dir,
                                     dataset_name_list=csc.config.path["dataset"],
                                     test_set_name_list=csc.config.path["test_set"])    
    cs_classifier.cross_val_score(42)
    



if __name__ == "__main__":
    main()
