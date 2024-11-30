#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from models import *
from utilities import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold


def print_result(n_data, setting, name_ml, model, X_test, real_prob):
    y_predic = model.predict_proba(X_test)[:, 1]
    error = 100 * np.linalg.norm(y_predic - real_prob, ord=1) / len(y_predic)
    print("{:<10} {:<25} {:<10} {:<5}".format(n_data, setting['feature_type'], name_ml, f"{error:.2f}"))

if __name__ == '__main__':
    """
    In this file, we create a dataset according to the simulation setting and we compare 
    the predicted probability of default with respect to the new ones.
    """

    # settings:
    rnd_seed = 1
    n_data = 500
    RANDOM_STATE = 1990
    sim_settings = json.load(
        open(os.path.join(".", "cfgs", "sim_settings.json"))
    )
    setting = sim_settings[0]
    np.random.seed(rnd_seed)

    # print intial results
    print(f"FEATURE TYPE: {setting['feature_type']}")
    print("{:<10} {:<25} {:<10} {:<5}".format("n_data", "feature_type", "ml_method", "error"))

    # simulate the features:
    fs = FeatureSimulator(
        n_feature = setting['n_features'],
        feature_type = setting['feature_type']
    )
    # Create dataset
    normalized_X = fs.generate_normalized_features(n_data)
    # Generate y_sim
    ls = LogitSimulator(fs, beta=setting['beta'])
    y_sim = ls.simulate_defaults(normalized_X)
    # Split training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_X, y_sim,
        test_size=0.2,
        random_state=rnd_seed
    )
    # real values:
    real_prob = ls.compute_real_probs(X_test)
    ###################
    # LR
    ###################
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)
    print_result(n_data, setting, "LR", model_LR, X_test, real_prob)
    
    ###################
    # RF_ISO
    ###################
    model_RF_iso = CalibratedClassifierCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE),
        method='isotonic'
    )
    model_RF_iso.fit(X_train, y_train)
    print_result(n_data, setting, "RF_iso", model_RF_iso, X_test, real_prob)

