#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from models import *
from utilities import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


np.random.seed(1)


if __name__ == '__main__': 
    # READ SETTING
    alpha = 0.9
    RANDOM_STATE = 1990
    sim_settings = json.load(
        open(os.path.join(".", "cfgs", "sim_settings.json"))
    )
    setting = sim_settings[0]
    n_data = 500
    rnd_seed = 1
    np.random.seed(rnd_seed)
    # START EXPERIMENT
    models = {
        (
            LogisticRegression(),
            'LogisticRegression'
        ),
        (
            AdaBoostClassifier(random_state=0),
            'AdaBoostClassifier'
        ),
        (
            RandomForestClassifier(random_state=RANDOM_STATE),
            'RandomForest'
        ),
        (
            MLPClassifier(random_state=RANDOM_STATE),
            'MLP'
        ),
        (
            KNeighborsClassifier(),
            'KNN'
        )
    }
    # Run experiment
    print(f">>> {setting['feature_type']} <<<")
    fs = FeatureSimulator(
        n_feature = setting['n_features'],
        feature_type = setting['feature_type']
    )
    str_to_print_general = f"{setting['feature_type']},{n_data},{alpha},{rnd_seed}"
    # Create dataset
    normalized_X = fs.generate_normalized_features(n_data)
    # Generate y_real
    ls = LogitSimulator(fs, beta = setting['beta'])
    y_real = ls.simulate_defaults(normalized_X)

    # Split training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_X, y_real,
        stratify=y_real,
        test_size=0.2,
        random_state=rnd_seed
    )

    # real values under exchangeability assumption:
    real_probs = ls.compute_real_probs(X_test)
    pi_1_real = np.average(real_probs)
    pi_2_real = np.average(real_probs**2)
    minVar_real, maxVar_real, minCVar_real, maxCVar_real = FontanaBounds(
        alpha, pi_1_real, n_data
    )
    beta_var_real, a_real, b_real = beta_bin_quantile(
        alpha = alpha,
        probs = real_probs,
        n_creditors = n_data
    )
    rho_real = 1/(1 + a_real + b_real)
    print(f"{'N DATA':<7} {'METHOD':<50} {'minVar':<10} {'minCVar':<10} {'beta_var':<10} {'pi_1':<5} {'pi_2':<5} {'rho':<5} {'a':<5} {'b':<5} ")
    print(f"{n_data:<7} {'Real [Exchangeable] CVaR Bounds':<50} {minVar_real:<10.0f} {minCVar_real:<10.0f} {beta_var_real:<10.0f} {pi_1_real:<5.2f} {pi_2_real:<5.2f} {rho_real:<5.2f} {a_real:<5.2f} {b_real:<5.2f} ")

    # create list of model fitted
    models_fitted = []
    # repeat for all the models
    for model, key in models:
        # Fit model
        model.fit(X_train, y_train)
        models_fitted.append(
            (model, key)
        )
        # Compute prob forecasted:
        probs_forecasted = model.predict_proba(X_test)[:,1]
    
        # Use these probs to compute exchangeables bound:
        pi_1_ml = np.average(probs_forecasted)
        pi_2_ml = np.average(probs_forecasted**2)
        minVar_ml, maxVar_ml, minCVar_ml, maxCVar_ml = FontanaBounds(
            alpha, np.average(pi_1_ml), n_data
        )
        beta_var_ml, a_ml, b_ml = beta_bin_quantile(
            alpha = alpha,
            probs = probs_forecasted,
            n_creditors = n_data
        )
        rho_ml = 1/(1+a_ml+b_ml)
        # print results
        model_str = f"{key} [Non Exchangeable]"
        print(f"{n_data:<7} {model_str:<50} {minVar_ml:<10.0f} {minCVar_ml:<10.0f} {beta_var_ml:<10.0f} {pi_1_ml:<5.2f} {pi_2_ml:<5.2f} {rho_ml:<5.2f} {a_ml:<5.2f} {b_ml:<5.2f}")

    # Compute VaR using the models (non exchangeable)
    var, cvar, var_models, cvar_models  = ls.compute_var_mc(
        alpha=alpha,
        n_creditors=n_data,
        n_obs=50,
        models_lst=models_fitted
    )
    print(f"{'N DATA':<7} {'METHOD':<50} {'minVar':<10} {'minCVar':<10}")
    print(f"{n_data:<7} {'Real [Non Exchangeable]':<50} {var:<10.0f} {cvar:<10.0f}")
    for model, key in models:
        model_str = f"{key} [Non Exchangeable]"
        print(f"{n_data:<7} {model_str:<50} {var_models[key]:<10.0f} {var_models[key]:<10.0f}")
