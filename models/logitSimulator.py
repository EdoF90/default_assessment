# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .generateDatasets import FeatureSimulator


class LogitSimulator():
    def __init__(self, fs:FeatureSimulator, beta: np.array):
        """
        The model is exchangeable since the function computing the probability
        is the same for all creditors.
        """
        self.beta = beta
        self.fs = fs

    def __compute_probs(self, X):
        return 1 / (1 + np.exp(-self.beta[0] - np.dot(X, self.beta[1:])))

    def compute_real_probs(self, X_test):
        return self.__compute_probs(X_test)
    
    # START SKLEARN METHODS
    def predict(self, X_test, threshold=0.5):
        return self.compute_real_probs(X_test) > threshold
    
    def predict_proba(self, X_test):
        ans = np.ones((X_test.shape[0], 2))
        prb = self.compute_real_probs(X_test)
        ans[:,0] = 1 - prb
        ans[:,1] = prb
        return ans 
    # END SKLEARN METHODS

    def simulate_defaults(self, normalized_X):
        probs = self.__compute_probs(normalized_X)
        return [np.random.binomial(1, p) for p in probs]

    def compute_var_mc(self, alpha=0.9, n_creditors=100, n_obs=100, models_lst=[]):
        """
        The simulation must be done together also for the models_lst in order
        to limit the variance.
        """
        cvar_models = {}
        var_models = {}
        for model, key in models_lst:
            cvar_models[key] = np.zeros(n_obs)
            var_models[key] = np.zeros(n_obs)
        # initialize
        n_defaults = np.zeros(n_obs)
        for n in tqdm(range(n_obs)):
            # generate population
            df_tmp = self.fs.generate_normalized_features(n_creditors)
            # get number of default for the model
            for model, key in models_lst:
                probs_model = model.predict_proba(df_tmp)[:,1]
                var_models[key][n] = sum([np.random.binomial(1, p) for p in probs_model])
            # get number of default for the real model
            probs = self.__compute_probs(df_tmp)
            y_sim = [np.random.binomial(1, p) for p in probs]
            n_defaults[n] = sum(y_sim)

        # from the default take the quantile
        var = np.quantile(n_defaults, alpha)
        for model, key in models_lst:
            var_models[key] = np.quantile(var_models[key], alpha)
        # and the expected value over
        cvar = sum(n_defaults[n_defaults>var]) / sum(n_defaults>var)
        return var, cvar, var_models, cvar_models

    def plot_graph(self, normalized_X, y_sim, file_path=None):
        if normalized_X.shape[1] > 2:
            # with more than two dimensions show the PCA graph
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            colors = ['#1F77B4', '#FF7F0E']
            markers = ['o', 's']
            for l, c, m in zip(np.unique(y), colors, markers):
                plt.scatter(
                    X[y==l, 0],
                    X[y==l, 1],
                    c=c, label=l, marker=m
                )
            plt.title('Imbalanced dataset (2 PCA components)')
            plt.legend(loc='upper right')
        else:
            probs = self.compute_real_probs(normalized_X)
            # plot
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
            # plt ax 1
            sc = ax1.scatter(
                normalized_X['A'], normalized_X['B'], c=probs,
                vmin=0, vmax=1
            )
            fig.colorbar(sc, ax=ax1)
            ax1.set_title("Real Probabilities")
            ax1.set_aspect('equal', adjustable='box')
            # plt ax 2
            default_pos = np.where(np.array(y_sim) > 0.9)
            no_default_pos = np.where(np.array(y_sim) < 0.1)
            ax2.scatter(
                normalized_X.iloc[no_default_pos]['A'],
                normalized_X.iloc[no_default_pos]['B'],
                c='blue',
                label='No Default'
            )
            ax2.scatter(
                normalized_X.iloc[default_pos]['A'],
                normalized_X.iloc[default_pos]['B'],
                c='red',
                label='Default'
            )
            ax2.legend()
            ax2.set_title("Defaults")
            ax2.set_aspect('equal', adjustable='box')

        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()
