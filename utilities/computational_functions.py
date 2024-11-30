# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as grb
from scipy.stats import beta
from scipy.stats import betabinom
import matplotlib.pyplot as plt
from scipy.optimize import root


def FontanaBounds(alpha, p, n_creditors):
    '''
    It computes the Bounds according to Proposition 2.1 in 
    " Exchangeable Bernoulli distributions: High dimensional
    simulation, estimation, and testing" by R. Fontana , P. Semeraro
    '''
    j_1M = np.floor(p * n_creditors)
    j_2m = np.ceil(p * n_creditors)
    j_1p = ((p - (1 - alpha)) * n_creditors) / alpha
    if p < 1 - alpha:
        minVar = 0
        maxVar = int(p * n_creditors / (1-alpha))
    elif p > 1 - alpha + alpha / n_creditors * j_1M: # third case of the Theorem
        minVar = j_2m
        maxVar = n_creditors
    else:
        minVar = np.floor(j_1p)
        maxVar = n_creditors
    # CVaR:
    if p * n_creditors - np.floor(p * n_creditors) > alpha:
        minCVar = np.floor(p*n_creditors)
    else:
        minCVar = np.ceil(p*n_creditors)
    maxCVar = n_creditors
    return minVar, maxVar, minCVar, maxCVar


def CVarBounds(alpha, d, mu_1, mu_2, gap=None, time_limit=None, verbose=False, file_name=None):
    '''
    It computes the CVarBounds using gurobi.
    '''
    model = grb.Model("Min CVaR")
    phi = model.addVar(
        vtype=grb.GRB.CONTINUOUS,
        name='phi'
    )
    Z = model.addVars(
        d, vtype=grb.GRB.CONTINUOUS,
        lb=0.0, name='Z'
    )
    P = model.addVars(
        d, vtype=grb.GRB.CONTINUOUS,
        lb=0.0, ub=1.0, name='P'
    )
    Y_not_null = model.addVars(
        d, vtype=grb.GRB.BINARY,
        name='Y_not_null'
    )
    obj_func = grb.quicksum(  
        1/(1 - alpha) * P[i] * Z[i] for i in range(d)
    )    
    model.setObjective(phi + obj_func, grb.GRB.MINIMIZE)
    model.addConstrs(
        (Z[i] >= i - phi for i in range(d)),
        "Z_def"
    )
    model.addConstr(
        grb.quicksum(
            i * P[i]
            for i in range(d)
        ) == mu_1 * d,
        name='first_moment'
    )
    model.addConstr(
        grb.quicksum(
            i**2 * P[i]
            for i in range(d)
        ) == mu_1 * d + d*(d-1)*mu_2,
        name='second_moment'
    )
    model.addConstr(
        grb.quicksum(
            P[i]
            for i in range(d)
        ) == 1,
        name='sum_to_one'
    )
    model.addConstrs(
        (P[i] <= Y_not_null[i] for i in range(d)),
        name='sum_to_one'
    )
    model.addConstr(
        grb.quicksum(
            Y_not_null[i]
            for i in range(d)
        ) == 3,
        name='just_3_non_null'
    )
    model.update()
    if gap:
        model.setParam('MIPgap', gap)
    if time_limit:
        model.setParam(grb.GRB.Param.TimeLimit, time_limit)
    if verbose:
        model.setParam('OutputFlag', 1)
    else:
        model.setParam('OutputFlag', 0)
    model.setParam('LogFile', './logs/gurobi.log')
    if file_name:
        model.write(f"./logs/{file_name}")
    start = time.time()
    model.params.NonConvex = 2
    model.optimize()
    end = time.time()
    comp_time = end - start
    if model.status == grb.GRB.Status.OPTIMAL or model.status == grb.GRB.Status.TIME_LIMIT:
        sol = [ P[i].X for i in range(d) ]
        return np.ceil(model.getObjective().getValue()), sol, comp_time
    else:
        pass

def CVarBoundsHeu(alpha, probs, gap=None, time_limit=None, verbose=False, file_name=None):
    pass



def beta_bin_quantile(alpha, probs, n_creditors):
    p = np.mean(probs)
    mu = np.mean(probs**2)
    sol_lr = root(moment_matching, [1.0, 1.0], args=(p, mu))
    a, b = sol_lr.x
    # a, b, loc, scale = beta.fit(
    #     probs,
    #     #floc=0, # set location equal to zero
    #     #fscale=1 # set scale equal to one
    # )
    return betabinom.ppf(alpha, n=n_creditors, a=a, b=b), a, b


# evaluate using the ECE (expect calibration error)
def expected_calibration_error(y, proba, bins = 'fd'):
    bin_count, bin_edges = np.histogram(proba, bins = bins)
    n_bins = len(bin_count)
    bin_edges[0] -= 1e-8 # because left edge is not included
    bin_id = np.digitize(proba, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins)
    bin_probasum = np.bincount(bin_id, weights = proba, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0)
    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
    return ece 

class Identity():
    def __init__(self):
        pass

    def fit_resample(self, X, y):
        return X, y

def moment_matching(X, p, mu):
    x, y = X
    # all RHS have to be 0
    f = [
        x / (x+y) - p,
        (x*(x+1)) / ((x+y)*(x+y+1)) - mu
    ]
    return f

    
def plot_beta_hist(pred_probs, a, b, title="", file_path=None):
    plt.hist(
        pred_probs,
        bins=30,
        density=True,
        label='Predicted probabilities'
    )
    x = np.linspace(
        beta.ppf(0.01, a, b),
        1,
        100
    )
    plt.ylim(0, 10)
    plt.grid()
    plt.plot(
        x, beta.pdf(x, a, b),
        'r-', alpha=0.6, label='Beta pdf'
    )
    plt.legend()
    plt.xlabel(r"$Q$")
    plt.title(title)
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()
