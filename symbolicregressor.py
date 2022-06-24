import random
import pandas as pd
import numpy as np
from torch import seed
import symbolic.etl as etl
from icecream import ic
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from symbolic.gpdeap import SymbRegressorDEAP

def sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def qerror(log_y_true, log_y_pred, min_val, max_val, original=False):
    model_output = sigmoid(log_y_pred)
    targets = etl.unnormalize_labels(log_y_true, min_val, max_val)
    preds = etl.unnormalize_labels(model_output, min_val, max_val)

    if original:
        return np.mean([preds[i] / targets[i] if preds[i] > targets[i] else targets[i] / preds[i] for i in range(len(targets))])
    else:
        return metrics.mean_squared_log_error(targets, preds)
    

def run_experiment(X, y, fitness_function, loss, population_size, verbose, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.1, 
        random_state=random_seed
    )

    fitfunc = make_fitness(function=fitness_function, greater_is_better=False)

    symbolic_estimator = SymbolicRegressor(
        population_size=population_size,
        generations=5, 
        stopping_criteria=0.01,
        p_crossover=0.7, 
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01, 
        p_point_mutation=0.1,
        p_point_replace=0.05,
        #max_samples=0.9, 
        #parsimony_coefficient='auto', 
        tournament_size=5,
        init_method='half and half',
        init_depth=(1, 10),
        metric=fitfunc,
        function_set=['add', 'sub', 'mul', 'cos', 'sin', 'inv', 'log'],
        const_range=(-1, 1),
        low_memory=True,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_seed,
    )

    #symbolic_estimator = SymbRegressorDEAP(fitness_function, population_size)
    symbolic_estimator.fit(X_train, y_train)

    best_program = str(symbolic_estimator._program)
    ic(best_program)

    y_pred_train = symbolic_estimator.predict(X_train)
    y_pred_test = symbolic_estimator.predict(X_test)

    gp_qerror_train = loss(y_train, y_pred_train)
    gp_qerror_test = loss(y_test, y_pred_test)
    
    ic(gp_qerror_train)
    ic(gp_qerror_test)
    
    baseline_regressor = Ridge(random_state=random_seed)
    baseline_regressor.fit(X_train, y_train)
    
    y_baseline_train = baseline_regressor.predict(X_train)
    y_baseline_test = baseline_regressor.predict(X_test)
    
    ic(metrics.mean_squared_error(y_train, y_baseline_train))
    ic(metrics.mean_squared_error(y_test, y_baseline_test))

    baseline_qerror_train = loss(y_train, y_baseline_train)
    baseline_qerror_test = loss(y_test, y_baseline_test)
    
    ic(baseline_qerror_train)
    ic(baseline_qerror_test)

    return gp_qerror_test, baseline_qerror_test

def load_data():
    dataset = etl.load_dataset()
    data = dataset.data
    X = data[:,:-1]
    y = data[:,-1]
    labels_min_max_values = dataset.labels_min_max_values
    
    return X, y, labels_min_max_values

def main():
    POPULATION_SIZE = 128
    VERBOSE = 1

    seeds = [0]
    ic(seeds)
    num_seeds = len(seeds)
    
    X, y, labels_min_max_values = load_data()

    fitness_function = lambda y_true, y_pred, w=None: qerror(y_true, y_pred, labels_min_max_values[0], labels_min_max_values[1], False)

    loss = lambda y_true, y_pred: qerror(y_true, y_pred, labels_min_max_values[0], labels_min_max_values[1], True)

    mean_error_symbolic = 0
    mean_error_baseline = 0

    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        
        error_symbolic, error_baseline  = run_experiment(
            X, 
            y, 
            fitness_function, 
            loss,
            population_size=POPULATION_SIZE, 
            verbose=VERBOSE, 
            random_seed=s
        )
        
        mean_error_symbolic += error_symbolic
        mean_error_baseline += error_baseline

    mean_error_symbolic = mean_error_symbolic/num_seeds
    mean_error_baseline = mean_error_baseline/num_seeds
    
    ic(mean_error_baseline)
    ic(mean_error_symbolic)

if __name__ == '__main__':
    main()
    