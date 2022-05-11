import random
import numpy as np
from torch import seed
import symbolic.etl as etl
from icecream import ic
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
  
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def qerror_loss(log_y_true, log_y_pred, min_val, max_val):
    qerror = []
    model_output = sigmoid(log_y_pred)
    targets = etl.unnormalize_labels(log_y_true, min_val, max_val)
    preds = etl.unnormalize_labels(model_output, min_val, max_val)

    for i in range(len(targets)):
        if preds[i] > targets[i]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    
    return np.mean(qerror)

def run_experiment(X, y, fitness_function, population_size, verbose, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)
    
    baseline_regressor = Ridge(random_state=random_seed)
    baseline_regressor.fit(X_train, y_train)

    baseline_train = baseline_regressor.predict(X_train)
    baseline_test = baseline_regressor.predict(X_test)
    
    # ic(metrics.mean_squared_error(y_train, baseline_train))
    # ic(metrics.mean_squared_error(y_test, baseline_test))
    # ic(metrics.mean_squared_error(y_test, baseline_test, squared=False))
    # ic(metrics.r2_score(y_train, baseline_train))
    # ic(metrics.r2_score(y_test, baseline_test))
    
    fitfunc = make_fitness(function=fitness_function, greater_is_better=False)

    train_error_baseline = fitfunc(y_train, baseline_train, None)
    test_error_baseline = fitfunc(y_test, baseline_test, None)
    
    ic(train_error_baseline)
    ic(test_error_baseline)

    symbolic_estimator = SymbolicRegressor(
        population_size=population_size,
        generations=30, 
        stopping_criteria=0.01,
        p_crossover=0.8, 
        p_subtree_mutation=0.05,
        p_hoist_mutation=0.01, 
        p_point_mutation=0.1,
        max_samples=0.9, 
        parsimony_coefficient=0.001, 
        tournament_size=10,
        metric=fitfunc,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'cos'],
        const_range=None,
        low_memory=True,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_seed,)
    
    symbolic_estimator.fit(X_train, y_train)
    
    y_pred_train = symbolic_estimator.predict(X_train)
    y_pred_test = symbolic_estimator.predict(X_test)

    train_error_symbolic_estimator = fitness_function(y_train, y_pred_train, None)
    test_error_symbolic_estimator = fitness_function(y_test, y_pred_test, None)
    best_program = str(symbolic_estimator._program)
    
    ic(train_error_symbolic_estimator)
    ic(test_error_symbolic_estimator)
    ic(best_program)
    
    # ic(metrics.mean_squared_error(y_train, y_pred_train))
    # ic(metrics.mean_squared_error(y_test, y_pred_test))
    # ic(metrics.mean_squared_error(y_test, y_pred_test, squared=False))
    # ic(metrics.r2_score(y_train, y_pred_train))
    # ic(metrics.r2_score(y_test, y_pred_test))

    return test_error_symbolic_estimator, test_error_baseline

def load_data():
    dataset = etl.load_dataset()
    data = dataset.data
    X = data[:,:-1]
    y = data[:,-1]
    labels_min_max_values = dataset.labels_min_max_values
    
    return X, y, labels_min_max_values

def main():
    POPULATION_SIZE = 100
    VERBOSE = 0

    seeds = [0, 42, 256]
    ic(seeds)
    num_seeds = len(seeds)
    
    X, y, labels_min_max_values = load_data()
    fitness_function = lambda y_true, y_pred, w: qerror_loss(y_true, y_pred, labels_min_max_values[0], labels_min_max_values[1])

    mean_error_symbolic = 0
    mean_error_baseline = 0

    for s in seeds:
        error_symbolic, error_baseline  = run_experiment(X, y, fitness_function, population_size=POPULATION_SIZE, verbose=VERBOSE, random_seed=s)
        mean_error_symbolic += error_symbolic
        mean_error_baseline += error_baseline

    mean_error_symbolic = mean_error_symbolic/num_seeds
    mean_error_baseline = mean_error_baseline/num_seeds
    
    ic(mean_error_baseline)
    ic(mean_error_symbolic)

if __name__ == '__main__':
    main()
    