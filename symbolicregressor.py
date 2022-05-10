import random
import symbolic.etl as etl
from icecream import ic
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicRegressor

def q_error(y_true, y_hat, w):
    error = sum([max(tr, pr)/min(tr, pr) for tr, pr in zip(y_true, y_hat)])
        
    error = error/len(y_true)
    
    return error

def run_experiment(X, y, population_size, verbose, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed)
    
    baseline_regressor = RandomForestRegressor(
        n_estimators=population_size, 
        n_jobs=-1)

    baseline_regressor.fit(X_train, y_train)

    ic(metrics.mean_squared_error(y_train, baseline_regressor.predict(X_train)))
    ic(metrics.mean_squared_error(y_test, baseline_regressor.predict(X_test)))

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
        metric='mse',
        function_set=['add', 'sub', 'mul', 'div', 'sub', 'sqrt', 'cos', 'sin'],
        const_range=None,
        low_memory=True,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_seed,)
    
    symbolic_estimator.fit(X_train, y_train)
    
    test_error_symbolic_estimator = metrics.mean_squared_error(y_test, symbolic_estimator.predict(X_test))
    best_program = str(symbolic_estimator._program)

    ic(metrics.mean_squared_error(y_train, symbolic_estimator.predict(X_train)))
    ic(test_error_symbolic_estimator)
    ic(best_program)

    return test_error_symbolic_estimator, best_program

def load_data():
    dataset = etl.load_dataset()
    X = dataset[:,:-1]
    y = dataset[:,-1]

    return X, y

def main():
    NUMBER_OF_SEEDS = 10
    RANGE_OF_SEEDS = (0, 100)
    seeds = [random.randint(RANGE_OF_SEEDS[0], RANGE_OF_SEEDS[1]) for i in range(NUMBER_OF_SEEDS)]

    X, y = load_data()
    mean_error = 0

    for s in seeds:
        error, best_program = run_experiment(X, y, population_size=200, verbose=0, random_seed=s)
        mean_error += error

    mean_error = mean_error/NUMBER_OF_SEEDS
    ic(mean_error)

if __name__ == '__main__':
    main()
    