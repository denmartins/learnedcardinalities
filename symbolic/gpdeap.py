import operator
import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def protectedDiv(left, right):
    if right == 0:
        return 1
    else:
        return left / right
    
class SymbRegressorDEAP:
    _X_train = None
    _y_train = None
    _toolbox = None
    _program = None

    def __init__(self, loss_function, population_size):
        self.loss_function = loss_function
        self.population_size = population_size

    def evaluate_individual(self, individual):
        y_hat = self.get_prediction(self._X_train, individual)
        return self.loss_function(self._y_train, y_hat),

    def get_prediction(self, X, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        y_hat = numpy.array([func(*x) for x in X])
        return y_hat

    def predict(self, X):
        return self.get_prediction(X, self._program)

    def construct(self):
        pset = gp.PrimitiveSet("MAIN", self._X_train.shape[1])
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", self.evaluate_individual)
        #toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.3)    
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))

        return toolbox

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train

        self.toolbox = self.construct()

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
    
        pop, log = algorithms.eaSimple(
            population=pop,
            ngen=20,
            toolbox=self.toolbox,
            cxpb=0.8,
            mutpb=0.2,
            stats=stats, 
            halloffame=hof, 
            verbose=True,)

        self._program =  hof[0]
