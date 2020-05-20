from source.environments.stationary.noisy_function.__dependencies import *
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.environments.stationary.noisy_function.noisy_function import NoisyFunction


class NoisyFunctionEnvironment(Environment):
    def __init__(self, candidates, nugget, slope, sill, sigma, seed=None):
        super().__init__(candidates, candidates, seed)
        self.function = NoisyFunction(nugget, slope, sill, sigma)
        self.range = (min(candidates), max(candidates))

    def simulate_round(self, candidate):
        return self.function(candidate)

    def show(self):
        self.function.show(x_range=self.range)

    def optimum(self):
        return np.max([self.function(x, true_value=True) for x in self.candidate_data.keys()])
