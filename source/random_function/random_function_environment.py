from source.random_function.__dependencies import *
from source.conversion_rate.environment import Environment
from source.random_function.random_function import RandomFunction


class RandomFunctionEnvironment(Environment):
    def __init__(self, candidates, sigma, seed=None):
        super().__init__(candidates, candidates, seed)
        self.function = RandomFunction(
            range_x=(np.min(candidates), np.max(candidates)),
            scale_y=(0, 1),
            sigma=sigma,
            seed=seed)

    def simulate_round(self, candidate):
        return self.function(candidate)

    def show(self):
        self.function.show()

    def optimum(self):
        return np.max([self.function(x, true_value=True) for x in self.candidate_data.keys()])
