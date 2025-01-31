from src.env.stn.rf.__dep import *
from src.env.stn.cr.env import Environment
from src.env.stn.rf.rf import RandomFunction


class RFEnvironment(Environment):
    def __init__(self, candidates, sigma, y_scale=1, seed=None):
        super().__init__(candidates, candidates, seed)
        self.function = RandomFunction(
            range_x=(np.min(candidates), np.max(candidates)),
            scale_y=(0, y_scale),
            sigma=sigma,
            seed=seed)

    def simulate_round(self, candidate):
        return self.function(candidate)

    def show(self):
        self.function.show()

    def optimum(self):
        return np.max([self.function(x, true_value=True) for x in self.candidate_data.keys()])
