from source.environments.stationary.stationary_conversion_rate.__dependencies import *
from source.base.base_environment import BaseEnvironment


class Environment(BaseEnvironment):
    def __init__(self, candidates, probabilities, seed=None):
        super().__init__()
        assert len(candidates) == len(probabilities)
        self.arms_number = len(candidates)
        self.candidate_data = dict(zip(candidates, probabilities))
        if seed is not None:
            np.random.seed(seed)

    def simulate_round(self, candidate):
        return np.random.binomial(1, self.candidate_data[candidate])

    def show(self):
        plt.figure(0)
        plt.xlabel('candidates')
        plt.ylabel('probability')
        plt.bar(
            x=range(len(self.candidate_data.values())),
            height=self.candidate_data.values(),
            tick_label=['candidate {}'.format(candidate) for candidate in self.candidate_data.keys()])
        plt.show()
        plt.close()
