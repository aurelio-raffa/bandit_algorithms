from source.conversion_rate.__dependencies import *


class Environment:
    def __init__(self, candidates, probabilities, seed=None):
        assert len(candidates) == len(probabilities)
        self.arms_number = len(candidates)
        self.__candidate_data = dict(zip(candidates, probabilities))
        if seed is not None:
            np.random.seed(seed)

    def simulate_round(self, candidate):
        random_reward = np.random.binomial(1, self.__candidate_data[candidate])
        return random_reward
