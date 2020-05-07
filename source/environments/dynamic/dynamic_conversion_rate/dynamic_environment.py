from source.environments.dynamic.dynamic_conversion_rate.__dependencies import *
from source.environments.stationary.stationary_conversion_rate.environment import Environment


class DynamicEnvironment(Environment):
    def __init__(self, candidates, probabilities, horizon, seed=None):
        super().__init__(candidates, probabilities, seed)
        self.t = 0
        self.horizon = horizon

    def simulate_round(self, candidate):
        random_reward = np.random.binomial(1, self.candidate_data[candidate](self.t))
        self.t += 1
        return random_reward

    def show(self):
        plt.figure(0)
        plt.xlabel('t')
        for candidate, probability in self.candidate_data.items():
            plt.plot([probability(t) for t in range(self.horizon)])
        plt.legend(['candidate {}'.format(candidate) for candidate in self.candidate_data.keys()])
        plt.show()
        plt.close()
