from src.lrn.stn.gdy.__dep import *
from src.lrn.stn.sln.lrn import Learner


class GreedyLearner(Learner):
    def __init__(self, candidates):
        super().__init__(candidates)
        self.__expected_rewards = np.zeros(self.arms_number)

    def select_arm(self):
        if self.t < self.arms_number:
            index = self.t
        else:
            ind_lis = np.flatnonzero(self.__expected_rewards == np.max(self.__expected_rewards))
            index = np.random.choice(ind_lis)
        return self.candidates[index]

    def update(self, candidate, reward):
        super().update(candidate, reward)
        n = len(self.candidates_rewards[candidate].values())
        self.__expected_rewards[self.candidate_indices[candidate]] = \
            (self.__expected_rewards[self.candidate_indices[candidate]] * (n - 1) + reward) / n

    def sample_candidates(self):
        return self.__expected_rewards

    def __str__(self):
        return 'Greedy Learner'
