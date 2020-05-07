from source.learners.stationary.stationary_learner.__dependencies import *
from source.base.base_learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self, candidates):
        super().__init__()
        self.candidates = candidates
        self.arms_number = len(candidates)
        self.candidates_rewards = {arm: {} for arm in candidates}
        self.candidate_indices = dict(zip(candidates, range(len(candidates))))

    def update(self, candidate, reward):
        self.candidates_rewards[candidate][self.t] = reward
        super().update(candidate, reward)

    def sample_candidates(self):
        pass
