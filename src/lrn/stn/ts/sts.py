from src.lrn.stn.ts.__dep import *
from src.lrn.stn.sln.lrn import Learner


class TSLearner(Learner):
    def __init__(self, candidates):
        super().__init__(candidates)
        self.beta_parameters = np.ones(shape=(self.arms_number, 2))

    def select_arm(self):
        index = np.argmax(
            np.random.beta(
                self.beta_parameters[:, 0],
                self.beta_parameters[:, 1]))
        return self.candidates[index]

    def update(self, candidate, reward):
        super().update(candidate, reward)
        self.beta_parameters[self.candidate_indices[candidate], :] += (reward, 1 - reward)

    def sample_candidates(self):
        return np.random.beta(
            self.beta_parameters[:, 0],
            self.beta_parameters[:, 1])

    def __str__(self):
        return 'Thompson Sampling Learner'
