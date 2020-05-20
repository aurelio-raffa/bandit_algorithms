from source.learners.stationary.gaussian_thompson_sampling.__dependencies import *
from source.learners.stationary.stationary_learner.learner import Learner


class GaussianThompsonSamplingLearner(Learner):
    def __init__(self, candidates, sigma):
        super().__init__(candidates)
        self.parameters = np.zeros(shape=(self.arms_number, 2))
        self.parameters[:, 1] += sigma

    def select_arm(self):
        index = np.argmax(
            np.random.normal(
                self.parameters[:, 0],
                self.parameters[:, 1]))
        return self.candidates[index]

    def update(self, candidate, reward):
        index = self.candidate_indices[candidate]
        n = len(self.candidates_rewards[candidate].values())
        sum_of_squares = ((n - 1) * self.parameters[index, 1]**2 + n * self.parameters[index, 0]**2) \
            if n > 1 else 0
        # print(n, sum_of_squares, self.parameters[index, :])
        super().update(candidate, reward)
        self.parameters[index, 0] = (self.parameters[index, 0] * n + reward) / (n + 1)
        if n >= 1:
            self.parameters[index, 1] = \
                np.std(list(self.candidates_rewards[candidate].values())) \
                if n == 1 else \
                np.sqrt((sum_of_squares + reward ** 2 - (n + 1) * self.parameters[index, 0] ** 2) / n)

    def sample_candidates(self):
        return np.random.normal(
            self.parameters[:, 0],
            self.parameters[:, 1])

    def __str__(self):
        return 'Gaussian Thompson Sampling Learner'
