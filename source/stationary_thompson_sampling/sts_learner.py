from source.stationary_thompson_sampling.__dependencies import *
from source.conversion_rate.learner import Learner


class ThompsonSamplingLearner(Learner):
    def __init__(self, candidates):
        super().__init__(candidates)
        self.__beta_parameters = np.ones(shape=(self.arms_number, 2))

    def select_arm(self):
        index = np.argmax(
            np.random.beta(
                self.__beta_parameters[:, 0],
                self.__beta_parameters[:, 1]))
        return self.candidates[index]

    def update(self, candidate, reward):
        super().update(candidate, reward)
        self.__beta_parameters[self.candidate_indices[candidate], :] += (reward, 1 - reward)
