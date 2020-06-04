from source.learners.stationary.average_reward_learner.__dependencies import *
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner


class AverageRewardThompsonSamplingLearner(ThompsonSamplingLearner):
    def __init__(self, candidates):
        super().__init__(candidates)

    def select_arm(self):
        index = np.argmax(self.sample_candidates() * np.array(self.candidates))
        return self.candidates[index]

    def __str__(self):
        return 'Average-Reward Thompson Sampling Learner'
