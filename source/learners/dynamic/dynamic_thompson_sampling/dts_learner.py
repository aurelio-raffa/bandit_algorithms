from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner


class SlidingWindowThompsonSamplingLearner(ThompsonSamplingLearner):
    def __init__(self, candidates, memory):
        super().__init__(candidates)
        self.memory = memory

    def update(self, candidate, reward):
        super().update(candidate, reward)
        if len(self.candidates_rewards[candidate].values()) >= self.memory:
            exiting_value =     \
                self.candidates_rewards[candidate][self.t - self.memory] \
                if (self.t - self.memory) in self.candidates_rewards[candidate].keys() else 0
            correction = [[exiting_value, 1-exiting_value]]
            fallback = [[1, 1]]
            self.beta_parameters[self.candidate_indices[candidate], :] = \
                np.max(
                    np.concatenate(
                        [fallback, self.beta_parameters[self.candidate_indices[candidate], :]-correction],
                        axis=0),
                    axis=0)

    def __str__(self):
        return 'Sliding Window Thompson Sampling Learner'
