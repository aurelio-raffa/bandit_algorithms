from src.lrn.stn.ar.__dep import *
from src.lrn.stn.ts.sts import TSLearner
from src.lrn.stn.sln.lrn import Learner


class AverageRewardTSL(TSLearner):
    def __init__(self, candidates):
        super().__init__(candidates)

    def select_arm(self):
        index = np.argmax(self.sample_candidates() * np.array(self.candidates))
        return self.candidates[index]

    def update(self, candidate, reward):
        self.beta_parameters[self.candidate_indices[candidate], :] += (reward, 1 - reward)
        self.candidates_rewards[candidate][self.t] = reward * candidate
        self.collected_rewards = np.append(self.collected_rewards, reward * candidate)
        self.t += 1

    def __str__(self):
        return 'Average-Reward Thompson Sampling Learner'
