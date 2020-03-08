from source.conversion_rate.__dependencies import *


class Learner:
    def __init__(self, candidates):
        self.arms_number = len(candidates)
        self.candidates = candidates
        self.candidates_rewards = {arm: {} for arm in candidates}
        self.collected_rewards = np.array([])
        self.t = 0
        self.candidate_indices = dict(zip(candidates, range(len(candidates))))

    def update(self, candidate, reward):
        self.candidates_rewards[candidate][self.t] = reward
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.t += 1
