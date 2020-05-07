from source.base.__dependencies import *


class BaseLearner:
    def __init__(self):
        self.collected_rewards = np.array([])
        self.t = 0

    def update(self, candidate, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.t += 1
