from src.env.stn.cbe.__dep import *
from src.env.stn.cbe.cb import ContextEnvironment


class TFCEnvironment(ContextEnvironment):
    def __init__(self, candidates, probabilities, features, class_probabilities):
        assert len(features) == 2
        classes = [
            [True, True],
            [True, False],
            [False, True],
            [False, False]]
        super().__init__(candidates, probabilities, classes, class_probabilities)
        self.features = features

    def simulate_round(self, candidate):
        reward = super().simulate_round(candidate)
        return reward

    def show(self):
        plt.figure(0)
        plt.xlabel('candidates')
        plt.ylabel('probability')
        for it in range(len(self.classes)):
            plt.bar(
                x=range(self.probabilities.shape[1]),
                height=self.probabilities[it, :],
                tick_label=['candidate {}'.format(candidate) for candidate in self.candidates])
        plt.legend(['class {}'.format(c) for c in self.classes])
        plt.show()
        plt.close()
