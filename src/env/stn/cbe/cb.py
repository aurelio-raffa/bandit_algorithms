from src.env.stn.cbe.__dep import *
from src.bse.bse_env import BaseEnvironment


class ContextEnvironment(BaseEnvironment):
    def __init__(self, candidates, probabilities, classes, class_probabilities):
        super().__init__()
        self.candidates = candidates
        self.candidates_indices = dict(zip(candidates, range(len(candidates))))
        self.probabilities = probabilities
        self.classes = classes
        self.class_probabilities = class_probabilities
        self.current_class_index = -1
        self.simulated_user = False

    def simulate_user(self):
        self.current_class_index = choice(range(len(self.classes)), p=self.class_probabilities)
        self.simulated_user = True
        return self.classes[self.current_class_index]

    def simulate_round(self, candidate):
        if not self.simulated_user:
            self.simulate_user()
        self.simulated_user = False
        return binomial(1, self.probabilities[self.current_class_index, self.candidates_indices[candidate]])
