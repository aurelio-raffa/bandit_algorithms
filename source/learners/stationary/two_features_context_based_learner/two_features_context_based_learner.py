from source.learners.stationary.two_features_context_based_learner.__dependencies import *
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.context.context_generator import ContextGenerator
from source.base.base_learner import BaseLearner


class TwoFeaturesContextBasedLearner(BaseLearner):
    def __init__(self, candidates, features, delta):
        assert len(features) == 2
        super().__init__()
        self.context_generator = ContextGenerator()
        self.context_generator.initialize_two_features(features)
        self.candidates = candidates
        self.features = features
        self.delta = delta
        self.learners_by_class = {
            frozenset({
                (True, True),
                (True, False),
                (False, True),
                (False, False)}): ThompsonSamplingLearner(candidates)}
        self.current_class = None

    def generate_context(self, data):
        self.context_generator.train(data, self.delta, deepcopy(self.features))
        temp_list = [set() for _ in range(self.context_generator.number_of_classes())]
        for key, val in self.context_generator.model.items():
            temp_list[val].add(key)
        temp_list = [frozenset(el) for el in temp_list]
        original_keys = list(self.learners_by_class.keys())
        for fset in original_keys:
            if fset not in temp_list:
                del self.learners_by_class[fset]
        for key in temp_list:
            if key not in self.learners_by_class.keys():
                self.learners_by_class[key] = ThompsonSamplingLearner(self.candidates)

    def select_class(self, current_features):
        for fset in self.learners_by_class.keys():
            if tuple(current_features) in fset:
                self.current_class = fset
                break

    def select_arm(self):
        return self.learners_by_class[self.current_class].select_arm()

    def update(self, candidate, reward):
        super().update(candidate, reward)
        self.learners_by_class[self.current_class].update(candidate, reward)

    def __str__(self):
        return 'Two-Features Context-Based Learner'
