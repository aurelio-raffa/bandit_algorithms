from src.lrn.stn.tfc.__dep import *
from src.lrn.stn.ar.art \
    import AverageRewardTSL
from src.ctx.con_gen import ContextGenerator
from src.bse.bse_lrn import BaseLearner


class TFCLearner(BaseLearner):
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
                (False, False)}): AverageRewardTSL(candidates)}
        self.current_class = None
        self.current_features = None
        self.data = None
        self.user = 0

    def generate_context(self):
        self.context_generator.train(self.data, self.delta, deepcopy(self.features))
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
                self.learners_by_class[key] = AverageRewardTSL(self.candidates)

    def select_class(self, current_features):
        self.current_features = current_features
        for fset in self.learners_by_class.keys():
            if tuple(current_features) in fset:
                self.current_class = fset
                break

    def select_arm(self):
        return self.learners_by_class[self.current_class].select_arm()

    def update(self, candidate, reward):
        super().update(candidate, reward * candidate)
        if self.data is None:
            self.data = pd.DataFrame(
                columns=['userID', *self.features, 'candidate', 'reward'])
            self.data.loc[:, 'reward'] = self.data['reward'].astype(float)
        self.data.loc[self.user] = [self.user, *self.current_features, candidate, reward]
        self.learners_by_class[self.current_class].update(candidate, reward)
        self.user += 1

    def __str__(self):
        return 'Two-Features Context-Based Learner'
