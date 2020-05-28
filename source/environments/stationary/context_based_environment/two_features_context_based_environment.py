from source.environments.stationary.context_based_environment.__dependencies import *
from source.environments.stationary.context_based_environment.context_based_environment import ContextBasedEnvironment


class TwoFeaturesContextBasedEnvironment(ContextBasedEnvironment):
    def __init__(self, candidates, probabilities, features, class_probabilities):
        assert len(features) == 2
        classes = [
            [True, True],
            [True, False],
            [False, True],
            [False, False]]
        super().__init__(candidates, probabilities, classes, class_probabilities)
        self.features = features
        self.data = pd.DataFrame(
            columns=['userID', *features, 'candidate', 'reward'])
        self.data.loc[:, 'reward'] = self.data['reward'].astype(float)
        self.user = 0

    def simulate_round(self, candidate):
        reward = super().simulate_round(candidate)
        self.data.loc[self.user] = [self.user, *self.classes[self.current_class_index], candidate, reward]
        self.user += 1
        return reward

    def get_data(self):
        return self.data

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
