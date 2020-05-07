from source.environments.stationary.stationary_multicampaign_environment.__dependencies import *
from source.base.base_environment import BaseEnvironment


class MulticampaignEnvironment(BaseEnvironment):
    def __init__(self, subenvironments):
        super().__init__()
        self.subenvs = subenvironments
        self.n_campaigns = len(subenvironments)

    def simulate_round(self, candidate):
        rewards = np.zeros(self.n_campaigns)
        for cndt, idx in zip(candidate, range(self.n_campaigns)):
            rewards[idx] = self.subenvs[idx].simulate_round(cndt)
        return rewards
