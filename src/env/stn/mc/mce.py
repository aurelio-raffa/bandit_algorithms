from src.env.stn.mc.__dep import *
from src.bse.bse_env import BaseEnvironment


class MCEnvironment(BaseEnvironment):
    def __init__(self, subenvironments):
        super().__init__()
        self.subenvs = subenvironments
        self.n_campaigns = len(subenvironments)

    def simulate_round(self, candidate):
        rewards = np.zeros(self.n_campaigns)
        for cndt, idx in zip(candidate, range(self.n_campaigns)):
            rewards[idx] = self.subenvs[idx].simulate_round(cndt)
        return rewards
