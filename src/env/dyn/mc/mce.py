from src.env.dyn.mc.__dep import *
from src.env.stn.mc.mnf import MCNFEnvironment


class DMCNFEnvironment(MCNFEnvironment):
    def __init__(self, candidates, nuggets, slopes, sills, sigmas, abrupt_change_times):
        assert nuggets.shape[0] == slopes.shape[0] == sills.shape[0] == sigmas.shape[0] == len(abrupt_change_times) + 1
        assert nuggets.shape[1] == slopes.shape[1] == sills.shape[1] == sigmas.shape[1]
        self.candidates = candidates
        self.nuggets = nuggets
        self.slopes = slopes
        self.sills = sills
        self.sigmas = sigmas
        self.t = -1
        self.phase = 0
        self.change_times = abrupt_change_times
        super().__init__(candidates, nuggets[0, :], slopes[0, :], sills[0, :], sigmas[0, :])

    def simulate_round(self, candidate):
        self.t += 1
        if self.t in self.change_times:
            self.phase += 1
            super().__init__(
                self.candidates,
                self.nuggets[self.phase, :],
                self.slopes[self.phase, :],
                self.sills[self.phase, :],
                self.sigmas[self.phase, :])
        return super().simulate_round(candidate)
