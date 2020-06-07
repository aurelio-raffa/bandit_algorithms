from src.env.stn.mc.__dep import *
from src.env.stn.mc.mce import MCEnvironment
from src.env.stn.nf.nfe import NFEnvironment


class MCNFEnvironment(MCEnvironment):
    def __init__(self, candidates, nuggets, slopes, sills, sigmas):
        assert len(nuggets) == len(slopes) == len(sills) == len(sigmas)
        n_campaigns = len(nuggets)
        subenvs = [
            NFEnvironment(candidates, nuggets[it], slopes[it], sills[it], sigmas[it])
            for it in range(n_campaigns)]
        super().__init__(subenvs)
