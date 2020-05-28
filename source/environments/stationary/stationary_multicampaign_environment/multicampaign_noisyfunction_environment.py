from source.environments.stationary.stationary_multicampaign_environment.__dependencies import *
from source.environments.stationary.stationary_multicampaign_environment.multicampaign_environment import MulticampaignEnvironment
from source.environments.stationary.noisy_function.noisy_function_environment import NoisyFunctionEnvironment


class MulticampaignNoisyfunctionEnvironment(MulticampaignEnvironment):
    def __init__(self, candidates, nuggets, slopes, sills, sigmas):
        assert len(nuggets) == len(slopes) == len(sills) == len(sigmas)
        n_campaigns = len(nuggets)
        subenvs = [
            NoisyFunctionEnvironment(candidates, nuggets[it], slopes[it], sills[it], sigmas[it])
            for it in range(n_campaigns)]
        super().__init__(subenvs)
