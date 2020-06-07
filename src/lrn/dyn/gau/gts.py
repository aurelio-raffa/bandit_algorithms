from src.lrn.stn.gau.gts import GTSLearner


class SWGTSLearner(GTSLearner):
    def __init__(self, candidates, sigma, window_size):
        super().__init__(candidates, sigma)
        self.window_size = window_size

    def update(self, candidate, reward):
        raise NotImplementedError

    def __str__(self):
        return 'Sliding-Window Gaussian Thompson Sampling Learner'
