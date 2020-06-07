from src.lrn.stn.gau.gps import GPTSLearner


class SWGPTSLearner(GPTSLearner):
    def __init__(self, candidates, window_size, sigma, theta, lenscale, optimizer_restarts=10):
        super().__init__(candidates, sigma, theta, lenscale, optimizer_restarts)
        self.window_size = window_size

    def update(self, candidate, reward):
        if self.data is not None and self.data.shape[0] == self.window_size:
            self.data = self.data[1:, :]
        super().update(candidate, reward)

    def __str__(self):
        return 'Sliding-Window Gaussian-Process Thompson Sampling Learner'
