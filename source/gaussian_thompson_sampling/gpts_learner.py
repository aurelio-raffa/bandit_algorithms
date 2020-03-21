from source.gaussian_thompson_sampling.__dependencies import *
from source.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner


class GaussianProcessThompsonSamplingLearner(GaussianThompsonSamplingLearner):
    def __init__(self, candidates, sigma, theta, lenscale, optimizer_restarts=10):
        super().__init__(candidates, sigma)
        self.sigma = sigma
        self.kernel = ConKer(theta) * RBF(lenscale)
        self.optimizer_restarts=optimizer_restarts
        self.data = None
        self.regressor = None

    def update(self, candidate, reward):
        super().update(candidate, reward)
        if self.data is None:
            self.data = np.array([[candidate, reward]])
        else:
            self.data = np.concatenate([self.data, [[candidate, reward]]], axis=0)
        self.regressor = GPReg(
            kernel=self.kernel,
            alpha=self.sigma ** 2,
            n_restarts_optimizer=self.optimizer_restarts,
            normalize_y=True)
        self.regressor.fit(np.reshape(self.data[:, 0], (-1, 1)), self.data[:, 1])
        means, stdevs = self.regressor.predict(np.reshape(self.candidates, (-1, 1)), return_std=True)
        self.kernel = self.regressor.kernel_
        self.parameters[:, 0] = means
        self.parameters[:, 1] = stdevs

    def __str__(self):
        return 'Gaussian-Process Thompson Sampling Learner'
