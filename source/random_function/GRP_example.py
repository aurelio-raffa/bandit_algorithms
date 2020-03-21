from sklearn.gaussian_process import GaussianProcessRegressor as GPReg
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as ConKer

from source.random_function.__dependencies import *
from source.random_function.randomfunction import RandomFunction


def main():
    # seed=7421304
    candidate_bids = np.linspace(0., 1., 30)
    iteration_num = 50
    noise_sigma = .08
    function = RandomFunction(range_x=(0, 1), scale_y=(0, 1), sigma=noise_sigma, seed=7421304)
    x_res = np.reshape(np.linspace(0, 1, 1000), (-1, 1))
    y_res = [function(x, true_value=True) for x in x_res]

    observed_candidates = []
    observed_scores = []
    for iteration in range(iteration_num):
        new_candidate = choice(candidate_bids)
        new_score = function(new_candidate)
        observed_candidates.append(new_candidate)
        observed_scores.append(new_score)

        theta = 1.
        lenscale = 1.
        kernel = ConKer(theta) * RBF(lenscale)
        regressor = GPReg(kernel=kernel, alpha=noise_sigma ** 2, n_restarts_optimizer=10, normalize_y=True)
        regressor.fit(np.reshape(observed_candidates, (-1, 1)), observed_scores)
        y_pred, sigma = regressor.predict(x_res, return_std=True)

        plt.figure(0)
        plt.axes(ylim=(0, 2.3))
        plt.grid()
        plt.plot(x_res, y_res)
        plt.plot(x_res, y_pred)
        plt.fill(
            np.concatenate([x_res, x_res[::-1]], axis=0),
            np.concatenate([y_pred + 1.97 * sigma, (y_pred - 1.97 * sigma)[::-1]], axis=0),
            alpha=.45)
        plt.scatter(x=observed_candidates, y=observed_scores)
        plt.legend(
            ['true function', 'Gaussian regression', 'envelope of confidence intervals', 'data points'],
            loc=2)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
