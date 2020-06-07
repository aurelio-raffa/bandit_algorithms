from src.env.stn.rf.__dep import *
from src.env.stn.rf.rf import RandomFunction
from src.env.stn.nf.nf import NoisyFunction


def main():
    # seed=35446304
    iteration_num = 50
    noise_sigma = 5
    theta = 50
    lenscale = 50
    y_scale = 100
    x_scale = 100
    candidate_bids = np.linspace(0., x_scale, 30)
    # function = RandomFunction(range_x=(0, x_scale), scale_y=(0, y_scale), sigma=noise_sigma, seed=35446304)
    # function.show()
    function = NoisyFunction(nugget=10, slope=5, sill=80, sigma=noise_sigma)
    function.show((0, y_scale))
    x_res = np.reshape(np.linspace(0, x_scale, 1000), (-1, 1))
    y_res = [function(x, true_value=True) for x in x_res]

    observed_candidates = []
    observed_scores = []
    kernel = ConKer(theta) * RBF(lenscale)
    for iteration in range(iteration_num):
        new_candidate = choice(candidate_bids)
        new_score = function(new_candidate)
        observed_candidates.append(new_candidate)
        observed_scores.append(new_score)

        regressor = GPReg(kernel=kernel, alpha=noise_sigma ** 2, n_restarts_optimizer=10, normalize_y=True)
        regressor.fit(np.reshape(observed_candidates, (-1, 1)), observed_scores)
        kernel = regressor.kernel_
        y_pred, sigma = regressor.predict(x_res, return_std=True)

        plt.figure(0)
        # plt.axes(ylim=function.scale)
        plt.axes(ylim=(0, y_scale))
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
