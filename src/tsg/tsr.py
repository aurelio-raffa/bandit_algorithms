from src.tsg.__dep import *
from src.tsg.sim import Simulation


class Tester:
    def __init__(self, env=None, lrns=None, oer=1., horizon=0, exps=1, sims=None):
        assert (env is not None and lrns is not None) or sims is not None
        self.simulations = []
        self.optimal_expected_reward = oer
        if sims is None:
            for learner in lrns:
                self.simulations.append(
                    Simulation(
                        env=env,
                        lrn=learner,
                        horizon=horizon,
                        exps=exps))
        else:
            self.simulations = sims

    def run(self, multiprocess=False, seed=None):
        if multiprocess:
            warnings.warn('this feature is currently in its testing phase!', RuntimeWarning)
        if seed is not None:
            np.random.seed(seed)
        for simulation in self.simulations:
            simulation.run(multiprocess=multiprocess, seed=seed)

    def show_results(self, k=1):
        def shift_mean(x):
            return np.lib.stride_tricks.as_strided(x, (k, len(x) - k + 1), (x.itemsize, x.itemsize)).mean(axis=0)

        def plot_mean(mat, exploration_horizon):
            plt.plot(
                range(exploration_horizon - k + 1),
                shift_mean(np.mean(mat, axis=0)))
            plt.fill_between(
                range(exploration_horizon - k + 1),
                shift_mean(
                    np.mean(mat, axis=0) -
                    np.std(mat, axis=0)),
                shift_mean(
                    np.mean(mat, axis=0) +
                    np.std(mat, axis=0)),
                interpolate=True, alpha=0.25)

        y_labels = [
            'average reward',           # plot reward over time, averaged on the number of experiments
            'average regret',           # plot of regret over time, averaged on the number of experiments
            'cumulative regret']        # plot of cum. average regret over time, averaged on the number of experiments
        sns.set(style='darkgrid')
        for index in range(3):
            plt.subplot(3, 1, index+1)
            plt.xlabel('t')
            plt.ylabel(y_labels[index])
            if index == 0:
                for simulation in self.simulations:
                    plot_mean(
                        simulation.collected_rewards,
                        simulation.exploration_horizon)
            elif index == 1:
                for simulation in self.simulations:
                    plot_mean(
                        self.optimal_expected_reward - simulation.collected_rewards,
                        simulation.exploration_horizon)
            else:
                for simulation in self.simulations:
                    plt.plot(np.cumsum(np.mean(self.optimal_expected_reward - simulation.collected_rewards, axis=0)))
            plt.legend([str(simulation.learner) for simulation in self.simulations])
        plt.show()
        plt.close()
