from source.testing.__dependencies import *
from source.testing.simulation import Simulation


class Tester:
    def __init__(
            self,
            environment,
            learners,
            optimal_expected_reward,
            exploration_horizon,
            experiments=1,
            simulations=None):
        self.simulations = []
        self.optimal_expected_reward = optimal_expected_reward
        if simulations is None:
            for learner in learners:
                self.simulations.append(
                    Simulation(
                        environment=environment,
                        learner=learner,
                        exploration_horizon=exploration_horizon,
                        experiments=experiments))
        else:
            self.simulations = simulations
            assert len(learners) == len(simulations)

    def run(self):
        for simulation in self.simulations:
            simulation.run()

    def show_results(self, k=1):
        def shift_mean(x):
            return np.lib.stride_tricks.as_strided(x, (k, len(x) - k + 1), (x.itemsize, x.itemsize)).mean(axis=0)

        def plot_vector(vec, exploration_horizon):
            plt.plot(
                range(exploration_horizon - k + 1),
                shift_mean(np.mean(vec, axis=0)))
            plt.fill_between(
                range(exploration_horizon - k + 1),
                shift_mean(
                    np.mean(vec, axis=0) -
                    np.std(vec, axis=0)),
                shift_mean(
                    np.mean(vec, axis=0) +
                    np.std(vec, axis=0)),
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
                    plot_vector(simulation.collected_rewards, simulation.exploration_horizon)
            elif index == 1:
                for simulation in self.simulations:
                    plot_vector(
                        self.optimal_expected_reward - simulation.collected_rewards,
                        simulation.exploration_horizon)
            else:
                for simulation in self.simulations:
                    plt.plot(np.cumsum(np.mean(self.optimal_expected_reward - simulation.collected_rewards, axis=0)))
            plt.legend([str(simulation.learner) for simulation in self.simulations])
        plt.show()
        plt.close()
