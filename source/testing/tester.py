from source.testing.__dependencies import *
from source.testing.simulation import Simulation


class Tester:
    def __init__(
            self,
            environment,
            learners,
            optimal_expected_reward,
            exploration_horizon,
            experiments=1):
        self.simulations = []
        self.optimal_expected_reward = optimal_expected_reward
        for learner in learners:
            self.simulations.append(
                Simulation(
                    environment=environment,
                    learner=learner,
                    exploration_horizon=exploration_horizon,
                    experiments=experiments))

    def run(self):
        for simulation in self.simulations:
            simulation.run()

    def show_results(self):
        plt.figure(0)
        y_labels = [
            'average reward',           # plot reward over time, averaged on the number of experiments
            'average regret',           # plot of regret over time, averaged on the number of experiments
            'cumulative regret']        # plot of cum. average regret over time, averaged on the number of experiments
        for index in range(3):
            plt.subplot(3, 1, index+1)
            plt.xlabel('t')
            plt.ylabel(y_labels[index])
            for simulation in self.simulations:
                if index == 0:
                    plt.plot(np.mean(simulation.collected_rewards, axis=0))
                elif index == 1:
                    plt.plot(np.mean(self.optimal_expected_reward - simulation.collected_rewards, axis=0))
                else:
                    plt.plot(np.cumsum(np.mean(self.optimal_expected_reward - simulation.collected_rewards, axis=0)))
            plt.legend([str(simulation.learner) for simulation in self.simulations])
        plt.show()
        plt.close()
