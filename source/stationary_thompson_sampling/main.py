from source.stationary_thompson_sampling.__dependencies import *
from source.conversion_rate.environment import Environment
from source.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.greedy.greedy_learner import GreedyLearner
from source.testing.simulation import Simulation


def main():
    # set up of variables
    print('{1}{0} Thompson Sampling algotithm {0}'.format('*'*10, '\n'*3))
    candidates = [1, 2, 3, 4]
    probabilities = [.5, .1, .1, .35]
    optimal_expected_reward = .5
    exploration_horizon = 300
    experiments = 5000
    environment = Environment(candidates=candidates, probabilities=probabilities)
    gr_learner = GreedyLearner(candidates=candidates)
    ts_learner = ThompsonSamplingLearner(candidates=candidates)

    # set up of simulations
    gr_sim = Simulation(
        environment=environment,
        learner=gr_learner,
        exploration_horizon=exploration_horizon,
        experiments=experiments)
    ts_sim = Simulation(
        environment=environment,
        learner=ts_learner,
        exploration_horizon=exploration_horizon,
        experiments=experiments)
    gr_sim.run()
    ts_sim.run()

    # plot reward over time, averaged on the number of experiments
    plt.figure(0)
    plt.xlabel('t')
    plt.ylabel('average reward')
    plt.plot(np.mean(gr_sim.collected_rewards, axis=0), 'r')
    plt.plot(np.mean(ts_sim.collected_rewards, axis=0), 'g')
    plt.legend(['greedy', 'Thompson sampling'])
    plt.show()

    # plot of regret over time, averaged on the number of experiments
    plt.figure(1)
    plt.xlabel('t')
    plt.ylabel('average regret')
    plt.plot(np.mean(optimal_expected_reward - gr_sim.collected_rewards, axis=0), 'r')
    plt.plot(np.mean(optimal_expected_reward - ts_sim.collected_rewards, axis=0), 'g')
    plt.legend(['greedy', 'Thompson sampling'])
    plt.show()

    # plot of cumulative average regret over time, averaged on the number of experiments
    plt.figure(2)
    plt.xlabel('t')
    plt.ylabel('cumulative average regret')
    plt.plot(np.cumsum(np.mean(optimal_expected_reward - gr_sim.collected_rewards, axis=0)), 'r')
    plt.plot(np.cumsum(np.mean(optimal_expected_reward - ts_sim.collected_rewards, axis=0)), 'g')
    plt.legend(['greedy', 'Thompson sampling'])
    plt.show()


if __name__ == '__main__':
    main()
