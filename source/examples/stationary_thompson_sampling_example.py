from source.examples.__dependencies import *
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.learners.stationary.greedy.greedy_learner import GreedyLearner
from source.testing.tester import Tester


def main():
    # set up of variables
    print('{1}{0} Thompson Sampling vs. greedy algorithm {0}'.format('*'*10, '\n'*3))
    candidates = [1, 2, 3, 4]
    probabilities = [.5, .1, .1, .35]
    exploration_horizon = 300
    experiments = 300
    environment = Environment(candidates=candidates, probabilities=probabilities)
    gr_learner = GreedyLearner(candidates=candidates)
    ts_learner = ThompsonSamplingLearner(candidates=candidates)

    # tester initialization
    tester = Tester(
        environment=environment,
        learners=(gr_learner, ts_learner),
        optimal_expected_reward=.5,
        exploration_horizon=exploration_horizon,
        experiments=experiments)

    # running tests
    # np.random.seed(1234)
    # tester.run()
    # tester.show_results()

    # running tests (multiprocessing) - 50% gain in time
    np.random.seed(1234)
    tester.run(multiprocess=True)
    tester.show_results()


if __name__ == '__main__':
    main()
