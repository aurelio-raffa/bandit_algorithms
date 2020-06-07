from src.examples.__dependencies import *
from src.env.stn.cr.env import Environment
from src.lrn.stn.ts.sts import TSLearner
from src.lrn.stn.gdy.grl import GreedyLearner
from src.testing.tester import Tester


def main():
    # set up of variables
    print('{1}{0} Thompson Sampling vs. greedy algorithm {0}'.format('*'*10, '\n'*3))
    candidates = [1, 2, 3, 4]
    probabilities = [.5, .1, .1, .35]
    exploration_horizon = 300
    experiments = 300
    environment = Environment(candidates=candidates, probabilities=probabilities)
    gr_learner = GreedyLearner(candidates=candidates)
    ts_learner = TSLearner(candidates=candidates)

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
