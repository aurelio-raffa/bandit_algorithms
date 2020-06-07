from src.examples.__dependencies import *
from src.env.dyn.cr.den import DynamicEnvironment
from src.lrn.stn.gdy.grl import GreedyLearner
from src.lrn.stn.ts.sts import TSLearner
from src.lrn.dyn.ts.swt import SWTSLearner
from src.testing.tester import Tester


def main():
    # set up of variables
    print('{1}{0} Sliding Window Thompson Sampling {0}'.format('*' * 10, '\n' * 3))

    def p1(t):
        if t <= 500:
            return .1
        elif t <= 1200:
            return .4
        else:
            return .6

    def p2(t):
        if t <= 750:
            return .65
        else:
            return .0

    def p3(t):
        if t <= 1000:
            return .4
        elif t <= 1500:
            return .2
        else:
            return .8

    def p4(t):
        return t/(100 + t)

    candidates = [1, 2, 3, 4]
    probabilities = [p1, p2, p3, p4]
    horizon = 3000
    optimal_expected_reward = np.array([
        max([probability(t) for probability in probabilities])
        for t in range(horizon)])
    window = 100
    experiments = 50
    environment = DynamicEnvironment(candidates=candidates, probabilities=probabilities, horizon=horizon)
    gr_learner = GreedyLearner(candidates=candidates)
    ts_learner = TSLearner(candidates=candidates)
    dt_learner = SWTSLearner(candidates=candidates, memory=window)

    # tester initialization
    tester = Tester(
        environment=environment,
        learners=(gr_learner, ts_learner, dt_learner),
        optimal_expected_reward=optimal_expected_reward,
        exploration_horizon=horizon,
        experiments=experiments)

    # running tests
    tester.run()
    environment.show()
    tester.show_results()


if __name__ == '__main__':
    main()
