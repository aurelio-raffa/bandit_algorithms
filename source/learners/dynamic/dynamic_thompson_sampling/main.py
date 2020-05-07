from source.environments.dynamic.dynamic_conversion_rate import DynamicEnvironment
from source.learners.stationary.greedy.greedy_learner import GreedyLearner
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.learners.dynamic.dynamic_thompson_sampling import SlidingWindowThompsonSamplingLearner
from source.testing.tester import Tester


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
    optimal_expected_reward = 1
    horizon = 2000
    window = 50
    experiments = 100
    environment = DynamicEnvironment(candidates=candidates, probabilities=probabilities, horizon=horizon)
    gr_learner = GreedyLearner(candidates=candidates)
    ts_learner = ThompsonSamplingLearner(candidates=candidates)
    dt_learner = SlidingWindowThompsonSamplingLearner(candidates=candidates, memory=window)

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
