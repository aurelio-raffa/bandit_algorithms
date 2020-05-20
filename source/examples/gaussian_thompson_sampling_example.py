from source.learners.stationary.gaussian_thompson_sampling.__dependencies import *
from source.environments.stationary.random_function.random_function_environment import RandomFunctionEnvironment
from source.learners.stationary.greedy.greedy_learner import GreedyLearner
from source.learners.stationary.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner
from source.learners.stationary.gaussian_thompson_sampling.gpts_learner import GaussianProcessThompsonSamplingLearner
from source.learners.dynamic.dynamic_gaussian_processes.gptssw_learner import SlidingWindowGaussianProcessThompsonSamplingLearner
from source.testing.tester import Tester


def main():
    # set up of variables
    print('{1}{0} Gaussian Thompson Sampling vs. greedy algorithm {0}'.format('*'*10, '\n'*3))
    candidates = np.linspace(0, 1, 20)
    sigma = .03
    theta = 1.
    lenscale = 1.5
    exploration_horizon = 20
    experiments = 20
    environment = RandomFunctionEnvironment(candidates=candidates, sigma=sigma, seed=35446304)
    gr_learner = GreedyLearner(candidates=candidates)
    gt_learner = GaussianThompsonSamplingLearner(candidates=candidates, sigma=sigma)
    gp_learner = GaussianProcessThompsonSamplingLearner(
        candidates=candidates,
        sigma=sigma,
        theta=theta,
        lenscale=lenscale)
    gw_learner = SlidingWindowGaussianProcessThompsonSamplingLearner(
        candidates=candidates,
        window_size=10,
        sigma=sigma,
        theta=theta,
        lenscale=lenscale)

    # tester initialization
    tester = Tester(
        environment=environment,
        learners=(gr_learner, gt_learner, gw_learner, gp_learner),
        optimal_expected_reward=environment.optimum(),
        exploration_horizon=exploration_horizon,
        experiments=experiments)

    # running tests
    tester.run()
    tester.show_results()


if __name__ == '__main__':
    main()
