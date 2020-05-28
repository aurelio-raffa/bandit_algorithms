import numpy as np

from source.environments.stationary.context_based_environment.two_features_context_based_environment import TwoFeaturesContextBasedEnvironment
from source.learners.stationary.two_features_context_based_learner.two_features_context_based_learner import TwoFeaturesContextBasedLearner
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.testing.simulation import Simulation
from source.testing.context_based_simulation import ContextBasedSimulation
from source.testing.tester import Tester


if __name__ == '__main__':
    # problem variables
    candidates = [10., 20., 30., 40.]
    # best_rate = .9
    # worst_rate = .1
    probabilities = np.array([
        [.9, .1, .1, .1],
        [.9, .1, .1, .1],
        [.1, .1, .9, .1],
        [.1, .1, .1, .9]])
    # probabilities = (best_rate - worst_rate) * np.identity(4) + worst_rate * np.ones((4, 4))
    class_probabilities = [.25, .25, .25, .25]
    # optimal_expected_reward = best_rate * .75 + worst_rate * .25
    optimal_expected_reward = .9
    features = ['male', 'over50']
    # simulation parameters
    exploration_horizon = 500
    experiments = 30
    delta = .1
    context_generation_period = 100
    # environment and learners
    environment = TwoFeaturesContextBasedEnvironment(candidates, probabilities, features, class_probabilities)
    basic_learner = ThompsonSamplingLearner(candidates)
    advanced_learner = TwoFeaturesContextBasedLearner(candidates, features, delta)
    # simulators and testers
    basic_simulator = Simulation(environment, basic_learner, exploration_horizon, experiments)
    advanced_simulator = ContextBasedSimulation(
        environment,
        advanced_learner,
        exploration_horizon,
        context_generation_period,
        experiments)
    tester = Tester(
        environment,
        learners=(advanced_learner, basic_learner),
        optimal_expected_reward=optimal_expected_reward,
        exploration_horizon=exploration_horizon,
        experiments=experiments,
        simulations=(advanced_simulator, basic_simulator))
    # execution and results
    tester.run()
    tester.show_results()
