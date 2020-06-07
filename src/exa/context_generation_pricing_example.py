import numpy as np

from src.env.stn.cbe.tfc import TFCEnvironment
from src.lrn.stn.tfc.tfl import TFCLearner
from src.lrn.stn.ts.sts import TSLearner
from src.tsg.sim import Simulation
from src.tsg.cxs import ContextSimulation
from src.tsg.tsr import Tester


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
    environment = TFCEnvironment(candidates, probabilities, features, class_probabilities)
    basic_learner = TSLearner(candidates)
    advanced_learner = TFCLearner(candidates, features, delta)
    # simulators and testers
    basic_simulator = Simulation(environment, basic_learner, exploration_horizon, experiments)
    advanced_simulator = ContextSimulation(
        environment,
        advanced_learner,
        exploration_horizon,
        context_generation_period,
        experiments)
    tester = Tester(
        environment,
        lrns=(advanced_learner, basic_learner),
        oer=optimal_expected_reward,
        horizon=exploration_horizon,
        exps=experiments,
        sims=(advanced_simulator, basic_simulator))
    # execution and results
    tester.run()
    tester.show_results()
