from src.project.__dependencies import *
from src.env.stn.cbe.tfc \
    import TFCEnvironment
from src.lrn.stn.tfc.tfl \
    import TFCLearner
from src.lrn.stn.ar.art \
    import AverageRewardTSL
from src.env.stn.nf.nf import NoisyFunction
from src.testing.simulation import Simulation
from src.testing.context_based_simulation import ContextBasedSimulation
from src.testing.tester import Tester


if __name__ == '__main__':
    # problem parameters
    candidates = [.99, 1.79, 2.99, 3.49]
    features = ['male', 'over45']
    probabilities = np.array([
        [.1, .1, .1, .9],
        [.1, .1, .1, .9],
        [.1, .1, .9, .1],
        [.1, .9, .1, .1]])
    n_campaigns = 3
    optimal_configuration = [25, 25, 50]
    sigma_true = 5
    nuggets = [5, 10, 15]
    slopes = [5, 3, 2]
    sills = [50, 60, 100]
    sigmas = [sigma_true] * n_campaigns
    average_clicks = np.array([
        NoisyFunction(nug, slope, sill, sigma)(budget, true_value=True)
        for nug, slope, sill, sigma, budget
        in zip(nuggets, slopes, sills, sigmas, optimal_configuration)])
    total_clics = np.sum(average_clicks)
    average_clicks = average_clicks/total_clics
    class_probabilities = [average_clicks[0]/2, average_clicks[0]/2, average_clicks[1], average_clicks[2]]
    optimal_expected_reward = np.sum(
        np.array(class_probabilities) * np.max(probabilities * np.array([candidates] * 4), axis=1))
    print('optimal reward: {}'.format(optimal_expected_reward))

    # environment
    environment = TFCEnvironment(candidates, probabilities, features, class_probabilities)

    # simulation parameters
    exploration_horizon = 10000
    experiments = 25
    delta = .2
    context_generation_period = 500

    # learners
    basic_learner = AverageRewardTSL(candidates)
    advanced_learner = TFCLearner(candidates, features, delta)

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
    tester.run(multiprocess=True)
    tester.show_results(k=30)
