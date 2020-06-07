from src.pjt.__dep import *
from src.env.stn.cbe.tfc import TFCEnvironment
from src.lrn.stn.tfc.tfl import TFCLearner
from src.lrn.stn.ar.art import AverageRewardTSL
from src.env.stn.nf.nf import NoisyFunction
from src.tsg.sim import Simulation
from src.tsg.cxs import ContextSimulation
from src.tsg.tsr import Tester


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

    # data - point 4
    optimal_expected_reward_p4 = np.max(
        np.dot((probabilities * np.array([candidates] * 4)).T, np.array(class_probabilities)))
    print('optimal reward (no context generation): {}'.format(optimal_expected_reward_p4))

    # data - point 5
    optimal_expected_reward_p5 = np.sum(
        np.array(class_probabilities) * np.max(probabilities * np.array([candidates] * 4), axis=1))
    print('optimal reward: {}'.format(optimal_expected_reward_p5))

    # environment
    environment = TFCEnvironment(candidates, probabilities, features, class_probabilities)

    # simulation parameters - point 4
    np.random.seed(12345)
    exploration_horizon_p4 = 5000
    experiments_p4 = 100

    # simulation parameters - point 5
    exploration_horizon_p5 = 10000
    experiments_p5 = 25
    delta = .2
    context_generation_period = 500

    # learners
    learner_p4 = AverageRewardTSL(candidates)
    learner_p5 = TFCLearner(candidates, features, delta)

    # simulator - point 4
    simulator_p4 = Simulation(environment, learner_p4, exploration_horizon_p4, experiments_p4)
    tester_p4 = Tester(
        environment,
        lrns=(learner_p4,),
        oer=optimal_expected_reward_p4,
        horizon=exploration_horizon_p4,
        exps=experiments_p4)

    # execution and results - point 4
    tester_p4.run(multiprocess=True)
    tester_p4.show_results()

    # simulators and testers
    simulator_p5 = ContextSimulation(
        environment,
        learner_p5,
        exploration_horizon_p5,
        context_generation_period,
        experiments_p5)
    tester_p5 = Tester(
        environment,
        lrns=(learner_p5, learner_p4),
        oer=optimal_expected_reward_p5,
        horizon=exploration_horizon_p5,
        exps=experiments_p5,
        sims=(simulator_p5, simulator_p4))

    # execution and results
    tester_p5.run(multiprocess=True)
    tester_p5.show_results(k=30)

