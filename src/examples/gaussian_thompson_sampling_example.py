from src.lrn.stn.gau.__dep import *
from src.env.stn.rf.rfe import RFEnvironment
from src.lrn.stn.gdy.grl import GreedyLearner
from src.lrn.stn.gau.gts import GTSLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.lrn.dyn.gau.gps import SWGPTSLearner
from src.testing.tester import Tester


def main():
    # set up of variables
    print('{1}{0} Gaussian Thompson Sampling vs. greedy algorithm {0}'.format('*'*10, '\n'*3))
    candidates = np.linspace(0, 1, 20)
    sigma = .03
    theta = 1.
    lenscale = 1.5
    exploration_horizon = 20
    experiments = 20
    environment = RFEnvironment(candidates=candidates, sigma=sigma, seed=35446304)
    gr_learner = GreedyLearner(candidates=candidates)
    gt_learner = GTSLearner(candidates=candidates, sigma=sigma)
    gp_learner = GPTSLearner(
        candidates=candidates,
        sigma=sigma,
        theta=theta,
        lenscale=lenscale)
    gw_learner = SWGPTSLearner(
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
