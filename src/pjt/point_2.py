from src.pjt.__dep import *
from src.env.stn.mc.mnf import MCNFEnvironment
from src.lrn.stn.bc.bcl import BudgetLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.lrn.dyn.gau.gps import SWGPTSLearner
from src.env.stn.mc.mnf import MCNFEnvironment
from src.env.stn.nf.nf import NoisyFunction
from src.tsg.sim import Simulation
from src.tsg.tsr import Tester
from src.opt.bop import budget_optimizer


if __name__ == '__main__':
    # problem parameters
    budgets = np.linspace(start=0, stop=100, num=5)
    n_campaigns = 3
    sigma_true = 5
    nuggets = [5, 10, 15]
    slopes = [5, 3, 2]
    sills = [50, 60, 100]
    sigmas = [sigma_true] * n_campaigns

    # environment
    environment = MCNFEnvironment(
        budgets,
        nuggets=nuggets,
        slopes=slopes,
        sills=sills,
        sigmas=sigmas)

    # simulation parameters
    np.random.seed(12345)
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    exploration_horizon = 75
    experiments = 25

    # learners
    gp_subcampaign_learner = GPTSLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gp_learner = BudgetLearner(budgets, gp_subcampaign_learner, n_campaigns)

    # data
    _, optimal_value = budget_optimizer(
        bb_matrices=[
            np.array([env.function(x, true_value=True) for x in budgets]).reshape((-1, 1))
            for env in environment.subenvs],
        budget_values=budgets, log=True)

    # testing
    tester = Tester(
        env=environment,
        lrns=(gp_learner,),
        oer=optimal_value,
        horizon=exploration_horizon,
        exps=experiments)
    tester.run(multiprocess=True)
    tester.show_results()




