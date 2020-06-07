from src.pjt.__dep import *
from src.env.stn.mc.mnf import MCNFEnvironment
from src.lrn.stn.bc.bcl import BudgetLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.lrn.dyn.gau.gps import SWGPTSLearner
from src.env.dyn.mc.mce import DMCNFEnvironment
from src.env.stn.nf.nf import NoisyFunction
from src.tsg.sim import Simulation
from src.tsg.tsr import Tester
from src.opt.bop import budget_optimizer


if __name__ == '__main__':
    # problem parameters
    budgets = np.linspace(start=0, stop=100, num=5)
    n_campaigns = 3
    sigma_true = 5
    abrupt_changes = [20, 60]
    nuggets = np.array([
        [5, 10, 15],
        [5, 15, 10],
        [20, 20, 30]])
    slopes = np.array([
        [5, 3, 2],
        [7, 9, 6],
        [4, 5, 6]])
    sills = np.array([
        [50, 60, 100],
        [40, 70, 50],
        [60, 50, 20]])
    sigmas = sigma_true * np.ones((len(abrupt_changes) + 1, n_campaigns))

    # environment
    environment = DMCNFEnvironment(
        budgets,
        nuggets=nuggets,
        slopes=slopes,
        sills=sills,
        sigmas=sigmas,
        abrupt_change_times=abrupt_changes)

    # simulation parameters
    np.random.seed(12345)
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    exploration_horizon = 100
    experiments = 10
    learner_window_size = 30

    # learners
    gp_subcampaign_learner = GPTSLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    sw_subcampaign_learner = SWGPTSLearner(
        candidates=budgets,
        window_size=learner_window_size,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gp_learner = BudgetLearner(budgets, gp_subcampaign_learner, n_campaigns)
    sw_learner = BudgetLearner(budgets, sw_subcampaign_learner, n_campaigns)

    # data
    optimal_values = np.zeros(exploration_horizon)
    phases = [0] + abrupt_changes + [exploration_horizon]
    for it1 in range(len(abrupt_changes) + 1):
        print('\n*** phase {} ***'.format(it1 + 1))
        function_list = [NoisyFunction(
            nugget=nuggets[it1, it2],
            slope=slopes[it1, it2],
            sill=sills[it1, it2],
            sigma=sigmas[it1, it2])
            for it2 in range(n_campaigns)]
        _, opt_val = budget_optimizer(
            bb_matrices=[
                np.array([fun(x, true_value=True) for x in budgets]).reshape((-1, 1))
                for fun in function_list],
            budget_values=budgets, log=True)
        optimal_values[phases[it1]:phases[it1 + 1]] = opt_val

    # testing
    tester = Tester(
        env=environment,
        lrns=(gp_learner, sw_learner),
        oer=optimal_values,
        horizon=exploration_horizon,
        exps=experiments)
    tester.run(multiprocess=True)
    tester.show_results()




