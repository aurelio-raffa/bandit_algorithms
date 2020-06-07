from src.optimization.__dependencies import *
from src.lrn.stn.bc.bcl import BudgetLearner
from src.lrn.stn.gau.gts import GTSLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.lrn.dyn.gau.gps import SWGPTSLearner
from src.env.dyn.mc.mce import DMCNFEnvironment
from src.env.stn.nf.nf import NoisyFunction
from src.testing.tester import Tester
from src.optimization.budget_optimizer import budget_optimizer

if __name__ == '__main__':
    # variables
    budgets = np.linspace(start=0, stop=100, num=5)
    sigma_true = 5
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    n_campaigns = 3
    exploration_horizon = 250
    experiments = 100
    learner_window_size = 30
    abrupt_changes = [50, 150]
    nuggets = np.array([
        [10, 30, 20],
        [5, 15, 10],
        [20, 20, 30]])
    slopes = np.array([
        [5, 4, 7],
        [7, 9, 6],
        [4, 5, 6]])
    sills = np.array([
        [30, 60, 90],
        [70, 110, 100],
        [60, 120, 50]])
    sigmas = sigma_true * np.ones((len(abrupt_changes) + 1, n_campaigns))
    # learners
    gts_subcampaign_learner = GTSLearner(candidates=budgets, sigma=sigma_learner)
    gpl_subcampaign_learner = GPTSLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    swl_subcampaign_learner = SWGPTSLearner(
        candidates=budgets,
        window_size=learner_window_size,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gts_learner = BudgetLearner(budgets, gts_subcampaign_learner, n_campaigns)
    gpl_learner = BudgetLearner(budgets, gpl_subcampaign_learner, n_campaigns)
    swl_learner = BudgetLearner(budgets, swl_subcampaign_learner, n_campaigns)
    # environments
    environment = DMCNFEnvironment(
        budgets,
        nuggets=nuggets,
        slopes=slopes,
        sills=sills,
        sigmas=sigmas,
        abrupt_change_times=abrupt_changes)
    # data
    optimal_values = np.zeros(exploration_horizon)
    phases = [0] + abrupt_changes + [exploration_horizon]
    for it1 in range(len(abrupt_changes) + 1):
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
        optimal_values[phases[it1]:phases[it1+1]] = opt_val
    # testing
    tester = Tester(
        environment=environment,
        learners=(gts_learner, gpl_learner, swl_learner),
        optimal_expected_reward=optimal_values,
        exploration_horizon=exploration_horizon,
        experiments=experiments)
    tester.run()
    tester.show_results()
