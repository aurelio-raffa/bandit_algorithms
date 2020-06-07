from src.opt.__dep import *
from src.lrn.stn.bc.bcl import BudgetLearner
from src.lrn.stn.gau.gts import GTSLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.env.stn.mc.mnf import MCNFEnvironment
from src.tsg.tsr import Tester
from src.opt.bop import budget_optimizer


if __name__ == '__main__':
    # variables
    budgets = np.linspace(start=0, stop=100, num=5)
    sigma_true = 5
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    n_campaigns = 3
    nugget_c1, slope_c1, sill_c1 = 10, 5, 30
    nugget_c2, slope_c2, sill_c2 = 30, 4, 60
    nugget_c3, slope_c3, sill_c3 = 20, 7, 90
    # learners
    gts_subcampaign_learner = GTSLearner(candidates=budgets, sigma=sigma_learner)
    gpl_subcampaign_learner = GPTSLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gts_learner = BudgetLearner(budgets, gts_subcampaign_learner, n_campaigns)
    gpl_learner = BudgetLearner(budgets, gpl_subcampaign_learner, n_campaigns)
    # environments
    environment = MCNFEnvironment(
        budgets,
        nuggets=[nugget_c1, nugget_c2, nugget_c3],
        slopes=[slope_c1, slope_c2, slope_c3],
        sills=[sill_c1, sill_c2, sill_c3],
        sigmas=[sigma_true, sigma_true, sigma_true])
    # data
    _, optimal_value = budget_optimizer(
        bb_matrices=[
            np.array([env.function(x, true_value=True) for x in budgets]).reshape((-1, 1))
            for env in environment.subenvs],
        budget_values=budgets, log=True)
    # tsg
    tester = Tester(
        env=environment,
        lrns=(gts_learner, gpl_learner),
        oer=optimal_value,
        horizon=10,
        exps=20)
    tester.run()
    tester.show_results()
