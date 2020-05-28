from source.optimization.__dependencies import *
from source.learners.stationary.budget_combinatorial_learner.budget_combinatorial_learner import BudgetCombinatorialLearner
from source.learners.stationary.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner
from source.learners.stationary.gaussian_thompson_sampling.gpts_learner import GaussianProcessThompsonSamplingLearner
from source.environments.stationary.stationary_multicampaign_environment.multicampaign_noisyfunction_environment import MulticampaignNoisyfunctionEnvironment
from source.testing.tester import Tester
from source.optimization.budget_optimizer import budget_optimizer


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
    gts_subcampaign_learner = GaussianThompsonSamplingLearner(candidates=budgets, sigma=sigma_learner)
    gpl_subcampaign_learner = GaussianProcessThompsonSamplingLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gts_learner = BudgetCombinatorialLearner(budgets, gts_subcampaign_learner, n_campaigns)
    gpl_learner = BudgetCombinatorialLearner(budgets, gpl_subcampaign_learner, n_campaigns)
    # environments
    environment = MulticampaignNoisyfunctionEnvironment(
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
    # testing
    tester = Tester(
        environment=environment,
        learners=(gts_learner, gpl_learner),
        optimal_expected_reward=optimal_value,
        exploration_horizon=10,
        experiments=20)
    tester.run()
    tester.show_results()
