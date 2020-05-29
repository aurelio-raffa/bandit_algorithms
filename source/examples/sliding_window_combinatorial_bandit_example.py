from source.optimization.__dependencies import *
from source.learners.stationary.budget_combinatorial_learner.budget_combinatorial_learner import BudgetCombinatorialLearner
from source.learners.stationary.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner
from source.learners.stationary.gaussian_thompson_sampling.gpts_learner import GaussianProcessThompsonSamplingLearner
from source.learners.dynamic.dynamic_gaussian_processes.gptssw_learner import SlidingWindowGaussianProcessThompsonSamplingLearner
from source.environments.dynamic.dynamic_mutlicampaign_noisyfunction_environment.dynamic_multicampaign_noisyfunction_environment import DynamicMulticampaignNoisyfunctionEnvironment
from source.environments.stationary.noisy_function.noisy_function import NoisyFunction
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
    gts_subcampaign_learner = GaussianThompsonSamplingLearner(candidates=budgets, sigma=sigma_learner)
    gpl_subcampaign_learner = GaussianProcessThompsonSamplingLearner(
        candidates=budgets,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    swl_subcampaign_learner = SlidingWindowGaussianProcessThompsonSamplingLearner(
        candidates=budgets,
        window_size=learner_window_size,
        sigma=sigma_learner,
        theta=theta_learner,
        lenscale=lenscale_learner)
    gts_learner = BudgetCombinatorialLearner(budgets, gts_subcampaign_learner, n_campaigns)
    gpl_learner = BudgetCombinatorialLearner(budgets, gpl_subcampaign_learner, n_campaigns)
    swl_learner = BudgetCombinatorialLearner(budgets, swl_subcampaign_learner, n_campaigns)
    # environments
    environment = DynamicMulticampaignNoisyfunctionEnvironment(
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
