from source.project.__dependencies import *
from source.environments.stationary.stationary_multicampaign_environment.multicampaign_noisyfunction_environment import MulticampaignNoisyfunctionEnvironment
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.learners.stationary.budget_combinatorial_learner.budget_combinatorial_learner import BudgetCombinatorialLearner
from source.learners.stationary.gaussian_thompson_sampling.gpts_learner import GaussianProcessThompsonSamplingLearner
from source.learners.stationary.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner
from source.learners.stationary.average_reward_learner.average_reward_thompson_sampling_learner import AverageRewardThompsonSamplingLearner
from source.testing.pricing_advertising_simulation import PricingAdvertisingSimulation
from source.testing.tester import Tester
from source.optimization.budget_optimizer import budget_optimizer


if __name__ == '__main__':
    # parameters
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    subcampaign_costs = [10, 20, 5]
    pricing_candidates = [50, 60, 70, 80, 90]
    probabilities_campaign1 = [.1, .2, .5, .7, .8]
    probabilities_campaign2 = [.2, .9, .55, .2, .6]
    probabilities_campaign3 = [.7, .65, .3, .4, .5]
    subcampaign_environments = [
        Environment(pricing_candidates, probabilities_campaign1),
        Environment(pricing_candidates, probabilities_campaign2),
        Environment(pricing_candidates, probabilities_campaign3)]
    budgets = np.linspace(start=0, stop=100, num=5)
    nuggets = [5, 10, 15]
    slopes = [5, 3, 2]
    sills = [50, 60, 100]
    sigmas = [5, 5, 5]
    exploration_horizon = 130
    experiments = 10
    # learners
    pricing_learners = [
        AverageRewardThompsonSamplingLearner(pricing_candidates) for _ in range(len(subcampaign_costs))]
    advertising_learner = BudgetCombinatorialLearner(
        budgets=budgets,
        subcampaign_learner=GaussianProcessThompsonSamplingLearner(
            candidates=budgets,
            sigma=sigma_learner,
            theta=theta_learner,
            lenscale=lenscale_learner),
        # subcampaign_learner=GaussianThompsonSamplingLearner(budgets, sigma_learner),
        n_campaigns=len(subcampaign_costs))
    # environments
    advertising_environment = MulticampaignNoisyfunctionEnvironment(
        candidates=budgets,
        nuggets=nuggets,
        slopes=slopes,
        sills=sills,
        sigmas=sigmas)
    # simulation
    simulation = PricingAdvertisingSimulation(
        subcampaign_environments=subcampaign_environments,
        advertising_environment=advertising_environment,
        pricing_learners=pricing_learners,
        advertising_learner=advertising_learner,
        subcampaign_costs=subcampaign_costs,
        exploration_horizon=exploration_horizon,
        experiments=experiments)
    # data
    subcampaign_values = [
        np.max(np.array(list(env.candidate_data.values())) * (np.array(list(env.candidate_data.keys())) - cost))
        for env, cost in zip(subcampaign_environments, subcampaign_costs)]
    print(subcampaign_values)
    _, optimal_value = budget_optimizer(
        bb_matrices=[
            np.array([env.function(x, true_value=True) for x in budgets]).reshape((-1, 1)) * value
            for env, value in zip(advertising_environment.subenvs, subcampaign_values)],
        budget_values=budgets, log=True)
    # tester
    tester = Tester(
        environment=(None,),
        learners=(None,),
        optimal_expected_reward=optimal_value,
        exploration_horizon=exploration_horizon,
        simulations=(simulation,))
    # run
    tester.run()
    tester.show_results()
