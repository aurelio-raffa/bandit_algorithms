from source.project.__dependencies import *
from source.environments.stationary.stationary_multicampaign_environment.multicampaign_noisyfunction_environment \
    import MulticampaignNoisyfunctionEnvironment
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.learners.stationary.budget_combinatorial_learner.budget_combinatorial_learner \
    import BudgetCombinatorialLearner
from source.learners.stationary.gaussian_thompson_sampling.gpts_learner import GaussianProcessThompsonSamplingLearner
from source.learners.stationary.gaussian_thompson_sampling.gts_learner import GaussianThompsonSamplingLearner
from source.learners.stationary.average_reward_learner.average_reward_thompson_sampling_learner \
    import AverageRewardThompsonSamplingLearner
from source.testing.pricing_advertising_simulation import PricingAdvertisingSimulation
from source.testing.tester import Tester
from source.optimization.budget_optimizer import budget_optimizer


if __name__ == '__main__':
    # problem parameters
    subcampaign_costs = [10, 20, 5]
    pricing_candidates = [50, 60, 70, 80, 90]
    probabilities = [
        [.1, .2, .5, .75, .1],      # campaign 1
        [.2, .9, .55, .2, .4],      # campaign 2
        [.7, .65, .3, .4, .5]]      # campaign 3
    budgets = np.linspace(start=0, stop=100, num=5)
    nuggets = [5, 10, 15]
    slopes = [5, 3, 2]
    sills = [50, 60, 100]
    sigmas = [5, 5, 5]

    # environments
    subcampaign_environments = [Environment(pricing_candidates, probs) for probs in probabilities]
    advertising_environment = MulticampaignNoisyfunctionEnvironment(
        candidates=budgets,
        nuggets=nuggets,
        slopes=slopes,
        sills=sills,
        sigmas=sigmas)

    # simulation parameters
    np.random.seed(12345)
    sigma_learner = 10
    theta_learner = 10
    lenscale_learner = 10
    exploration_horizon = 120       # 200
    experiments = 10                # 25

    # pricing learners - point 6
    pricing_learners_p6 = [
        AverageRewardThompsonSamplingLearner(pricing_candidates) for _ in range(len(subcampaign_costs))]

    # pricing learners - point 7
    sub_learn = AverageRewardThompsonSamplingLearner(pricing_candidates)
    pricing_learners_p7 = [sub_learn] * len(subcampaign_costs)

    # advertising learner
    advertising_learner = BudgetCombinatorialLearner(
        budgets=budgets,
        subcampaign_learner=GaussianProcessThompsonSamplingLearner(
            candidates=budgets,
            sigma=sigma_learner,
            theta=theta_learner,
            lenscale=lenscale_learner),
        n_campaigns=len(subcampaign_costs))

    # simulation - point 6
    simulation_p6 = PricingAdvertisingSimulation(
        subcampaign_environments=subcampaign_environments,
        advertising_environment=advertising_environment,
        pricing_learners=pricing_learners_p6,
        advertising_learner=advertising_learner,
        subcampaign_costs=subcampaign_costs,
        exploration_horizon=exploration_horizon,
        experiments=experiments)

    # simulation - point 7
    simulation_p7 = PricingAdvertisingSimulation(
        subcampaign_environments=subcampaign_environments,
        advertising_environment=advertising_environment,
        pricing_learners=pricing_learners_p7,
        advertising_learner=advertising_learner,
        subcampaign_costs=subcampaign_costs,
        exploration_horizon=exploration_horizon,
        experiments=experiments)

    # data
    optimal_prices_indices = [
        int(np.argmax(np.array(probs) * np.array(prices)))
        for probs, prices
        in zip(probabilities, [pricing_candidates] * len(probabilities))]
    optimal_prices = [pricing_candidates[index] for index in optimal_prices_indices]
    subcampaign_values = [
        probs[index] * price - cost
        for probs, index, price, cost
        in zip(probabilities, optimal_prices_indices, optimal_prices, subcampaign_costs)]
    print(
        '\noptimal prices (disaggregated):\n', *[
            '\tcampaign {}:\t{}'.format(index, price)
            for index, price
            in zip(range(len(optimal_prices)), optimal_prices)], sep='\n')
    print(
        '\nsubcampaign values:\n', *[
            '\tcampaign {}:\t{}'.format(index, value)
            for index, value
            in zip(range(len(subcampaign_values)), subcampaign_values)], sep='\n')
    _, optimal_value = budget_optimizer(
        bb_matrices=[
            np.array([env.function(x, true_value=True) for x in budgets]).reshape((-1, 1)) * value
            for env, value
            in zip(advertising_environment.subenvs, subcampaign_values)],
        budget_values=budgets, log=True)

    # tester
    tester = Tester(
        optimal_expected_reward=optimal_value,
        exploration_horizon=exploration_horizon,
        simulations=(simulation_p6, simulation_p7))

    # run
    tester.run()
    tester.show_results()
