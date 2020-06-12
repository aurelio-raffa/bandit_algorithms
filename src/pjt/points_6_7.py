from src.pjt.__dep import *
from src.env.stn.mc.mnf import MCNFEnvironment
from src.env.stn.cr.env import Environment
from src.lrn.stn.bc.bcl import BudgetLearner
from src.lrn.stn.gau.gps import GPTSLearner
from src.lrn.stn.gau.gts import GTSLearner
from src.lrn.stn.ar.art import AverageRewardTSL
from src.tsg.pas import PASimulation
from src.tsg.tsr import Tester
from src.opt.bop import budget_optimizer


def ps6_7():
    # problem parameters
    subcampaign_costs = [.25, .55, .35]                       # [10, 20, 5]
    pricing_candidates = [.99, 1.79, 2.99, 3.49, 4.99]        # [50, 60, 70, 80, 90]
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
    advertising_environment = MCNFEnvironment(
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
    exploration_horizon = 150       # 200
    experiments = 10                # 25

    # pricing learners - point 6
    pricing_learners_p6 = [AverageRewardTSL(pricing_candidates) for _ in range(len(subcampaign_costs))]

    # pricing learners - point 7
    pricing_learner_p7 = AverageRewardTSL(pricing_candidates)
    pricing_learners_p7b = [pricing_learner_p7] * len(subcampaign_costs)

    # advertising learner
    advertising_learner = BudgetLearner(
        budgets=budgets,
        subcampaign_learner=GPTSLearner(
            candidates=budgets,
            sigma=sigma_learner,
            theta=theta_learner,
            lenscale=lenscale_learner),
        n_campaigns=len(subcampaign_costs))

    # simulation - point 6
    simulation_p6 = PASimulation(
        sub_envs=subcampaign_environments,
        ad_env=advertising_environment,
        pr_lrns=pricing_learners_p6,
        ad_lrn=advertising_learner,
        sbcmp_costs=subcampaign_costs,
        horizon=exploration_horizon,
        exps=experiments)

    # simulation - point 7
    simulation_p7 = PASimulation(
        sub_envs=subcampaign_environments,
        ad_env=advertising_environment,
        pr_lrns=pricing_learner_p7,
        ad_lrn=advertising_learner,
        sbcmp_costs=subcampaign_costs,
        horizon=exploration_horizon,
        exps=experiments)
    simulation_p7b = PASimulation(
        sub_envs=subcampaign_environments,
        ad_env=advertising_environment,
        pr_lrns=pricing_learners_p7b,
        ad_lrn=advertising_learner,
        sbcmp_costs=subcampaign_costs,
        horizon=exploration_horizon,
        exps=experiments)

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
            '\tcampaign {0}:\t{1:.3f}'.format(index, value)
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
        oer=optimal_value,
        horizon=exploration_horizon,
        sims=(simulation_p6, simulation_p7, simulation_p7b))

    # run
    tester.run()
    tester.show_results()


if __name__ == '__main__':
    ps6_7()
