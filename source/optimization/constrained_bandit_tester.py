from source.optimization.__dependencies import *
from source.learners.stationary.budget_combinatorial_learner.budget_combinatorial_learner import BudgetCombinatorialLearner
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner
from source.environments.stationary.stationary_multicampaign_environment.multicampaign_environment import MulticampaignEnvironment
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.testing.tester import Tester
from source.optimization.budget_optimizer import budget_optimizer


if __name__ == '__main__':
    budgets = [0., 1., 2., 3.]
    subcampaign_learner = ThompsonSamplingLearner(candidates=budgets)
    n_campaigns = 4
    learner = BudgetCombinatorialLearner(budgets, subcampaign_learner, n_campaigns)
    env_c1 = Environment(candidates=budgets, probabilities=[0., .1, .3, .4])
    env_c2 = Environment(candidates=budgets, probabilities=[0., .3, .2, .7])
    env_c3 = Environment(candidates=budgets, probabilities=[0., .5, .5, .1])
    env_c4 = Environment(candidates=budgets, probabilities=[0., .3, .1, .2])
    environment = MulticampaignEnvironment(subenvironments=[env_c1, env_c2, env_c3, env_c4])
    _, optimal_value = budget_optimizer(
        bb_matrices=[
            np.array([0., .1, .3, .4]).reshape((-1, 1)),
            np.array([0., .3, .2, .7]).reshape((-1, 1)),
            np.array([0., .5, .5, .1]).reshape((-1, 1)),
            np.array([0., .3, .1, .2]).reshape((-1, 1))],
        budget_values=budgets, pedantic=True)
    tester = Tester(
        environment=environment,
        learners=(learner,),
        optimal_expected_reward=optimal_value,
        exploration_horizon=500,
        experiments=300)
    tester.run()
    tester.show_results()
