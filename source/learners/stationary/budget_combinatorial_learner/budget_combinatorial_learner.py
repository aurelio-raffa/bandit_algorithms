from source.learners.stationary.budget_combinatorial_learner.__dependencies import *
from source.base.base_learner import BaseLearner
from source.optimization.budget_optimizer import budget_optimizer


class BudgetCombinatorialLearner(BaseLearner):
    def __init__(self, budgets, subcampaign_learner, n_campaigns, values=1.):
        """
        learner class for budget-constrained multi-armed-bandit
        :param budgets: the values of budgets, common among every subcampaign
        :param subcampaign_learner: the learner object for each subcampaign
        :param n_campaigns: the number of subcampaigns
        :param values: a list values per click for every possible bid and budget
            (in form of a matrix) corresponding to each subcampaign, or a single scalar
            if all wights are equal (defaults to 1.)
        """
        super().__init__()
        self.budgets = budgets
        self.learners = [deepcopy(subcampaign_learner) for _ in range(n_campaigns)]
        self.n_campaigns = n_campaigns
        self.values = values \
            if type(values) not in (int, float) \
            else [np.ones(shape=(len(budgets),)) * values for _ in range(n_campaigns)]

    def select_arm(self):
        bb_matrices = [learner.sample_candidates() * values for learner, values in zip(self.learners, self.values)]
        optimal_allocation, _ = budget_optimizer(bb_matrices, self.budgets)
        return optimal_allocation

    def update(self, candidate, reward):
        super().update(candidate, np.sum(reward))
        for idx, cndt, rwd in zip(range(self.n_campaigns), candidate, reward):
            self.learners[idx].update(cndt, rwd)

    def __str__(self):
        return 'Budget-constrained Combinatorial Learner ({} Sublearner)'.format(str(self.learners[0]))
