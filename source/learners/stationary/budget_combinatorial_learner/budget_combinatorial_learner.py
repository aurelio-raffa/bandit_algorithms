from source.learners.stationary.budget_combinatorial_learner.__dependencies import *
from source.base.base_learner import BaseLearner
from source.optimization.budget_optimizer import budget_optimizer


class BudgetCombinatorialLearner(BaseLearner):
    def __init__(self, budgets, subcampaign_learner, n_campaigns):
        """
        learner class for budget-constrained multi-armed-bandit
        :param budgets: the values of budgets, common among every subcampaign
        :param subcampaign_learner: the learner object for each subcampaign
        :param n_campaigns: the number of subcampaigns
        """
        super().__init__()
        self.budgets = budgets
        self.learners = [deepcopy(subcampaign_learner) for _ in range(n_campaigns)]
        self.n_campaigns = n_campaigns

    def select_arm(self):
        bb_matrices = [learner.sample_candidates() for learner in self.learners]
        optimal_allocation, _ = budget_optimizer(bb_matrices, self.budgets)
        return optimal_allocation

    def update(self, candidate, reward):
        super().update(candidate, np.sum(reward))
        for idx, cndt, rwd in zip(range(self.n_campaigns), candidate, reward):
            self.learners[idx].update(cndt, rwd)

    def __str__(self):
        return 'Budget-constrained Combinatorial Learner'
