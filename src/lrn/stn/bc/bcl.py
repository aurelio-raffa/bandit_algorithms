from src.lrn.stn.bc.__dep import *
from src.bse.bse_lrn import BaseLearner
from src.opt.bop import budget_optimizer


class BudgetLearner(BaseLearner):
    def __init__(self, budgets, subcampaign_learner, n_campaigns, values=1.):
        """
        learner class for budget-constrained multi-armed-bandit
        :param budgets: the values of budgets, common among every subcampaign
        :param subcampaign_learner: the learner object for each subcampaign
        :param n_campaigns: the number of subcampaigns
        :param values: a list values per click for every subcampaign, or a single scalar
            if all wights are equal (defaults to 1.)
        """
        super().__init__()
        self.budgets = budgets
        self.learners = [deepcopy(subcampaign_learner) for _ in range(n_campaigns)]
        self.n_campaigns = n_campaigns
        self.values = values if type(values) not in (int, float) else [values] * n_campaigns

    def select_arm(self, get_value=False):
        bb_matrices = [learner.sample_candidates() * values for learner, values in zip(self.learners, self.values)]
        optimal_allocation, optimal_value = budget_optimizer(bb_matrices, self.budgets)
        if get_value:
            return optimal_allocation, optimal_value
        else:
            return optimal_allocation

    def update(self, candidate, reward):
        total_reward = np.sum([rwd * val for rwd, val in zip(reward, self.values)])
        super().update(candidate, total_reward)
        for idx, cndt, rwd in zip(range(self.n_campaigns), candidate, reward):
            self.learners[idx].update(cndt, rwd)

    def __str__(self):
        return 'Budget-constrained Combinatorial Learner ({} Sublearner)'.format(str(self.learners[0]))
