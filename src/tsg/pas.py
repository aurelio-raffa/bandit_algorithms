from src.tsg.__dep import *
from src.tsg.sim import Simulation


class PASimulation(Simulation):
    def __init__(self, sub_envs, ad_env, pr_lrns, ad_lrn, sbcmp_costs, horizon, exps):
        super().__init__(
            env=ad_env,
            lrn=ad_lrn,
            horizon=horizon,
            exps=exps)
        self.secondary_environments = sub_envs
        self.secondary_learners = pr_lrns
        self.subcampaign_costs = sbcmp_costs

    def run_subcycle(self, learner, environment, experiment=None):
        if experiment is not None:
            self.reseed(experiment)
        subcampaign_environments = deepcopy(self.secondary_environments)
        pricing_learners = deepcopy(self.secondary_learners)
        n_subcampaigns = len(subcampaign_environments)
        for iteration in range(self.exploration_horizon):
            if type(pricing_learners) == list:
                values = [
                    np.max(
                        pricing_learner.beta_parameters[:, 0]
                        / np.sum(pricing_learner.beta_parameters, axis=1)
                        * np.array(pricing_learner.candidates) - cost)
                    for pricing_learner, cost in zip(pricing_learners, self.subcampaign_costs)]
                learner.values = values
                allocation = learner.select_arm()
            else:
                allocation = None
                best_objective = -np.inf
                for it in range(len(pricing_learners.candidates)):
                    curr_values = [
                        pricing_learners.beta_parameters[:, 0]
                        / np.sum(pricing_learners.beta_parameters, axis=1)
                        * np.array(pricing_learners.candidates) - cost
                        for cost in self.subcampaign_costs]
                    learner.values = curr_values
                    curr_allocation, curr_objective = learner.select_arm(get_value=True)
                    if curr_objective > best_objective:
                        best_objective = curr_objective
                        allocation = curr_allocation
            clicks = environment.simulate_round(allocation)
            aux_values = np.zeros(n_subcampaigns)
            sub_learners = pricing_learners if type(pricing_learners) == list else [pricing_learners] * n_subcampaigns
            for iterations, subcampaign_learner, subcampaign_enviroment, cost, index in zip(
                    clicks, sub_learners, subcampaign_environments, self.subcampaign_costs, range(n_subcampaigns)):
                aux_values[index] = .0
                for _ in range(int(iterations)):
                    reward = self.run_subroutine(subcampaign_learner, subcampaign_enviroment)
                    aux_values[index] += (reward - cost) / iterations
            learner.values = aux_values
            learner.update(allocation, clicks)
        return learner
