from source.testing.__dependencies import *
from source.testing.simulation import Simulation


class PricingAdvertisingSimulation(Simulation):
    def __init__(
            self,
            subcampaign_environments,
            advertising_environment,
            pricing_learners,
            advertising_learner,
            subcampaign_costs,
            exploration_horizon,
            experiments):
        super().__init__(
            environment=advertising_environment,
            learner=advertising_learner,
            exploration_horizon=exploration_horizon,
            experiments=experiments)
        self.secondary_environments = subcampaign_environments
        self.secondary_learners = pricing_learners
        self.subcampaign_costs = subcampaign_costs

    def run_subcycle(self, learner, environment, experiment=None):
        if experiment is not None:
            self.reseed(experiment)
        subcampaign_environments = deepcopy(self.secondary_environments)
        pricing_learners = deepcopy(self.secondary_learners)
        n_subcampaigns = len(subcampaign_environments)
        for iteration in range(self.exploration_horizon):
            values = [
                np.max(
                    pricing_learner.beta_parameters[:, 0]
                    / np.sum(pricing_learner.beta_parameters, axis=1)
                    * np.array(pricing_learner.candidates) - cost)
                for pricing_learner, cost in zip(pricing_learners, self.subcampaign_costs)]
            learner.values = values
            allocation = learner.select_arm()
            clicks = environment.simulate_round(allocation)
            aux_values = np.zeros(n_subcampaigns)
            for iterations, subcampaign_learner, subcampaign_enviroment, cost, index in zip(
                    clicks, pricing_learners, subcampaign_environments, self.subcampaign_costs, range(n_subcampaigns)):
                aux_values[index] = .0
                for _ in range(int(iterations)):
                    reward = self.run_subroutine(subcampaign_learner, subcampaign_enviroment)
                    aux_values[index] += (reward - cost) / iterations
            learner.values = aux_values
            learner.update(allocation, clicks)
        return learner
