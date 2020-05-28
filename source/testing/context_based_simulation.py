from source.testing.simulation import Simulation


class ContextBasedSimulation(Simulation):
    def __init__(self, environment, learner, exploration_horizon, context_generation_period, experiments=1):
        super().__init__(environment, learner, exploration_horizon, experiments)
        self.context_generation_period = context_generation_period

    @staticmethod
    def run_subroutine(learner, environment):
        user_features = environment.simulate_user()
        learner.select_class(user_features)
        Simulation.run_subroutine(learner, environment)

    def run_subcycle(self, learner, environment):
        for iteration in range(self.exploration_horizon):
            if iteration >= self.context_generation_period and iteration % self.context_generation_period == 0:
                learner.generate_context(environment.get_data())
            self.run_subroutine(learner, environment)
