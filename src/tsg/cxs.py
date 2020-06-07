from src.tsg.sim import Simulation


class ContextSimulation(Simulation):
    def __init__(self, env, lrn, horizon, ctxgen_period, exps=1):
        super().__init__(env, lrn, horizon, exps)
        self.context_generation_period = ctxgen_period

    @staticmethod
    def run_subroutine(learner, environment):
        user_features = environment.simulate_user()
        learner.select_class(user_features)
        Simulation.run_subroutine(learner, environment)

    def run_subcycle(self, learner, environment, experiment=None):
        if experiment is not None:
            self.reseed(experiment)
        for iteration in range(self.exploration_horizon):
            if iteration >= self.context_generation_period and iteration % self.context_generation_period == 0:
                learner.generate_context()
            self.run_subroutine(learner, environment)
        return learner
