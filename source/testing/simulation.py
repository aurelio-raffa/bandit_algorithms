from source.testing.__dependencies import *


class Simulation:
    def __init__(self, environment, learner, exploration_horizon, experiments=1):
        self.environment = environment
        self.learner = learner
        self.exploration_horizon = exploration_horizon
        self.experiments = experiments
        self.cumulative_rewards = []
        self.collected_rewards = np.zeros(shape=(experiments, exploration_horizon))

    def run(self):
        dialog = 'running simulation...'
        print('\n>> ' + dialog, end='')
        start_time = time()
        for experiment in range(self.experiments):
            new_dialog = 'running simulation (experiment {} of {})...'.format(experiment, self.experiments)
            print('\b'*len(dialog) + new_dialog, end='')
            dialog = new_dialog
            learner = deepcopy(self.learner)
            for iteration in range(self.exploration_horizon):
                selected = learner.select_arm()
                reward = self.environment.simulate_round(candidate=selected)
                learner.update(candidate=selected, reward=reward)
            self.collected_rewards[experiment, :] = learner.collected_rewards
            self.cumulative_rewards.append(np.sum(learner.collected_rewards))
        end_time = time()
        print('{0}simulation comlpleted in {1:.2f} seconds'.format(
            '\b'*len(dialog),
            end_time-start_time))

