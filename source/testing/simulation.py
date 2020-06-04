from source.testing.__dependencies import *


class Simulation:
    def __init__(self, environment, learner, exploration_horizon, experiments=1):
        self.environment = environment
        self.learner = learner
        self.exploration_horizon = exploration_horizon
        self.experiments = experiments
        self.cumulative_rewards = []
        self.collected_rewards = np.zeros(shape=(experiments, exploration_horizon))

    @staticmethod
    def run_subroutine(learner, environment):
        selected = learner.select_arm()
        reward = environment.simulate_round(candidate=selected)
        learner.update(candidate=selected, reward=reward)
        return reward * selected

    def run_subcycle(self, learner, environment):
        for iteration in range(self.exploration_horizon):
            self.run_subroutine(learner, environment)

    @staticmethod
    def time_for_humans(time_in_seconds):
        def to_string(number, quantity):
            return ' {} {}{}'.format(number, quantity, 's' if number > 1 else '') if number else ''

        int_time = int(np.ceil(time_in_seconds))
        seconds = int_time % 60
        minutes = int((int_time - seconds)/60) % 60
        hours = int(((int_time - seconds)/60 - minutes)/60) % 24
        days = int(((((int_time - seconds)/60 - minutes)/60) - hours)/24)
        return '{}{}{}{}'.format(
            to_string(days, 'day'),
            to_string(hours, 'hour'),
            to_string(minutes, 'minute'),
            to_string(seconds, 'second'))[1:]

    def run(self):
        dialog = 'running simulations...'
        print('\n» ' + dialog, end='')
        start_time = time()
        average_iteration_time = None
        for experiment in range(self.experiments):
            local_time = time()
            new_dialog = \
                'running simulations [experiment {} of {}{}]...'.format(
                    experiment+1,
                    self.experiments,
                    ' | average time per experiment: {0:.3f} s. | remaining (predicted): {1}'.format(
                        average_iteration_time,
                        self.time_for_humans(average_iteration_time * (self.experiments - experiment)))
                    if average_iteration_time is not None else '')
            print('\b'*len(dialog) + new_dialog, end='')
            dialog = new_dialog
            learner = deepcopy(self.learner)
            environment = deepcopy(self.environment)
            self.run_subcycle(learner, environment)
            self.collected_rewards[experiment, :] = learner.collected_rewards
            self.cumulative_rewards.append(np.sum(learner.collected_rewards))
            local_time = time() - local_time
            average_iteration_time = \
                local_time \
                if average_iteration_time is None \
                else (average_iteration_time * experiment + local_time)/(experiment + 1)
        end_time = time()
        print('{}simulations comlpleted in {} seconds'.format(
            '\b'*len(dialog),
            self.time_for_humans(end_time-start_time)))

