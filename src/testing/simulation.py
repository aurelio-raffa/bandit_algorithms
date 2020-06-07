from src.testing.__dependencies import *


class Simulation:
    def __init__(self, environment, learner, exploration_horizon, experiments=1, seed=None):
        self.environment = environment
        self.learner = learner
        self.exploration_horizon = exploration_horizon
        self.experiments = experiments
        self.cumulative_rewards = []
        self.collected_rewards = np.zeros(shape=(experiments, exploration_horizon))
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def run_subroutine(learner, environment):
        selected = learner.select_arm()
        reward = environment.simulate_round(candidate=selected)
        learner.update(candidate=selected, reward=reward)
        return reward * selected

    def reseed(self, number):
        self.seed = \
            self.seed * self.seed * number % 4294967295 \
            if self.seed is not None else \
            number * number * number * number % 4294967295
        np.random.seed(self.seed)

    def run_subcycle(self, learner, environment, experiment=None):
        if experiment is not None:
            self.reseed(experiment)
        for iteration in range(self.exploration_horizon):
            self.run_subroutine(learner, environment)
        return learner

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

    def run(self, multiprocess=False, seed=None):
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        dialog = 'running simulations{}...'.format(' [using multiprocessing] ' if multiprocess else '')
        print('\n» ' + dialog, end='')
        start_time = time()
        average_iteration_time = None
        if not multiprocess:
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
        else:
            with Executor() as exe:
                processes = [
                    exe.submit(self.run_subcycle, self.learner, self.environment, exp)
                    for exp in range(self.experiments)]
                new_dialog = 'running simulations [multiprocessing: all proc. started (time estimate unavailable)] ...'
                print('\b' * len(dialog) + new_dialog, end='')
                dialog = new_dialog
                for process, count in zip(as_completed(processes), range(self.experiments)):
                    lrn = process.result()
                    self.collected_rewards[count, :] = lrn.collected_rewards
                    self.cumulative_rewards.append(np.sum(lrn.collected_rewards))
        end_time = time()
        print('{}simulations completed in {}'.format(
            '\b'*len(dialog),
            self.time_for_humans(end_time-start_time)))

