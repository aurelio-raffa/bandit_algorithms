from source.context.__dependencies import *
from source.environments.stationary.stationary_conversion_rate.environment import Environment
from source.learners.stationary.stationary_thompson_sampling.sts_learner import ThompsonSamplingLearner


if __name__ == '__main__':

    n_arms = 5 #(immaginiamo siano ex:10, 20, 55, 80 ,100 euro)

    p1 = np.array([0.75, 0.5, 0.25, 0.15, 0.20]) # Valori da leggere sulle curve di domanda (
    p2 = np.array([0.35, 0.55, 0.20, 0.30, 0.1])
    p3 = np.array([0.6, 0.2, 0.15, 0.15, 0.1])
    opt1 = p1[0] # This is the optimal arm (0.35 is the greatest) --> My guess
    opt2 = p2[1]
    opt3 = p3[0]
    T_1 = 100  # NUMERO DI PERSONE DELLA CLASSE 1 CHE HANNO CLICCATO
    T_2 = 80
    T_3 = 50

    #n_experiments = 1000


    '''for e in range(n_experiments):
        print("Experiment ", e)
        env = Environment(n_arms=n_arms, probabilities=p)
        ts_learner = TS_Learner(n_arms)'''
    env1 = Environment(candidates=range(n_arms), probabilities=p1)
    env2 = Environment(candidates=range(n_arms), probabilities=p2)
    env3 = Environment(candidates=range(n_arms), probabilities=p3)
    ts = ThompsonSamplingLearner(range(n_arms)) #se adottiamo gli stessi 5 prezzi epr ogni classe il elarner è lo stesso e basta crearne uno
    c_rewards = np.zeros(shape=(T_1+T_2+T_3))
    print(c_rewards.shape)
    for it in range(50):
        ts_learner = copy.deepcopy(ts)
        for t in range(T_1):
            # TS Learner
             pulled_arm = ts_learner.select_arm()
             reward = env1.simulate_round(pulled_arm)
             ts_learner.update(pulled_arm, reward)

        for t in range(T_2):
            # TS Learner
             pulled_arm = ts_learner.select_arm()
             reward = env2.simulate_round(pulled_arm)
             ts_learner.update(pulled_arm, reward)

        for t in range(T_3):
            # TS Learner
             pulled_arm = ts_learner.select_arm()
             reward = env3.simulate_round(pulled_arm)
             ts_learner.update(pulled_arm, reward)

        c_rewards+=ts_learner.collected_rewards
        # Regret = T*opt - sum_t(rewards_t)
    c_rewards/=50

    plt.figure(0)
    plt.xlabel("t")
    plt.ylabel("Regret")


    regret1 = opt1 - c_rewards
    regret2 = opt2 - c_rewards
    regret3 = opt3 - c_rewards

    avg_regret1 = np.cumsum(regret1[0:T_1])
    avg_regret2 = np.cumsum(regret2[T_1:T_1+T_2])+avg_regret1[-1]
    avg_regret3 = np.cumsum(regret3[T_1+T_2:T_1+T_2+T_3])+avg_regret2[-1]
    # Plot
    plt.plot(avg_regret1, 'r')
    plt.show()
    plt.plot(avg_regret2, 'g')
    plt.show()
    plt.plot(avg_regret3, 'b')
    plt.show()
    # The same is done for Greedy_Learner
    plt.show()
    #plt.legend(["TS", "Greedy"])
    plt.plot(np.concatenate([avg_regret1, avg_regret2, avg_regret3]))
    plt.show()










