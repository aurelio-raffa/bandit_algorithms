from source.context.__dependencies import *
from source.context.context_generator import ContextGenerator


def generate_dataset(
        class_probabilities,
        average_rewards,
        candidates,
        candidates_probabilities,
        size):
    classes = [
        [True, True],
        [True, False],
        [False, True],
        [False, False]]
    ans = pd.DataFrame(
        columns=['userID', 'feature1', 'feature2', 'candidate', 'reward'])
    ans.loc[:, 'reward'] = ans['reward'].astype(float)
    for iteration in range(size):
        class_index = choice(range(len(classes)), p=class_probabilities)
        candidate_index = choice(range(len(candidates)), p=candidates_probabilities)
        candidate = candidates[candidate_index]
        reward = binomial(1, average_rewards[class_index][candidate_index])
        ans.loc[iteration] = [iteration, *classes[class_index], candidate, reward]
    return ans


if __name__ == '__main__':
    data = generate_dataset(
        class_probabilities=[.25, .25, .25, .25],
        average_rewards=[
            [.6, .2, .2, .2],
            [.2, .6, .2, .2],
            [.2, .2, .6, .2],
            [.2, .2, .2, .6]],
        candidates=[1, 2, 3, 4],
        candidates_probabilities=[.25, .25, .25, .25],
        size=1000)
    print(data)

    cg = ContextGenerator()
    cg.train(data=data, delta=.5, features=['feature1', 'feature2'], log=True)
    cg.show_model()
    data['predicted'] = data.apply(cg.predict, axis=1)
    print(data)
