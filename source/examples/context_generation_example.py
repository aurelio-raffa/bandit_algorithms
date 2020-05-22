from source.context.__dependencies import *
from source.context.context_generation import ContextGeneration


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
    for iteration in range(size):
        class_index = choice(range(len(classes)), p=class_probabilities)
        candidate_index = choice(range(len(candidates)), p=candidates_probabilities)
        candidate = candidates[candidate_index]
        reward = binomial(1, average_rewards[class_index][candidate_index])
        ans.loc[iteration] = [iteration, *classes[class_index], candidate, reward]
    ans.loc[:, 'reward'] = ans['reward'].astype(float)
    return ans


if __name__ == '__main__':
    # data = pd.DataFrame(
    #     data=[
    #         [1, True, False, 1, 0],
    #         [2, True, True, 1, 1],
    #         [3, False, False, 2, 1],
    #         [4, False, True, 2, 1],
    #         [5, True, False, 1, 1],
    #         [6, False, True, 2, 1],
    #         [7, True, False, 2, 0],
    #         [8, True, True, 2, 0],
    #         [9, False, False, 1, 0],
    #         [10, False, True, 1, 0],
    #         [11, True, False, 2, 0],
    #         [12, False, True, 1, 0]],
    #     columns=['userID', 'feature1', 'feature2', 'candidate', 'reward'])
    data = generate_dataset(
        class_probabilities=[.25, .25, .25, .25],
        average_rewards=[
            [.6, .2, .2, .2],
            [.2, .6, .2, .2],
            [.2, .2, .6, .2],
            [.2, .2, .6, .2]],
        candidates=[1, 2, 3, 4],
        candidates_probabilities=[.25, .25, .25, .25],
        size=500)
    print(data)
    cg = ContextGeneration()
    cut_feature, margin = cg.context_cluster(
        data=data,
        delta=.5,
        features=['feature1', 'feature2'],
        log=True)
    print(cut_feature, margin)
    second_cut_feature, margin = cg.context_cluster(
        data=data.loc[data[cut_feature], :],
        delta=.5,
        features=['feature2'],
        log=True)
    print(second_cut_feature, margin)
