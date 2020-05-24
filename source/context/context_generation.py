from source.context.__dependencies import *


class ContextGeneration:
    def __init__(self):
        self.features = None
        self.model = None

    @staticmethod
    def split_margin(data, delta, feature_name, optimal, log=False):
        num_record = data.shape[0]
        expected = data.loc[:, [feature_name, 'candidate', 'reward']].groupby(
            [feature_name, 'candidate']).agg(np.mean)
        count = data[['userID', feature_name]].groupby(feature_name).count()
        p1 = float(count.loc[True, :] / num_record)
        p2 = float(count.loc[False, :] / num_record)
        h1 = sqrt(- .5 * np.log(delta) / count.loc[True, :])
        h2 = sqrt(- .5 * np.log(delta) / count.loc[False, :])
        h0 = sqrt(- .5 * np.log(delta) / num_record)
        v1 = expected.loc[True, 'reward'].max()
        v2 = expected.loc[False, 'reward'].max()
        margin = float((p1 - h0) * (v1 - h1) + (p2 - h0) * (v2 - h2) - (optimal - h0))
        if log:
            print('\nassessing split on feature:', feature_name)
            print('\tlower-bound on baseline optimal reward:', optimal)
            print('\tlower-bound on class probability (value "True"):', p1 - h1)
            print('\tlower-bound on class probability (value "False"):', p2 - h2)
            print('\tlower-bound on expected reward for class (value "True"):', v1 - h1)
            print('\tlower-bound on expected reward for class (value "False"):', v2 - h2)
            print('\tmargin:', margin)
        return margin

    @staticmethod
    def context_cluster(data, delta, features, log=False):
        optimal = np.max(data.loc[:, ['candidate', 'reward']].groupby('candidate').agg(np.mean))[0]
        margins = [ContextGeneration.split_margin(data, delta, feature, optimal, log) for feature in features]
        index = int(np.argmax(margins))
        value = margins[index]
        return (features[index], value) if value > 0 else (None, value)

    def train(self, data, delta, features, max_splits=2, log=False):
        t_start = time() if log else 0
        assert len(features) == 2 and max_splits == 2
        self.model = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        self.features = deepcopy(features)
        feature, margin = self.context_cluster(data, delta, features, log=log)
        first = (feature == features[0])
        if feature is not None:
            if first:
                self.model[(False, True)] += 1
                self.model[(False, False)] += 1
            else:
                self.model[(True, False)] += 1
                self.model[(False, False)] += 1
            sub_feature, sub_margin = None, 0
            features.remove(feature)
            current_split = None
            for level in [True, False]:
                temp_data = data.loc[data[feature] == level]
                temp_feature, temp_margin = self.context_cluster(temp_data, delta, features, log=log)
                if temp_margin > sub_margin:
                    sub_feature, sub_margin = temp_feature, temp_margin
                    current_split = level
            if current_split is not None:
                if first:
                    self.model[(current_split, False)] += 2 - self.model[(current_split, False)]
                else:
                    self.model[(False, current_split)] += 2 - self.model[(False, current_split)]
        if log:
            print('[training took {0:.3f} seconds]\n'.format(time()-t_start))

    def show_model(self):
        assert len(self.features) == 2
        print(
            ' ' * (len(self.features[0]) + 5),
            '{}: {}'.format(self.features[1], True),
            '{}: {}'.format(self.features[1], False), sep='\t')
        for key1 in (True, False):
            print(
                '{}: {}\t'.format(self.features[0], key1),
                self.model[(key1, True)],
                ' ' * (len(self.features[1]) + 5),
                self.model[(key1, False)])

    def predict(self, datum):
        return self.model[(datum[self.features[0]], datum[self.features[1]])]
