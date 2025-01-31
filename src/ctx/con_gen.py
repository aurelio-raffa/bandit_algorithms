from src.ctx.__dep import *


class ContextGenerator:
    def __init__(self):
        self.features = None
        self.model = None

    @staticmethod
    def split_margin(data, delta, feature_name, log=False):
        num_record = data.shape[0]
        current = data.loc[:, ['candidate', 'reward']].groupby(['candidate']).agg(np.mean)
        current['total'] = current['reward'] * current.index
        i0 = current['total'].argmax()
        v0, c0 = current.iloc[i0]['reward'], current.index[i0]
        expected = data.loc[:, [feature_name, 'candidate', 'reward']].groupby(
            [feature_name, 'candidate']).agg(np.mean)
        count = data[['userID', feature_name]].groupby(feature_name).count()
        p1 = float(count.loc[True, :] / num_record)
        p2 = float(count.loc[False, :] / num_record)
        h1 = sqrt(- .5 * np.log(delta) / count.loc[True, :])
        h2 = sqrt(- .5 * np.log(delta) / count.loc[False, :])
        h0 = sqrt(- .5 * np.log(delta) / num_record)
        expected['total'] = expected['reward'] * np.array([tp[1] for tp in expected.index])
        i1 = expected.loc[True, 'total'].argmax()
        v1, c1 = expected.loc[True, 'reward'].iloc[i1], expected.loc[True].index[i1]
        i2 = expected.loc[False, 'total'].argmax()
        v2, c2 = expected.loc[False, 'reward'].iloc[i2], expected.loc[False].index[i2]
        margin = float((p1 - h0) * (v1 - h1) * c1 + (p2 - h0) * (v2 - h2) * c2 - (v0 - h0) * c0)
        if log:
            print('\nassessing split on feature:', feature_name)
            print('\tmargin:', margin)
        return margin

    @staticmethod
    def context_cluster(data, delta, features, log=False):
        margins = [ContextGenerator.split_margin(data, delta, feature, log) for feature in features]
        index = int(np.argmax(margins))
        value = margins[index]
        return (features[index], value) if value > 0 else (None, value)

    def initialize_two_features(self, features):
        assert len(features) == 2
        self.model = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        self.features = deepcopy(features)

    def train(self, data, delta, features, max_splits=2, log=False):
        t_start = time() if log else 0
        self.initialize_two_features(features)
        assert max_splits == 2
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
        assert len(self.features) == 2
        return self.model[(datum[self.features[0]], datum[self.features[1]])]

    def number_of_classes(self):
        return len(np.unique(list(self.model.values())))
