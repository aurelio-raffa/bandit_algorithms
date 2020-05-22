from source.context.__dependencies import *


class ContextGeneration:
    def __init__(self):
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
        margin = float((p1 - h1) * (v1 - h1) + (p2 - h2) * (v2 - h2) - (optimal - h0))
        if log:
            print('\nassessing split on feature:', feature_name)
            print('\tlower-bound on baseline optimal reward:', optimal)
            print('\tlower-bound on class probability (value "True"):', p1 - h1)
            print('\tlower-bound on class probability (value "False"):', p2 - h2)
            print('\tlower-bound on expected reward for class (value "True"):', v1 - h1)
            print('\tlower-bound on expected reward for class (value "False"):', v2 - h2)
            print('\tmargin:', margin)
        return margin

    def context_cluster(self, data, delta, features, log=False):
        optimal = np.max(data.loc[:, ['candidate', 'reward']].groupby('candidate').agg(np.mean))[0]
        margins = [self.split_margin(data, delta, feature, optimal, log) for feature in features]
        index = int(np.argmax(margins))
        value = margins[index]
        if value > 0:
            # update self.model !
            return features[index], value
        else:
            return None, value
