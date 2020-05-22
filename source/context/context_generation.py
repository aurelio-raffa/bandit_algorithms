from source.context.__dependencies import *


class ContextGeneration:
    def __init__(self):
        """
        è una matrice con prima colonna feature1, colonna2=feature2, colonne_arms= prezzi_arms
        in cui ho 0 se non è un prezzo visto ed uno se è il prezzo che ha visto
        ultima colonna = comprato si o no. #trasforma in Dataframe
        """

    def context_cluster(self, data, delta, opt_aggregate_arm, n_candidates):

        numrecord = data.shape[0]
        expected_feature1 = data.loc[:, ['feature1', 'candidate', 'reward']].groupby(['feature1', 'candidate']).agg(np.mean)
        expected_feature2 = data.loc[:, ['feature2', 'candidate', 'reward']].groupby(['feature2', 'candidate']).agg(np.mean)
        count_feature1 = data.groupby('feature1').count()
        count_feature2 = data.groupby('feature2').count()

        pc_1 = count_feature1['feature1.0'] / numrecord
        pc_2 = count_feature1['feature1.1'] / numrecord
        pc_3 = count_feature1['feature2.0'] / numrecord
        pc_4 = count_feature1['feature2.1'] / numrecord

        Hoff_tot_pop = mt.sqrt(np.log(delta)/numrecord)
        Hoff_1 = mt.sqrt(np.log(delta)/count_feature1['feature1.0'])
        Hoff_2 = mt.sqrt(np.log(delta)/count_feature1['feature1.1'])
        Hoff_3 = mt.sqrt(np.log(delta)/count_feature1['feature2.0'])
        Hoff_4 = mt.sqrt(np.log(delta)/count_feature1['feature2.1'])




        #def auxiliary():

        margine_feature1 = (pc_1 - Hoff_1) * (max(expected_feature1.to_numpy()[0:n_candidates]) - Hoff_1) + (pc_2 - Hoff_2) * (max(expected_feature1.to_numpy()[n_candidates:]) - Hoff_2) - opt_aggregate_arm

        margine_feature2 = (pc_3 - Hoff_3) * (max(expected_feature2.to_numpy()[0:n_candidates]) - Hoff_3) + (pc_4 - Hoff_4) * (max(expected_feature2.to_numpy()[n_candidates:]) - Hoff_4) - opt_aggregate_arm


        if margine_feature1 > 0 and margine_feature1 > margine_feature2:
            pass

        if margine_feature2 >0:
            pass
        else:
            pass


