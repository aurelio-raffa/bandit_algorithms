from source.optimization.__dependencies import *


def __moving_window_maximizer(top_slice, temporary_row, temp_row_campaign, budget_values, pedantic=False):
    """
    auxiliary  function to compute a new row from
    :param top_slice: the last computed row in the computation matrix
    :param temporary_row: the current temporary row
    :param temp_row_campaign: the index of the campaign encoded by the temporary row
    :param budget_values: the budget values, common to each subcampaign
    :param pedantic: boolean parameter to print the steps of the algorithm, for testing purposes only
    :return: the value of the newly computed row and the corresponding allocation for each campaign
    """
    n_campaigns = top_slice.shape[0]
    n_budgets = top_slice.shape[1]
    new_slice = np.zeros(top_slice.shape)
    for current_index in range(n_budgets):
        current_budget = budget_values[current_index]
        if pedantic:
            print('\n\tconsidering budget {}:'.format(current_budget))
        optimum = -np.inf
        allocation = -np.inf * np.ones((n_campaigns, 1))
        for old_slice_budget in range(n_budgets):
            for new_campaign_budget in range(n_budgets):
                if budget_values[old_slice_budget] + budget_values[new_campaign_budget] == current_budget:
                    if pedantic:
                        print(
                            '\t\toption encountered: {} from previous configuration, {} from current campaign'.format(
                                budget_values[old_slice_budget], budget_values[new_campaign_budget]))
                    if top_slice[0, old_slice_budget] + temporary_row[new_campaign_budget] > optimum:
                        if pedantic:
                            print(
                                '\t\t\t» objective improving from {} to {}'.format(
                                    optimum, top_slice[0, old_slice_budget] + temporary_row[new_campaign_budget]))
                        optimum = top_slice[0, old_slice_budget] + temporary_row[new_campaign_budget]
                        allocation = top_slice[1:n_campaigns+1, old_slice_budget]
                        allocation[temp_row_campaign] = budget_values[new_campaign_budget]
                        if pedantic:
                            print('\t\t\t» new configuration:', allocation)
                    elif pedantic:
                        print('\t\t\t» no objective improvement')
                    break
        new_slice[0, current_index] = optimum
        new_slice[1:n_campaigns+1, current_index] = allocation

    return new_slice


def budget_optimizer(bb_matrices, budget_values, pedantic=False):
    """
    funcion computing the optimal budget allocation given
    :param bb_matrices: a list of budget (rows) vs, bid (columns) matrices, each representing
        the values v_j*n_j for every possible budget y_j and bid x_j of subcampaign j;
        the possible budget values must be shared among all matrices (use -inf to flag banned possibilities)
    :param budget_values: a vector containing the values for the budgets common to each subcampaign
    :param pedantic: boolean parameter to print the steps of the algorithm, for testing purposes only
    :return: a vector containing the optimal budget allocation for each subcampaign j
    """
    n_campaigns = len(bb_matrices)
    n_budgets = bb_matrices[0].shape[0]
    base_matrix = np.zeros((n_campaigns, n_budgets))
    t_start = time() if pedantic else 0
    for bb_mat, index in zip(bb_matrices, range(n_campaigns)):
        base_matrix[index, :] = np.max(bb_mat, axis=1)
    if pedantic:
        print('original matrices:', *bb_matrices, '\nbase matrix:', base_matrix, sep='\n')

    computation_matrix = np.zeros((n_campaigns+1, n_campaigns+1, n_budgets))
    for temporary_row, index in zip(base_matrix, range(1, n_campaigns+1)):
        new_slice = __moving_window_maximizer(
            computation_matrix[index-1, :, :],
            temporary_row,
            index - 1,
            budget_values,
            pedantic)
        if pedantic:
            print('\nadded subcampaign {}'.format(index-1), new_slice, sep='\n')
        computation_matrix[index, :, :] = new_slice

    best_allocation_index = np.argmax(computation_matrix[n_campaigns, 0, :])
    best_value = computation_matrix[n_campaigns, 0, best_allocation_index]
    best_allocation = computation_matrix[n_campaigns, 1:n_campaigns+1, best_allocation_index]
    if pedantic:
        t_end = time()
        print(
            '\noptimal configuration:',
            best_allocation,
            'value of the configuration: {}'.format(best_value),
            '\n[optimization problem solved in {0:.3f} seconds]'.format(t_end-t_start),
            sep='\n')

    return best_allocation, best_value

