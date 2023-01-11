"""
Main calculations.
"""
# Imports:
import numpy as np
import pandas as pd
import shap
import streamlit as st


def predict_treatment(
        X, model, stroke_teams_list, highlighted_teams_list,
        benchmark_rank_list, hb_teams_list
        ):
    probs_list = model.predict_proba(X)[:, 1]

    # Put everything into a DataFrame:
    results = pd.DataFrame()
    results['Stroke team'] = stroke_teams_list
    results['Highlighted team'] = highlighted_teams_list
    results['Benchmark rank'] = benchmark_rank_list
    results['HB team'] = hb_teams_list
    results['Probability'] = probs_list
    results['Probability_perc'] = probs_list*100.0
    results['Thrombolyse'] = probs_list >= 0.5
    results['Index'] = np.arange(len(results))

    sorted_results = results.\
        sort_values('Probability', ascending=False)

    # Add column of sorted index:
    sorted_results['Sorted rank'] = np.arange(1, len(results) + 1)

    # # Add column of '*' for benchmark rank in top 30:
    # benchmark_bool = []
    # for i in sorted_results['Benchmark rank']:
    #     val = '\U00002605' if i <= 30 else ''
    #     benchmark_bool.append(val)
    # sorted_results['Benchmark'] = benchmark_bool

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(sorted_results), 'No ')
    thrombolyse_str[np.where(sorted_results['Thrombolyse'])] = 'Yes'
    sorted_results['Thrombolyse_str'] = thrombolyse_str

    return sorted_results


def find_shapley_values(explainer_probability, X):
    """
    Inputs:
    explainer_probability - imported model.
    X                     - data array.

    Returns:
    shap_values_probability_extended - All of the output data.
    shap_values_probability          - np.array of just the
                                       probabilities.
    """
    # # Get Shapley values along with base and features
    shap_values_probability_extended = explainer_probability(X)

    # Shap values exist for each classification in a Tree
    shap_values_probability = shap_values_probability_extended.values
    return shap_values_probability_extended, shap_values_probability


def convert_explainer_01_to_noyes(sv):
    """
    Change some SHAP explainer data values so that input 0/1 features
    are changed to "no" or "yes" strings for display on waterfall.

    Input:
    sv - a SHAP explainer object.

    Returns:
    sv_fake - a copy of the object with some data swapped.
    """
    # Take the input data from the existing object:
    data_yn = np.copy(sv.data)

    # Swap out the data for these features:
    expected_features = [
        'Infarction',
        'Precise onset time',
        'Use of AF anticoagulants',
        'Onset during sleep'
        ]
    # Find where these features are in the list:
    inds = [sv.feature_names.index(feature) for feature in expected_features]

    # Also update all of the "team_" entries.
    # Find where they are in the list:
    for i, feature in enumerate(sv.feature_names):
        if feature[:5] == 'team_':
            inds.append(i)

    # Update the data behind those features:
    for i in inds:
        data_yn[i] = 'No' if data_yn[i] == 0 else 'Yes'

    # Make a new explainer object with the new data:
    sv_fake = shap.Explanation(
        # Changed data:
        data=data_yn,
        # Everything else copied directly from the start object:
        base_values=sv.base_values,
        # clustering=sv.clustering,
        # display_data=sv.display_data,
        # error_std=sv.error_std,
        feature_names=sv.feature_names,
        # hierarchical_values=sv.hierarchical_values,
        # instance_names=sv.instance_names,
        # lower_bounds=sv.lower_bounds,
        # main_effects=sv.main_effects,
        # output_indexes=sv.output_indexes,
        # output_names=sv.output_names,
        # upper_bounds=sv.upper_bounds,
        values=sv.values
    )
    return sv_fake


def make_heat_grids(headers, stroke_team_list, sorted_inds,
                    shap_values_probability):
    # Experiment
    n_teams = shap_values_probability.shape[0]
    # n_features = len(shap_values_probability_extended[0].values)
    grid = np.transpose(shap_values_probability)

    # Expect most of the mismatched one-hot-encoded hospitals to make
    # only a tiny contribution to the SHAP. Moosh them down into one
    # column instead.

    # Have 9 features other than teams. Index 9 is the first team.
    ind_first_team = 9

    # Make a new grid and copy over most of the values:
    grid_cat = np.zeros((ind_first_team + 1, n_teams))
    grid_cat[:ind_first_team, :] = grid[:ind_first_team, :]

    # For the remaining column, loop over to pick out the value:
    for i, sorted_ind in enumerate(sorted_inds):
        row = i + ind_first_team
        # Combine all SHAP values into one:
        grid_cat[ind_first_team, i] = np.sum(grid[ind_first_team:, i])

        # # Pick out the value we want:
        # value_of_matching_stroke_team = grid[row, i]
        # # Add the wanted value to the new grid:
        # grid_cat[ind_first_team, i] = value_of_matching_stroke_team
        # # Take the sum of all of the team values:
        # value_of_merged_stroke_teams = np.sum(grid[ind_first_team:, i])
        # # Subtract the value we want:
        # value_of_merged_stroke_teams -= value_of_matching_stroke_team
        # # And store this as a merged "all other teams" value:
        # grid_cat[ind_first_team+1, i] = value_of_merged_stroke_teams

    # Multiply values by 100 to get probability in percent:
    grid_cat *= 100.0

    # Sort the values into the same order as sorted_results:
    grid_cat_sorted = grid_cat[:, sorted_inds]

    headers = np.append(headers[:9], 'Stroke team attended')
    # headers = np.append(headers, 'Stroke teams not attended')

    # 2D grid of stroke_teams:
    stroke_team_2d = np.tile(
        stroke_team_list, len(headers)).reshape(grid_cat_sorted.shape)
    return grid, grid_cat_sorted, stroke_team_2d, headers


def make_waterfall_df(
        grid_cat_sorted, headers, stroke_team_list, highlighted_team_list,
        hb_team_list, patient_data_waterfall, base_values=0.2995270168908044
        ):
    base_values_perc = 100.0 * base_values

    grid_waterfall = np.copy(grid_cat_sorted)
    # Sort the grid in order of increasing standard deviation:
    inds_std = np.argsort(np.std(grid_waterfall, axis=1))
    grid_waterfall = grid_waterfall[inds_std, :]
    features_waterfall = headers[inds_std]
    patient_data_waterfall = patient_data_waterfall[inds_std]

    # Add a row for the starting probability:
    grid_waterfall = np.vstack(
        (np.zeros(grid_waterfall.shape[1]), grid_waterfall))
    # Make a cumulative probability line for each team:
    grid_waterfall_cumsum = np.cumsum(grid_waterfall, axis=0)
    # Add the starting probability to all values:
    grid_waterfall_cumsum += base_values_perc
    # Keep final probabilities separate:
    final_probs_list = grid_waterfall_cumsum[-1, :]
    # Feature names:
    features_waterfall = np.append('Base probability', features_waterfall)
    # features_waterfall = np.append(features_waterfall, 'Final probability')

    # Get the grid into a better format for the data frame:
    # Column containing shifts in probabilities for each feature:
    column_probs_shifts = grid_waterfall.T.ravel()
    # Column containing cumulative probabilities (for x axis):
    column_probs_cum = grid_waterfall_cumsum.T.ravel()
    # Column of feature names:
    column_features = np.tile(features_waterfall, len(stroke_team_list))
    # Column of the rank:
    a = np.arange(1, len(stroke_team_list)+1)
    column_sorted_rank = np.tile(a, len(features_waterfall))\
        .reshape(len(features_waterfall), len(a)).T.ravel()
    # Column of stroke teams:
    column_stroke_team = np.tile(stroke_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(stroke_team_list)).T.ravel()
    # Column of highlighted teams:
    column_highlighted_teams = np.tile(
        highlighted_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(highlighted_team_list)).T.ravel()
    # Column of highlighted/benchmark teams:
    column_hb_teams = np.tile(
        hb_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(hb_team_list)).T.ravel()
    # Column of final probability of thrombolysis:
    column_probs_final = np.tile(final_probs_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(final_probs_list)).T.ravel()

    # Put this into a data frame:
    df_waterfalls = pd.DataFrame()
    df_waterfalls['Sorted rank'] = column_sorted_rank
    df_waterfalls['Stroke team'] = column_stroke_team
    df_waterfalls['Probabilities'] = column_probs_cum
    df_waterfalls['Prob shift'] = column_probs_shifts
    df_waterfalls['Prob final'] = column_probs_final
    df_waterfalls['Features'] = column_features
    df_waterfalls['Highlighted team'] = column_highlighted_teams
    df_waterfalls['HB team'] = column_hb_teams
    return df_waterfalls, final_probs_list, patient_data_waterfall

