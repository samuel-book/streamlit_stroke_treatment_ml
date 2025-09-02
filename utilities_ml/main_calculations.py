"""
Main calculations.
"""
# Imports:
import numpy as np
import pandas as pd
import shap
import streamlit as st

import utilities_ml.main_calculations
from utilities_ml.fixed_params import plain_str, bench_str, model_version, n_benchmark_teams


def predict_treatment(
        X, model, stroke_teams_list, highlighted_teams_list,
        benchmark_rank_list, hb_teams_list,
        allow_maybe=False, prob_maybe_min=0.1, prob_maybe_max=0.9
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

    thromb_decision = np.full(probs_list.shape, 0)
    if allow_maybe:
        thromb_decision[probs_list >= prob_maybe_max] = 2       # Yes
        thromb_decision[((probs_list < prob_maybe_max) &
                        (probs_list > prob_maybe_min))] = 1     # Maybe
        thromb_decision[probs_list <= prob_maybe_min] = 0       # No
    else:
        thromb_decision[probs_list >= 0.5] = 2             # Yes
        thromb_decision[probs_list < 0.5] = 0              # No

    results['Thrombolyse'] = thromb_decision
    results['Index'] = np.arange(len(results))

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(results), 'Maybe')
    thrombolyse_str[np.where(results['Thrombolyse'] == 2)] = 'Yes  '
    thrombolyse_str[np.where(results['Thrombolyse'] == 0)] = 'No   '
    results['Thrombolyse_str'] = thrombolyse_str

    sorted_results = results.sort_values('Probability', ascending=False)

    # Add column of sorted index:
    sorted_results['Sorted rank'] = np.arange(1, len(results) + 1)

    # # Add column of '*' for benchmark rank in top 30:
    # benchmark_bool = []
    # for i in sorted_results['Benchmark rank']:
    #     val = '\U00002605' if i <= 30 else ''
    #     benchmark_bool.append(val)
    # sorted_results['Benchmark'] = benchmark_bool

    return sorted_results


def predict_treatment_proto(
        df_proto, X_proto, model,
        allow_maybe=False, prob_maybe_min=0.1, prob_maybe_max=0.9
        ):
    probs_list = model.predict_proba(X_proto)[:, 1]

    # Put everything into a DataFrame:
    results = pd.DataFrame()
    results['Patient prototype'] = df_proto['Patient prototype']
    results['Stroke team'] = df_proto['stroke_team']
    results['HB team'] = df_proto['hb_team']
    results['Probability'] = probs_list
    results['Probability_perc'] = probs_list*100.0

    thromb_decision = np.full(probs_list.shape, 0)
    if allow_maybe:
        thromb_decision[probs_list >= prob_maybe_max] = 2       # Yes
        thromb_decision[((probs_list < prob_maybe_max) &
                        (probs_list > prob_maybe_min))] = 1     # Maybe
        thromb_decision[probs_list <= prob_maybe_min] = 0       # No
    else:
        thromb_decision[probs_list >= 0.5] = 2             # Yes
        thromb_decision[probs_list < 0.5] = 0              # No

    results['Thrombolyse'] = thromb_decision

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(results), 'Maybe')
    thrombolyse_str[np.where(results['Thrombolyse'] == 2)] = 'Yes  '
    thrombolyse_str[np.where(results['Thrombolyse'] == 0)] = 'No   '
    results['Thrombolyse_str'] = thrombolyse_str

    return results


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


def convert_explainer_01_to_noyes(sv, model_version='SAMueL-1'):
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
    # if 'SAMueL-1' in model_version:
    #     expected_features = [
    #         'Infarction',
    #         'Precise onset time',
    #         'Use of AF anticoagulants',
    #         'Onset during sleep'
    #         ]
    # else:
    #     expected_features = [
    #         'infarction',
    #         'precise onset known',
    #         'use of AF anticoagulants',
    #         'onset during sleep'
    #         ]

    expected_features = [
        'infarction',
        'precise_onset_known',
        'afib_anticoagulant',
        'onset_during_sleep'
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


def calculations_for_shap_values(
        sorted_results,
        explainer_probability,
        X,
        hb_teams_input,
        headers_X,
        starting_probabilities
        ):

    # ----- Shapley probabilities -----
    # Make Shapley values for all teams:
    shap_values_probability_extended_all, shap_values_probability_all = \
        utilities_ml.main_calculations.find_shapley_values(
            explainer_probability, X)

    # Make separate arrays of the Shapley values for certain teams.
    # Get indices of highest, most average, and lowest probability teams.
    index_high = sorted_results.iloc[0]['Index']
    index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
    index_low = sorted_results.iloc[-1]['Index']
    indices_high_mid_low = [index_high, index_mid, index_low]
    # Get indices of highlighted teams:
    indices_highlighted = []
    for team in hb_teams_input:
        if plain_str not in team and bench_str not in team:
            # If it's not the default benchmark or non-benchmark
            # team label, then add this index to the list:
            ind_team = sorted_results['Index'][
                sorted_results['HB team'] == team].values[0]
            indices_highlighted.append(ind_team)

    # Shapley values for the high/mid/low indices:
    shap_values_probability_extended_high_mid_low = \
        shap_values_probability_extended_all[indices_high_mid_low]
    # Shapley values for the highlighted indices:
    shap_values_probability_extended_highlighted = \
        shap_values_probability_extended_all[indices_highlighted]

    # ----- Other grids and dataframes for Shap probabilities:
    # Get big SHAP probability grid:
    grid, grid_cat_sorted, stroke_team_2d, headers = \
        utilities_ml.main_calculations.make_heat_grids(
            headers_X,
            sorted_results['Stroke team'],
            sorted_results['Index'],
            shap_values_probability_all
            )
    # These grids have teams in the same order as sorted_results.

    # Pick out the subset of benchmark teams:
    inds_bench = np.where(
        sorted_results['Benchmark rank'].to_numpy() <= n_benchmark_teams)[0]
    inds_nonbench = np.where(
        sorted_results['Benchmark rank'].to_numpy() > n_benchmark_teams)[0]
    # Make separate grids of just the benchmark or non-benchmark teams:
    grid_cat_bench = grid_cat_sorted[:, inds_bench]
    grid_cat_nonbench = grid_cat_sorted[:, inds_nonbench]

    # Make a list of the input patient data for labelling
    # features+values on e.g. the combined waterfall plot.
    # Pull out the feature values:
    patient_data_waterfall = X.iloc[0][:9].to_numpy()
    # Add empty value for stroke team attended:
    patient_data_waterfall = np.append(patient_data_waterfall, '')
    # Find which values are 0/1 choice and can be changed to no/yes:
    if 'SAMueL-1' in model_version:
        features_yn = [
            'Infarction',
            'Precise onset time',
            'Use of AF anticoagulants',
            'Onset during sleep',
        ]
    else:
        features_yn = [
            'infarction',
            'precise onset known',
            'use of AF anticoagulants',
            'onset during sleep'
        ]
    for feature in features_yn:
        i = np.where(np.array(headers_X) == feature)[0]
        # Annoying nested list to pacify DeprecationWarning for
        # checking for element of empty array.
        if patient_data_waterfall[i].size > 0:
            if patient_data_waterfall[i] > 0:
                patient_data_waterfall[i] = 'Yes'
            else:
                patient_data_waterfall[i] = 'No'
        else:
            patient_data_waterfall[i] = 'No'
    # Resulting list format e.g.:
    #     [15, 'Yes', 15, 'Yes', 0, 'No', 90, 'No', 72.5, '']
    # where headers_X provides the feature names to match the values.

    # Make dataframe for combo waterfalls:
    # patient_data_waterfall is returned here with the order of the
    # values switched to match the order the features are plotted in
    # in the combo waterfall.
    df_waterfalls, final_probs, patient_data_waterfall = \
        utilities_ml.main_calculations.make_waterfall_df(
            grid_cat_sorted,
            headers,
            sorted_results['Stroke team'],
            sorted_results['Highlighted team'],
            sorted_results['HB team'],
            patient_data_waterfall,
            base_values=starting_probabilities
            )
    # Columns of df_waterfalls:
    #     Sorted rank
    #     Stroke team
    #     Probabilities
    #     Prob shift
    #     Prob final
    #     Features
    #     Highlighted team
    #     HB team
    return (
        indices_highlighted,
        shap_values_probability_extended_highlighted,
        indices_highlighted,
        df_waterfalls,
        final_probs,
        patient_data_waterfall,
        grid_cat_sorted,
        grid_cat_bench,
        grid_cat_nonbench,
        headers,                    
        shap_values_probability_extended_high_mid_low,
        indices_high_mid_low
        )
