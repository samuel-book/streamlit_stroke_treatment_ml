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
        benchmark_rank_list
        ):
    probs_list = model.predict_proba(X)[:, 1]

    # Put everything into a DataFrame:
    results = pd.DataFrame()
    results['Stroke team'] = stroke_teams_list
    results['Highlighted team'] = highlighted_teams_list
    results['Benchmark rank'] = benchmark_rank_list
    results['Probability'] = probs_list
    results['Probability_perc'] = probs_list*100.0
    results['Thrombolyse'] = probs_list >= 0.5
    results['Index'] = np.arange(len(results))

    sorted_results = results.\
        sort_values('Probability', ascending=False)

    # Add column of sorted index:
    sorted_results['Sorted rank'] = np.arange(1, len(results) + 1)

    # Add column of '*' for benchmark rank in top 30:
    benchmark_bool = []
    for i in sorted_results['Benchmark rank']:
        val = '\U00002605' if i <= 30 else ''
        benchmark_bool.append(val)
    sorted_results['Benchmark'] = benchmark_bool

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(sorted_results), 'No ')
    thrombolyse_str[np.where(sorted_results['Thrombolyse'])] = 'Yes'
    sorted_results['Thrombolyse_str'] = thrombolyse_str

    return sorted_results


def predict_treatment_batches(
        X, model, stroke_teams_list, highlighted_teams_list,
        benchmark_rank_list
        ):

    # Split the data into batches to reduce CPU usage
    probs_list = np.array([])
    n_batches = 5
    rows_per_batch = int(X.shape[0] / n_batches)
    # First go round the loop:
    start_ind = 0
    end_ind = rows_per_batch

    import time
    for batch in range(n_batches):
        if batch == n_batches - 1:
            # For final pass, use all of the remaining values:
            end_ind = X.shape[0]
        probs = model.predict_proba(X)[start_ind:end_ind, 1]
        probs_list = np.concatenate((probs_list, probs))
        # Set up for the next go round the loop:
        start_ind += rows_per_batch
        end_ind += rows_per_batch
        # Wait for a moment to reduce spike in CPU usage
        time.sleep(0.1)

    # Put everything into a DataFrame:
    results = pd.DataFrame()
    results['Stroke team'] = stroke_teams_list
    results['Highlighted team'] = highlighted_teams_list
    results['Benchmark rank'] = benchmark_rank_list
    results['Probability'] = probs_list
    results['Probability_perc'] = probs_list*100.0
    results['Thrombolyse'] = probs_list >= 0.5
    results['Index'] = np.arange(len(results))

    sorted_results = results.\
        sort_values('Probability', ascending=False)

    # Add column of sorted index:
    sorted_results['Sorted rank'] = np.arange(1, len(results) + 1)

    # Add column of '*' for benchmark rank in top 30:
    benchmark_bool = []
    for i in sorted_results['Benchmark rank']:
        val = '\U00002605' if i <= 30 else ''
        benchmark_bool.append(val)
    sorted_results['Benchmark'] = benchmark_bool

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(sorted_results), 'No ')
    thrombolyse_str[np.where(sorted_results['Thrombolyse'])] = 'Yes'
    sorted_results['Thrombolyse_str'] = thrombolyse_str

    return sorted_results


    # Add column of '*' for benchmark rank in top 30:
    benchmark_bool = []
    for i in sorted_results['Benchmark rank']:
        val = '\U00002605' if i <= 30 else ''
        benchmark_bool.append(val)
    sorted_results['Benchmark'] = benchmark_bool

    # Add column of str to print when thrombolysed or not
    thrombolyse_str = np.full(len(sorted_results), 'No ')
    thrombolyse_str[np.where(sorted_results['Thrombolyse'])] = 'Yes'
    sorted_results['Thrombolyse_str'] = thrombolyse_str

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


def find_shapley_values_batched(explainer_probability, X):
    # # Get Shapley values along with base and features
    shap_values_probability_extended = explainer_probability(X)

    # Split the data into batches to reduce CPU usage
    # shap_list = np.array([])
    n_batches = 8
    rows_per_batch = int(X.shape[0] / n_batches)
    # First go round the loop:
    start_ind = 0
    end_ind = rows_per_batch


    import time
    shap_list = []
    for batch in range(n_batches):
        if batch == n_batches - 1:
            # For final pass, use all of the remaining values:
            end_ind = X.shape[0]
        shap_values_probability_extended = \
            explainer_probability(X[start_ind:end_ind])
        shap_list.append(shap_values_probability_extended)
        # Set up for the next go round the loop:
        start_ind += rows_per_batch
        end_ind += rows_per_batch
        # # Wait for a moment to reduce spike in CPU usage
        # time.sleep(0.5)

    shap_list = combine_batch_shap(shap_list)
    # Shap values exist for each classification in a Tree
    shap_values_probability = shap_list.values

    return shap_list, shap_values_probability


def combine_batch_shap(shap_list):

    for i, sl in enumerate(shap_list):
        if i == 0:
            data = sl.data
            base_values = sl.base_values
            feature_names = sl.feature_names
            values = sl.values
        else:
            data = np.concatenate((data, sl.data))
            base_values = np.concatenate((base_values, sl.base_values))
            values = np.concatenate((values, sl.values))

    # Make a new explainer object with the new data:
    sv_combo = shap.Explanation(
        data=data,
        base_values=base_values,
        feature_names=feature_names,
        values=values
    )
    return sv_combo


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
        'Use of AF anticoagulents',
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
