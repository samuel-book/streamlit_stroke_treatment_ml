"""
Main calculations.
"""
# Imports:
import numpy as np
import pandas as pd
import shap


def predict_treatment(X, synthetic, model):
    stroke_team = synthetic['Stroke team']
    Highlighted_team = synthetic['Highlighted team']
    benchmark_rank = synthetic['Benchmark rank']

    probs = model.predict_proba(X)[:, 1]
    results = pd.DataFrame()
    results['Stroke team'] = stroke_team
    results['Highlighted team'] = Highlighted_team
    results['Benchmark rank'] = benchmark_rank
    results['Probability'] = probs
    results['Probability_perc'] = probs*100.0
    results['Thrombolyse'] = probs >= 0.5
    results['Index'] = np.arange(len(results))

    sorted_results = results.\
        sort_values('Probability', ascending=False)

    # Add column of sorted index:
    sorted_results['Sorted rank'] = np.arange(len(results)) + 1

    return sorted_results


def find_shapley_values(explainer_probability, X):
    # Get Shapley values along with base and features
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
        clustering=sv.clustering,
        display_data=sv.display_data,
        error_std=sv.error_std,
        feature_names=sv.feature_names,
        hierarchical_values=sv.hierarchical_values,
        instance_names=sv.instance_names,
        lower_bounds=sv.lower_bounds,
        main_effects=sv.main_effects,
        output_indexes=sv.output_indexes,
        output_names=sv.output_names,
        upper_bounds=sv.upper_bounds,
        values=sv.values
    )
    return sv_fake
