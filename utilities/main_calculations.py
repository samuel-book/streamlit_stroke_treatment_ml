"""
Main calculations.
"""
# Imports:
import numpy as np
import pandas as pd


def predict_treatment(X, synthetic, model):
    stroke_team = synthetic['Stroke team']
    probs = model.predict_proba(X)[:, 1]
    results = pd.DataFrame()
    results['Stroke team'] = stroke_team
    results['Probability'] = probs
    results['Thrombolyse'] = probs >= 0.5
    results['Index'] = np.arange(len(results))

    sorted_results = results.\
        sort_values('Probability', ascending=False)
    return sorted_results


def find_shapley_values(explainer_probability, X):
    # Get Shapley values along with base and features
    shap_values_probability_extended = explainer_probability(X)
    # Shap values exist for each classification in a Tree
    shap_values_probability = shap_values_probability_extended.values
    return shap_values_probability_extended, shap_values_probability
