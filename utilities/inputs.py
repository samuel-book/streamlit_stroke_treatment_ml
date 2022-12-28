"""
Import data from file.
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle


def write_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    with open(filename, 'r', encoding="utf-8") as f:
        text_to_print = f.readlines()[head_lines_to_skip:]

    # Turn the list of all of the lines into one long string
    # by joining them up with an empty '' string in between each pair.
    text_to_print = ''.join(text_to_print)

    # Write the text in streamlit.
    st.markdown(f"""{text_to_print}""")


@st.cache
def import_patient_data():
    synthetic = pd.read_csv('./data/synthetic_10_features.csv')
    return synthetic


@st.cache
def import_benchmark_data():
    all_teams_and_probs = pd.read_csv('./data/hospital_10k_thrombolysis.csv')
    # Add an index row to rank the teams:
    all_teams_and_probs['Rank'] = \
        np.arange(1, len(all_teams_and_probs['stroke_team'])+1)
    return all_teams_and_probs


def build_X(user_inputs_dict, stroke_teams_list):
    """
    """
    # Banished this call to build_dataframe_from_inputs() to this
    # function so the "synthetic" array doesn't sit in memory.
    synthetic = build_dataframe_from_inputs(
        user_inputs_dict, stroke_teams_list)
    # Store the column names:
    headers_synthetic = tuple(synthetic.columns)

    # Make a copy of this data that is ready for the model.
    # The same data except the Stroke Team column is one-hot-encoded.
    X = one_hot_encode_data(synthetic)
    # Store the column names:
    headers_X = tuple(X.columns)
    return X, headers_X, headers_synthetic


def one_hot_encode_data(synthetic):
    # One-hot encode hospitals
    # Keep copy of original, with 'Stroke team' not one-hot encoded
    X = synthetic.copy(deep=True)

    # One-hot encode 'Stroke team'
    X_hosp = pd.get_dummies(X['Stroke team'], prefix='team')
    X = pd.concat([X, X_hosp], axis=1)
    X.drop('Stroke team', axis=1, inplace=True)

    return X


@st.cache
def read_stroke_teams_from_file():
    stroke_teams = pd.read_csv('./data/stroke_teams.csv')
    stroke_teams = stroke_teams.values.ravel()
    return stroke_teams


def build_dataframe_from_inputs(dict, stroke_teams_list):
    # First build a 2D array where each row is the patient details.
    # Column headings:
    headers = np.array([
        'Arrival-to-scan time',
        'Infarction',
        'Stroke severity',
        'Precise onset time',
        'Prior disability level',
        'Stroke team',
        'Use of AF anticoagulents',
        'Onset-to-arrival time',
        'Onset during sleep',
        'Age'
    ])

    # One row of the array:
    row = np.array([
        dict['arrival_to_scan_time'],
        dict['infarction'],
        dict['stroke_severity'],
        dict['onset_time_precise'],
        dict['prior_disability'],
        'temp',  # Stroke team
        dict['anticoag'],
        dict['onset_to_arrival_time'],
        dict['onset_during_sleep'],
        dict['age']
        ], dtype=object)

    # Repeat these row values for the number of stroke teams:
    table = np.tile(row, len(stroke_teams_list))
    # Reshape to a 2D array:
    table = table.reshape(len(stroke_teams_list), len(headers))
    # Update the "Stroke team" column with the names:
    table[:, 5] = stroke_teams_list

    # Turn this array into a DataFrame with labelled columns.
    df = pd.DataFrame(table, columns=headers)
    return df


@st.cache(hash_funcs={'builtins.dict': lambda _: None})
def load_pretrained_model():
    # Load XGB Model
    filename = ('./data/model.p')
    with open(filename, 'rb') as filehandler:
        model = pickle.load(filehandler)
    return model


@st.cache(hash_funcs={'builtins.dict': lambda _: None})
def load_explainer():
    # Load SHAP explainers
    filename = ('./data/shap_explainer.p')
    with open(filename, 'rb') as filehandler:
        explainer = pickle.load(filehandler)
    return explainer


@st.cache(hash_funcs={'builtins.dict': lambda _: None})
def load_explainer_probability():
    filename = ('./data/shap_explainer_probability.p')
    with open(filename, 'rb') as filehandler:
        explainer_probability = pickle.load(filehandler)
    return explainer_probability
