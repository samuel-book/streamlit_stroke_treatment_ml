"""
Import data from file.
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle

try:
    stroke_teams_test = pd.read_csv('./data_ml/stroke_teams.csv')
    dir = './'
except FileNotFoundError:
    dir = 'streamlit_stroke_treatment_ml/'

from utilities_ml.fixed_params import plain_str, bench_str, \
    display_name_of_default_highlighted_team, default_highlighted_team


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


# @st.cache
# def import_patient_data():
#     synthetic = pd.read_csv(dir + 'data_ml/synthetic_10_features.csv')
#     return synthetic


@st.cache_data()
def import_benchmark_data():
    all_teams_and_probs = pd.read_csv(dir + 'data_ml/hospital_10k_thrombolysis.csv')
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

    # Make a copy of this data that is ready for the model.
    # The same data except the Stroke Team column is one-hot-encoded.
    X = one_hot_encode_data(synthetic)
    # Store the column names:
    headers_X = tuple(X.columns)
    return X, headers_X


def one_hot_encode_data(synthetic):
    # One-hot encode hospitals
    # Keep copy of original, with 'Stroke team' not one-hot encoded
    X = synthetic.copy(deep=True)

    # One-hot encode 'Stroke team'
    X_hosp = pd.get_dummies(X['Stroke team'], prefix='team')
    X = pd.concat([X, X_hosp], axis=1)
    X.drop('Stroke team', axis=1, inplace=True)

    return X


@st.cache_data()
def read_stroke_teams_from_file():
    stroke_teams = pd.read_csv(dir + 'data_ml/stroke_teams.csv')
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
        'Use of AF anticoagulants',
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


@st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_pretrained_model():
    # Load XGB Model
    filename = (dir + 'data_ml/model.p')
    with open(filename, 'rb') as filehandler:
        model = pickle.load(filehandler)
    return model


@st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_explainer():
    # Load SHAP explainers
    filename = (dir + 'data_ml/shap_explainer.p')
    with open(filename, 'rb') as filehandler:
        explainer = pickle.load(filehandler)
    return explainer


@st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_explainer_probability():
    filename = (dir + 'data_ml/shap_explainer_probability.p')
    with open(filename, 'rb') as filehandler:
        explainer_probability = pickle.load(filehandler)
    return explainer_probability


def find_highlighted_hb_teams(stroke_teams_list, inds_benchmark, highlighted_teams_input):
    # Create a "Highlighted teams" column for the sorted_results.
    # Start off with everything '-' (NOT as plain_str):
    highlighted_teams_list = np.array(
        ['-' for team in stroke_teams_list], dtype=object)
    # Create a combined highlighted and benchmark (HB) column.
    # Initially mark all of the teams with the plain string:
    hb_teams_list = np.array(
        [plain_str for team in stroke_teams_list], dtype=object)
    # Then change the benchmark teams to the benchmark string:
    hb_teams_list[inds_benchmark] = bench_str

    # Keep a shorter list of all of the unique values in hb_teams_list:
    hb_teams_input = [plain_str, bench_str]

    # In the following loop, update the highlighted column, HB column,
    # and HB input list with the input highlighted teams:
    for team in highlighted_teams_input:
        if team == display_name_of_default_highlighted_team:
            team = default_highlighted_team
        # Find where this team is in the full list:
        ind_t = np.argwhere(stroke_teams_list == team)[0][0]
        # Set the highlighted teams column here to just the name
        # of the highlighted team:
        highlighted_teams_list[ind_t] = team
        # Check whether this team is also a benchmark team.
        # If it is, add this string (unicode for a star).
        if ind_t in inds_benchmark:
            team = team + ' \U00002605'
        # Update this team in the HB column...
        hb_teams_list[ind_t] = team
        # ... and add it to the shorter unique values list.
        hb_teams_input.append(team)
    return highlighted_teams_list, hb_teams_list, hb_teams_input
