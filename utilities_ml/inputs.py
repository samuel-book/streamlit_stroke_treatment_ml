"""
Import data from file.
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle

import utilities_ml.container_inputs

from utilities_ml.plot_utils import remove_old_colours_for_highlights, \
                                    choose_colours_for_highlights

try:
    stroke_teams_test = pd.read_csv('./data_ml/stroke_teams.csv')
    dir = './'
except FileNotFoundError:
    dir = 'streamlit_stroke_treatment_ml/'

from utilities_ml.fixed_params import plain_str, bench_str, \
    model_version, stroke_teams_file, stroke_team_col, \
    benchmark_filename, benchmark_team_column, n_benchmark_teams, \
    default_highlighted_team, display_name_of_default_highlighted_team


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


def read_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    with open(filename, 'r', encoding="utf-8") as f:
        all_lines = f.readlines()[head_lines_to_skip:]

    # Split by subsubsection:
    list_of_bits = []
    str_here = ''
    for line in all_lines:
        if line.strip()[:3] == '###':
            # If this is a new subsection, save the old string
            # to the list of bits and create a new string.
            if len(str_here.strip()) > 0:
                list_of_bits.append(str_here)
            str_here = line
        else:
            str_here += line
    # Catch the final subsubsection:
    if len(str_here.strip()) > 0:
        list_of_bits.append(str_here)

    return list_of_bits


# @st.cache
# def import_patient_data():
#     synthetic = pd.read_csv(dir + 'data_ml/synthetic_10_features.csv')
#     return synthetic


@st.cache_data()
def import_benchmark_data(filename='hospital_10k_thrombolysis.csv',
                          team_col='stroke_team'):
    all_teams_and_probs = pd.read_csv(dir + 'data_ml/' + filename)
    # Add an index row to rank the teams:
    all_teams_and_probs['Rank'] = \
        np.arange(1, len(all_teams_and_probs[team_col])+1)
    return all_teams_and_probs


def import_proto_patients(filename='prototype_patients.csv'):
    df_proto = pd.read_csv(dir + 'data_ml/' + filename)
    return df_proto


def import_hospital_data():
    df_hosp = pd.read_csv(dir + 'data_ml/' + 'data_for_sim_anon.csv')
    # df_hosp['stroke_team'] = df_hosp['stroke_team'].astype(str)
    return df_hosp.set_index('stroke_team')


def build_X(
        user_inputs_dict,
        stroke_teams_list,
        stroke_team_col='Stroke team',
        model_version='SAMueL-1'
        ):
    """
    """
    # Banished this call to build_dataframe_from_inputs() to this
    # function so the "synthetic" array doesn't sit in memory.
    synthetic = build_dataframe_from_inputs(
        user_inputs_dict, stroke_teams_list, model_version)

    # Make a copy of this data that is ready for the model.
    # The same data except the Stroke Team column is one-hot-encoded.
    X = one_hot_encode_data(synthetic, one_hot_column=stroke_team_col)
    # Store the column names:
    headers_X = tuple(X.columns)
    return X, headers_X


def one_hot_encode_data(synthetic, one_hot_column='Stroke team'):
    # One-hot encode hospitals
    # Keep copy of original, with 'Stroke team' not one-hot encoded
    X = synthetic.copy(deep=True)

    # One-hot encode 'Stroke team'
    X_hosp = pd.get_dummies(X[one_hot_column], prefix='team')
    X = pd.concat([X, X_hosp], axis=1)
    X.drop(one_hot_column, axis=1, inplace=True)

    return X


@st.cache_data()
def read_stroke_teams_from_file(filename='stroke_teams.csv'):
    stroke_teams = pd.read_csv(dir + 'data_ml/' + filename)
    stroke_teams = stroke_teams.astype(dtype={'team_code': str})
    stroke_teams = stroke_teams.values.ravel()
    return stroke_teams


def build_dataframe_from_inputs(dict, stroke_teams_list, model_type):
    # First build a 2D array where each row is the patient details.

    # if 'SAMueL-1' in model_type:
    #     # Column headings:
    #     headers = np.array([
    #         'Arrival-to-scan time',
    #         'Infarction',
    #         'Stroke severity',
    #         'Precise onset time',
    #         'Prior disability level',
    #         'Stroke team',
    #         'Use of AF anticoagulants',
    #         'Onset-to-arrival time',
    #         'Onset during sleep',
    #         'Age'
    #     ])

    #     # One row of the array:
    #     row = np.array([
    #         dict['arrival_to_scan_time'],
    #         dict['infarction'],
    #         dict['stroke_severity'],
    #         dict['onset_time_precise'],
    #         dict['prior_disability'],
    #         'temp',  # Stroke team
    #         dict['anticoag'],
    #         dict['onset_to_arrival_time'],
    #         dict['onset_during_sleep'],
    #         dict['age']
    #         ], dtype=object)

    #     stroke_team_col = 5
    # else:
    #     # Column headings:
    #     headers = np.array([
    #         'stroke team',
    #         'age',
    #         'infarction',
    #         'stroke severity',
    #         'onset-to-arrival time',
    #         'precise onset known',
    #         'onset during sleep',
    #         'use of AF anticoagulants',
    #         'prior disability',
    #         'arrival-to-scan time',
    #         # 'thrombolysis'
    #     ])

    #     # One row of the array:
    #     row = np.array([
    #         'temp',  # Stroke team
    #         dict['age'],
    #         dict['infarction'],
    #         dict['stroke_severity'],
    #         dict['onset_to_arrival_time'],
    #         dict['onset_time_precise'],
    #         dict['onset_during_sleep'],
    #         dict['anticoag'],
    #         dict['prior_disability'],
    #         dict['arrival_to_scan_time']
    #         ], dtype=object)

    #     stroke_team_col = 0

    # Column headings:
    headers = np.array([
        'stroke_team_id',
        'stroke_severity',
        'prior_disability',
        'age',
        'infarction',
        'onset_to_arrival_time',
        'precise_onset_known',
        'onset_during_sleep',
        'arrival_to_scan_time',
        'afib_anticoagulant'
        # 'thrombolysis'
    ])

    # One row of the array:
    row = np.array([
        'temp',  # Stroke team
        dict['stroke_severity'],
        dict['prior_disability'],
        dict['age'],
        dict['infarction'],
        dict['onset_to_arrival_time'],
        dict['onset_time_precise'],
        dict['onset_during_sleep'],
        dict['arrival_to_scan_time'],
        dict['anticoag']
        ], dtype=object)

    stroke_team_col = 0

    # Repeat these row values for the number of stroke teams:
    table = np.tile(row, len(stroke_teams_list))
    # Reshape to a 2D array:
    table = table.reshape(len(stroke_teams_list), len(headers))
    # Update the "Stroke team" column with the names:
    table[:, stroke_team_col] = stroke_teams_list

    # Turn this array into a DataFrame with labelled columns.
    # Make stroke team ID integers for later sorting -
    # we want team_1, team_2, team_3, ...
    # instead of team_1, team_10, team_100, ...
    df = pd.DataFrame(table, columns=headers).astype(dtype={
        'stroke_team_id': int,
        'stroke_severity': int,
        'prior_disability': int,
        'age': float,
        'infarction': bool,
        'onset_to_arrival_time': float,
        'precise_onset_known': int,
        'onset_during_sleep': int,
        'arrival_to_scan_time': float,
        'afib_anticoagulant': bool
        # 'thrombolysis'
    })

    return df


def build_dataframe_proto(df_proto, all_teams):
    df_proto = df_proto.copy()
    df_proto = df_proto.rename(columns={
        'stroke_team': 'stroke_team_id',
    })
    teams_here = df_proto['stroke_team_id'].unique()
    # First build a 2D array where each row is the patient details.
    # Column headings:
    headers = np.array([
        'stroke_team_id',
        'stroke_severity',
        'prior_disability',
        'age',
        'infarction',
        'onset_to_arrival_time',
        'precise_onset_known',
        'onset_during_sleep',
        'arrival_to_scan_time',
        'afib_anticoagulant'
        # 'thrombolysis'
    ])

    # Turn this array into a DataFrame with labelled columns.
    # Make stroke team ID integers for later sorting -
    # we want team_1, team_2, team_3, ...
    # instead of team_1, team_10, team_100, ...
    df = df_proto[headers].copy().astype(dtype={
        'stroke_team_id': int,
        'stroke_severity': int,
        'prior_disability': int,
        'age': float,
        'infarction': bool,
        'onset_to_arrival_time': float,
        'precise_onset_known': int,
        'onset_during_sleep': int,
        'arrival_to_scan_time': float,
        'afib_anticoagulant': bool
        # 'thrombolysis'
    })

    # Make a copy of this data that is ready for the model.
    # The same data except the Stroke Team column is one-hot-encoded.
    X = one_hot_encode_data(df, one_hot_column='stroke_team_id')
    # Add in ohe columns of the missing teams:
    teams_missing = [f'team_{t}' for t in list(set(all_teams.astype(int)) - set(teams_here.astype(int)))]
    X[teams_missing] = 0

    # Place columns in the order expected by the model:
    column_order = list(headers[1:]) + [f'team_{t}' for t in all_teams]
    X = X[column_order]
    return X


def build_dataframe_outcomes(df_proto, all_teams):
    df_proto = df_proto.copy()
    df_proto = df_proto.rename(columns={
        'stroke_team': 'stroke_team_id',
        'onset_to_thrombolysis_time': 'onset_to_thrombolysis',
        'afib_anticoagulant': 'any_afib_diagnosis',
    })
    teams_here = df_proto['stroke_team_id'].unique()
    # First build a 2D array where each row is the patient details.
    # Column headings:
    headers = np.array([
        'stroke_team_id',
        'prior_disability',
        'stroke_severity',
        'onset_to_thrombolysis',
        'age',
        'precise_onset_known',
        'any_afib_diagnosis',
        # 'discharge_disability'
    ])

    # Turn this array into a DataFrame with labelled columns.
    # Make stroke team ID integers for later sorting -
    # we want team_1, team_2, team_3, ...
    # instead of team_1, team_10, team_100, ...
    df = df_proto[headers].copy().astype(dtype={
        'stroke_team_id': int,
        'stroke_severity': int,
        'prior_disability': int,
        'age': float,
        'onset_to_thrombolysis': float,
        'precise_onset_known': int,
        'any_afib_diagnosis': bool
        # 'discharge_disability'
    })

    # Make a copy of this data that is ready for the model.
    # The same data except the Stroke Team column is one-hot-encoded.
    X = one_hot_encode_data(df, one_hot_column='stroke_team_id')
    # Add in ohe columns of the missing teams:
    teams_missing = [f'team_{t}' for t in list(set(all_teams.astype(int)) - set(teams_here.astype(int)))]
    X[teams_missing] = 0

    # Place columns in the order expected by the model:
    column_order = list(headers[1:]) + [f'team_{t}' for t in all_teams]
    X = X[column_order]
    return X


# @st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_pretrained_model(model_file='model.p'):
    # Load XGB Model
    filename = (dir + 'data_ml/' + model_file)
    with open(filename, 'rb') as filehandler:
        model = pickle.load(filehandler)
    return model


# @st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_explainer():
    # Load SHAP explainers
    filename = (dir + 'data_ml/shap_explainer.p')
    with open(filename, 'rb') as filehandler:
        explainer = pickle.load(filehandler)
    return explainer


# @st.cache_resource()  #hash_funcs={'builtins.dict': lambda _: None})
def load_explainer_probability(model_file='shap_explainer_probability.p'):
    filename = (dir + 'data_ml/' + model_file)
    with open(filename, 'rb') as filehandler:
        explainer_probability = pickle.load(filehandler)
    return explainer_probability


def load_outcomes_ml():
    filename = (dir + 'data_ml/' + 'outcome_model.p')
    with open(filename, 'rb') as filehandler:
        outcome_model = pickle.load(filehandler)
    return outcome_model


def find_highlighted_hb_teams(
        stroke_teams_list,
        inds_benchmark,
        highlighted_teams_input,
        default_highlighted_team,
        display_name_of_default_highlighted_team
        ):
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


def locate_benchmarks(
        benchmark_filename,
        benchmark_team_column,
        n_benchmark_teams=25
        ):
    # Find which teams are "benchmark teams" by using the imported
    # data:
    benchmark_df = utilities_ml.inputs.import_benchmark_data(
        benchmark_filename,
        benchmark_team_column
    )
    # Make list of benchmark rank:
    # Currently benchmark_df is sorted from highest to lowest
    # probability of thrombolysis, where the first 30 highest
    # are the benchmark teams.
    # X array is sorted alphabetically by stroke team,
    # so first sort the benchmark dataframe alphabetically to match
    # and then keep a copy of the resulting "Rank" column.
    # This list will be used in the sorted_results array:
    benchmark_rank_list = \
        benchmark_df.sort_values(benchmark_team_column)['Rank'].to_numpy()
    # Find indices of benchmark data at the moment
    # for making a combined benchmark-highlighted team list.
    inds_benchmark = np.where(benchmark_rank_list <= n_benchmark_teams)[0]
    return benchmark_df, benchmark_rank_list, inds_benchmark


def setup_for_app(
        container_input_highlighted_teams,
        container_input_patient_details,
        ):

    # ----- Load data from file -----
    # List of stroke teams that this patient will be sent to:
    stroke_teams_list = read_stroke_teams_from_file(
        stroke_teams_file
    )

    # ----- User inputs -----
    # ----- Highlighted teams -----
    # The user can select teams to highlight on various plots.
    # The teams are selected using either a streamlit input widget
    # in the following container, or by clicking directly on
    # certain plots.
    with container_input_highlighted_teams:
        # Pick teams to highlight on the bar chart:
        highlighted_teams_input = utilities_ml.container_inputs.\
            highlighted_teams(
                stroke_teams_list,
                default_highlighted_team,
                display_name_of_default_highlighted_team
                )
    # All patient detail widgets go in the sidebar:
    with container_input_patient_details:
        user_inputs_dict = utilities_ml.container_inputs.user_inputs()

    # ----- Build the X array -----
    # Build the patient details and stroke teams
    # into a 2D DataFrame:
    X, headers_X = build_X(
            user_inputs_dict,
            stroke_teams_list,
            stroke_team_col,
            model_version
            )
    # This array X is now ready to be run through the model.
    # After the model is run, we'll create an array sorted_results
    # that contains all of the useful information for plotting and
    # the metrics.
    # Before that, we'll create a few arrays that will be added to
    # sorted_results.

    # ----- Benchmark teams -----
    benchmark_df, benchmark_rank_list, inds_benchmark = (
        locate_benchmarks(
            benchmark_filename,
            benchmark_team_column,
            n_benchmark_teams
            ))

    # Columns for highlighted teams and highlighted+benchmark (HB),
    # and a shorter list hb_teams_input with just the unique values
    # from hb_teams_list in the order that the highlighted teams
    # were added to the highlighted input list.
    highlighted_teams_list, hb_teams_list, hb_teams_input = \
        find_highlighted_hb_teams(
            stroke_teams_list,
            inds_benchmark,
            highlighted_teams_input,
            default_highlighted_team,
            display_name_of_default_highlighted_team
            )

    # Find colour lists for plotting (saved to session state):
    remove_old_colours_for_highlights(hb_teams_input)
    choose_colours_for_highlights(hb_teams_input)

    # Prototype patients:
    df_proto = import_proto_patients()
    # Add in "this patient" from user selections:
    cols = ['onset_to_arrival_time', 'onset_during_sleep',
            'arrival_to_scan_time', 'infarction', 'stroke_severity',
            'onset_time_precise', 'prior_disability', 'anticoag', 'age']
    row = ['This patient', np.nan] + [user_inputs_dict[k] for k in cols]
    df_here = pd.DataFrame(pd.Series(row, index=df_proto.columns)).T
    df_proto = pd.concat((df_here, df_proto), axis='rows', ignore_index=True)
    # Create onset to thrombolysis time:
    df_proto['onset_to_thrombolysis_time'] = (
        df_proto['onset_to_arrival_time'] + df_proto['arrival_to_scan_time'])
    # Names of prototype patients:
    proto_names = df_proto['Patient prototype'].values
    # Gather all team names for highlighted and benchmark teams:
    highlighted_teams = [int(t) for t in highlighted_teams_input]
    benchmark_teams = benchmark_df.loc[
        benchmark_df['Rank'] < n_benchmark_teams, 'stroke_team_id'].values
    teams_proto = list(set(np.append(highlighted_teams, benchmark_teams)))

    # Make a copy of these patients for each benchmark
    # and highlighted team:
    dfs_proto = []
    for team in teams_proto:
        df_here = df_proto.copy()
        if team in highlighted_teams:
            hb_name = hb_teams_list[
                np.where(highlighted_teams_list == f'{team}')][0]
        else:
            hb_name = bench_str
        df_here['stroke_team'] = team
        df_here['hb_team'] = hb_name
        dfs_proto.append(df_here)
    # Stack these resulting data:
    df_proto = pd.concat(dfs_proto, axis='rows', ignore_index=True)
    # Prepare for predictions:
    X_proto = build_dataframe_proto(df_proto, stroke_teams_list)
    X_outcomes = build_dataframe_outcomes(df_proto, stroke_teams_list)

    return (
        stroke_teams_list,
        highlighted_teams_input,
        X,
        headers_X,
        benchmark_df,
        benchmark_rank_list,
        inds_benchmark,
        highlighted_teams_list,
        hb_teams_list,
        hb_teams_input,
        user_inputs_dict,
        df_proto,
        X_proto,
        proto_names,
        X_outcomes,
    )


def set_up_sidebar(path_to_details):
    st.markdown(
        '## Patient details',
        help=''.join([
            '🔍 - [Which patient details are included?]',
            f'({path_to_details}which-features-are-used)',
            '\n\n',
            '🔍 - [Why do we model only ten features?]',
            f'({path_to_details}why-these-features)'
            ])
        )
    # Put all of the user input widgets in here later:
    container_input_patient_details = st.container()

    # Add an option for removing plotly_events()
    # which doesn't play well on skinny screens / touch devices.

    # st.markdown('-'*50)
    # st.markdown('## Advanced options')
    # if st.checkbox('Disable interactive plots'):
    #     use_plotly_events = False
    # else:
    #     use_plotly_events = True
    # st.caption(''.join([
    #     'The clickable plots sometimes appear strange ',
    #     'on small screens and touch devices, ',
    #     'so select this option to convert them to normal plots.'
    # ]))
    return container_input_patient_details
    # return use_plotly_events, container_input_patient_details
