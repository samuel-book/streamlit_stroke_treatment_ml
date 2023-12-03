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
    n_benchmark_teams = 25
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

    # What are the inds to look up similar test patients?
    masks_severity = [
        (user_inputs_dict['stroke_severity'] < 8),
        ((user_inputs_dict['stroke_severity'] >= 8) & (user_inputs_dict['stroke_severity'] <= 32)),
        (user_inputs_dict['stroke_severity'] > 32)
        ]
    masks_mrs = [
        ((user_inputs_dict['prior_disability'] == 0) | (user_inputs_dict['prior_disability'] == 1)),
        ((user_inputs_dict['prior_disability'] == 2) | (user_inputs_dict['prior_disability'] == 3)),
        ((user_inputs_dict['prior_disability'] == 4) | (user_inputs_dict['prior_disability'] == 5)),
        ]
    masks_age = [
        (user_inputs_dict['age'] < 80),
        (user_inputs_dict['age'] >= 80)
        ]
    masks_infarction = [
        (user_inputs_dict['infarction'] == 0),
        (user_inputs_dict['infarction'] != 0)
        ]
    masks_onset_scan = [
        (user_inputs_dict['onset_to_arrival_time'] + user_inputs_dict['arrival_to_scan_time'] <= 4*60),
        (user_inputs_dict['onset_to_arrival_time'] + user_inputs_dict['arrival_to_scan_time'] > 4*60)
        ]
    masks_precise = [
        (user_inputs_dict['onset_time_precise'] == 0),
        (user_inputs_dict['onset_time_precise'] != 0)
        ]
    masks_sleep = [
        (user_inputs_dict['onset_during_sleep'] == 0),
        (user_inputs_dict['onset_during_sleep'] != 0)
        ]
    masks_anticoag = [
        (user_inputs_dict['anticoag'] == 0),
        (user_inputs_dict['anticoag'] != 0)
        ]

    masks = {
        'onset_scan':masks_onset_scan,
        'severity':masks_severity,
        'mrs':masks_mrs,
        'age':masks_age,
        'infarction':masks_infarction,
        'precise':masks_precise,
        'sleep':masks_sleep,
        'anticoag':masks_anticoag
    }

    # masks_names = list(masks.keys())
    # masks_lists = list(masks.values())

    inds = {}
    for key, val in zip(masks.keys(), masks.values()):
        for i, m in enumerate(val):
            if m == 1:
                inds[key] = i

    # Which mask number is this?
    df = pd.read_csv(f'{dir}data_ml/mask_numbers.csv')
    df_mask = df[
        (df['onset_scan_mask_number'] == inds['onset_scan']) &
        (df['severity_mask_number'] == inds['severity']) &
        (df['mrs_mask_number'] == inds['mrs']) &
        (df['age_mask_number'] == inds['age']) &
        (df['infarction_mask_number'] == inds['infarction']) &
        (df['precise_mask_number'] == inds['precise']) &
        (df['sleep_mask_number'] == inds['sleep']) &
        (df['anticoag_mask_number'] == inds['anticoag'])
    ]
    mask_number = df_mask['mask_number'].values[0]

    # Import all probabilities and thrombolysis:
    df_all_accuracy = pd.read_csv(f'{dir}data_ml/masks_probabilities.csv')
    all_probs = df_all_accuracy['predicted_probs']
    all_reals = df_all_accuracy['thrombolysis']

    # Mask for just this mask number:
    mask = (df_all_accuracy['mask_number'] == mask_number)
    test_probs = all_probs[mask]
    test_reals = all_reals[mask]

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
        all_probs,
        all_reals,
        test_probs,
        test_reals
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

    st.markdown('-'*50)
    st.markdown('## Advanced options')
    if st.checkbox('Disable interactive plots'):
        use_plotly_events = False
    else:
        use_plotly_events = True
    st.caption(''.join([
        'The clickable plots sometimes appear strange ',
        'on small screens and touch devices, ',
        'so select this option to convert them to normal plots.'
    ]))
    return use_plotly_events, container_input_patient_details
