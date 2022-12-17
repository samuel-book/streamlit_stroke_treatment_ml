"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st


def user_inputs():
    # Arrival-to-scan time
    # Number (integer?) input in minutes
    arrival_to_scan_time_input = st.number_input(
        'Arrival-to-scan time (minutes)',
        min_value=0,
        max_value=600,
        value=20,
        step=5,
        help='Ranges from 0 to 600.',
        key='arrival_to_scan_time_input'
    )

    # Infarction
    # String input for user friendliness.
    # We convert to integer for use with the model.
    infarction_input_str = st.radio(
        'Infarction',
        options=['Yes', 'No'],
        index=0,
        horizontal=True,
        key='infarction_input_str'
    )
    infarction_input = 1 if infarction_input_str == 'Yes' else 0

    # Stroke severity
    # Integer input from 0 to 42.
    stroke_severity_input = st.number_input(
        'Stroke severity',
        min_value=0,
        max_value=42,
        value=15,
        step=1,
        help='Ranges from 0 (least severe) to 42 (most severe).',
        key='stroke_severity_input'
    )

    # Precise onset time
    # String input for user friendliness.
    # We convert to integer for use with the model.
    onset_time_precise_input_str = st.radio(
        'Precise onset time',
        options=['Yes', 'No'],
        index=0,
        horizontal=True,
        key='onset_time_precise_input_str'
    )
    onset_time_precise_input = 1 \
        if onset_time_precise_input_str == 'Yes' else 0

    # Prior disability level
    # Integer input, 0 to 5 inclusive.
    prior_disability_input = st.number_input(
        'Prior disability (mRS score)',
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help='Ranges from 0 to 5.',
        key='prior_disability_input'
    )

    # Use of AF anticoagulents
    # String input for user friendliness.
    # We convert to integer for use with the model.
    anticoag_input_str = st.radio(
        'AF Anticoagulents',
        options=['Yes', 'No'],
        index=1,
        horizontal=True,
        key='anticoag_input_str'
    )
    anticoag_input = 1 if anticoag_input_str == 'Yes' else 0

    # Onset-to-arrival time
    # Number (integer?) input in minutes.
    onset_to_arrival_time_input = st.number_input(
        'Onset-to-arrival time (minutes)',
        min_value=0,
        max_value=600,
        value=80,
        step=5,
        help='Ranges from 0 to 600.',
        key='onset_to_arrival_time_input'
    )

    # Onset during sleep
    # String input for user friendliness.
    # We convert to integer for use with the model.
    onset_during_sleep_input_str = st.radio(
        'Onset during sleep',
        options=['Yes', 'No'],
        index=1,
        horizontal=True,
        key='onset_during_sleep_input_str'
    )
    onset_during_sleep_input = 1 \
        if onset_during_sleep_input_str == 'Yes' else 0

    # Age
    # Float input in steps of 5 years from 2.5 years.
    age_input = st.number_input(
        'Age',
        min_value=2.5,
        max_value=127.5,
        value=72.5,
        step=5.0,
        help='Ranges from 0 to 600.',
        key='age_input'
    )
    # It's possible to override this by typing your own entry.
    # Instead manually convert to nearest step of 5 and say so.
    # if age_input % 2.5 != 0.0:
    age_input = round(age_input, 2)


    # Stick all of these inputs into a dictionary:
    user_input_dict = dict(
        arrival_to_scan_time=arrival_to_scan_time_input,
        infarction_str=infarction_input_str,
        infarction=infarction_input,
        stroke_severity=stroke_severity_input,
        onset_time_precise_str=onset_time_precise_input_str,
        onset_time_precise=onset_time_precise_input,
        prior_disability=prior_disability_input,
        anticoag_str=anticoag_input_str,
        anticoag=anticoag_input,
        onset_to_arrival_time=onset_to_arrival_time_input,
        onset_during_sleep_str=onset_during_sleep_input_str,
        onset_during_sleep=onset_during_sleep_input,
        age=age_input
    )
    return user_input_dict


def favourite_teams(stroke_teams_list):
    favourite_teams_input = st.multiselect(
        'Pick some stroke teams to highlight',
        stroke_teams_list,
        help='Pick up to 9 before the colours repeat.'
    )
    return favourite_teams_input
