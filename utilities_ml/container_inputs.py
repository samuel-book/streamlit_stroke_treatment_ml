"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st

# from utilities_ml.fixed_params import \
#     display_name_of_default_highlighted_team, default_highlighted_team

def user_inputs(choose_infarction=False):
    st.markdown('### ')  # Breathing room
    st.markdown('### Clinical data')
    # Prior disability level
    # Integer input, 0 to 5 inclusive.
    prior_disability_input = st.slider(
        'Prior disability (mRS score)',
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help='Ranges from 0 (highest utility) to 5 (lowest utility).',
        key='prior_disability_input'
    )

    # Stroke severity
    # Integer input from 0 to 42.
    stroke_severity_input = st.slider(
        'Stroke severity',
        min_value=0,
        max_value=42,
        value=15,
        step=1,
        help='Ranges from 0 (least severe) to 42 (most severe).',
        key='stroke_severity_input'
    )

    # Age
    # Float input in steps of 5 years from 2.5 years.
    age_input = st.number_input(
        'Age',
        min_value=37.5,
        max_value=97.5,
        value=72.5,
        step=5.0,
        help='Ranges from 37.5 to 97.5.',
        key='age_input'
    )
    # It's possible to override this by typing your own entry.
    # Instead manually convert to nearest step of 5 and say so.
    # if age_input % 2.5 != 0.0:
    # age_input = round(age_input, 2)

    # Infarction
    if choose_infarction:
        # String input for user friendliness.
        # We convert to integer for use with the model.
        infarction_input_str = st.radio(
            'Infarction',
            options=['Yes', 'No'],
            index=0,
            horizontal=True,
            key='infarction_input_str',
            help='Whether the stroke is caused by a blood clot.'
        )
        infarction_input = 1 if infarction_input_str == 'Yes' else 0
    else:
        infarction_input = 1
        infarction_input_str = 'Yes'

    # Use of AF anticoagulants
    # String input for user friendliness.
    # We convert to integer for use with the model.
    anticoag_input_str = st.radio(
        'AF Anticoagulants',
        options=['Yes', 'No'],
        index=1,
        horizontal=True,
        key='anticoag_input_str',
        help='Whether the patient takes blood-thinning medication.'
    )
    anticoag_input = 1 if anticoag_input_str == 'Yes' else 0

    st.markdown('### ')  # Breathing room
    st.markdown('### Pathway data')
    # Onset-to-arrival time
    # Number (integer?) input in minutes.
    onset_to_arrival_time_input = st.number_input(
        'Onset-to-arrival time (minutes)',
        min_value=0,
        max_value=300,
        value=90,
        step=15,
        help='Ranges from 0 to 300.',
        key='onset_to_arrival_time_input'
    )

    # Arrival-to-scan time
    # Number (integer?) input in minutes
    arrival_to_scan_time_input = st.number_input(
        'Arrival-to-scan time (minutes)',
        min_value=0,
        max_value=90,
        value=15,
        step=15,
        help='Ranges from 0 to 90.',
        key='arrival_to_scan_time_input'
    )

    # Precise onset time
    # String input for user friendliness.
    # We convert to integer for use with the model.
    onset_time_precise_input_str = st.radio(
        'Onset time',
        options=['Precise', 'Estimated'],
        index=0,
        horizontal=True,
        key='onset_time_precise_input_str',
        help='Whether the onset time is known precisely.'
    )
    onset_time_precise_input = (1 
        if onset_time_precise_input_str == 'Precise' else 0)

    # Onset during sleep
    # String input for user friendliness.
    # We convert to integer for use with the model.
    onset_during_sleep_input_str = st.radio(
        'Onset during sleep',
        options=['Yes', 'No'],
        index=1,
        horizontal=True,
        key='onset_during_sleep_input_str',
        help='Whether the stroke began during sleep.'
    )
    onset_during_sleep_input = 1 \
        if onset_during_sleep_input_str == 'Yes' else 0

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


def highlighted_teams(
        stroke_teams_list,
        default_highlighted_team,
        display_name_of_default_highlighted_team
        ):
    try:
        # If we've already selected highlighted teams using the
        # clickable plotly graphs, then load that list:
        existing_teams = st.session_state['highlighted_teams_with_click']
    except KeyError:
        # Make a dummy list so streamlit behaves as normal:
        existing_teams = [display_name_of_default_highlighted_team]

    # Swap out the default highlighted team's name for
    # the label chosen in fixed_params.
    teams_input_list = stroke_teams_list.copy().tolist()
    # Remove the chosen default team...
    teams_input_list.remove(default_highlighted_team)
    # ... and add the new name to the start of the list.
    teams_input_list = [display_name_of_default_highlighted_team] + \
                       teams_input_list

    # Remove any existing teams that are not in this input list.
    existing_teams = [team for team in existing_teams
                      if team in teams_input_list]
    st.session_state['highlighted_teams_with_click'] = existing_teams

    highlighted_teams_input = st.multiselect(
        'Stroke teams to highlight:',
        teams_input_list,
        # help='Pick up to 9 before the colours repeat.',
        key='highlighted_teams',
        default=existing_teams
    )
    return highlighted_teams_input
