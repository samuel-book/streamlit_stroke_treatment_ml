"""
Interactive demo, the main page of the SAMueL machine learning
model explanation app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this app, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

# For compatibility with combo app,
# add an extra bit to the path if we need to.
try:
    # Try importing something as though we're running this from the
    # same directory as the landing page, Introduction.py.
    from utilities_ml.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./streamlit_stroke_treatment_ml/')
    # Now the following import will work:
    from utilities_ml.fixed_params import page_setup

# Custom functions:
from utilities_ml.fixed_params import write_markdown_in_colour, bench_str
from utilities_ml.inputs import set_up_sidebar
import utilities_ml.inputs
import utilities_ml.main_calculations
from utilities_ml.container_uncertainty import \
    find_similar_test_patients, get_numbers_each_accuracy_band, \
    find_accuracy, write_confusion_matrix, fudge_100_test_patients

# Containers:
import utilities_ml.container_metrics
import utilities_ml.container_bars
import utilities_ml.container_proto
import utilities_ml.container_outcomes


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Set up the link to the "Details" page:
    import sys
    if './streamlit_stroke_treatment_ml/' in sys.path:
        path_to_details = './🔍_Details:_Thrombolysis_decisions#'
    else:
        path_to_details = './Details#'

    # Background info
    # Title:
    st.title(':crystal_ball: Thrombolysis decisions')

    st.markdown(
        '''
        The SAMueL-2 model finds the probability of
        any stroke team thrombolysing any patient.
        '''
        )
    # Draw the image with the basic model summary.
    try:
        st.image('./utilities_ml/SAMueL2_model_wide.png')
    except (FileNotFoundError,
            st.runtime.media_file_storage.MediaFileStorageError):
        # Add an extra bit to the path for the combo app.
        st.image('./streamlit_stroke_treatment_ml/' +
                 'utilities_ml/SAMueL2_model_wide.png')
    st.markdown(
        '''
        We can use the same patient details
        for all stroke teams
        to compare their decision-making.
        ''',
        help=''.join([
            '🔍 - [Which stroke teams are included?]',
            f'({path_to_details}which-teams-are-included)',
            '\n\n',
            '🔍 - [What do the probabilities mean?]',
            f'({path_to_details}what-do-the-probabilities-mean)'
        ])
        )

    # User inputs
    with st.sidebar:
        # use_plotly_events,
        container_input_patient_details = (
            set_up_sidebar(path_to_details))
        with st.expander('Advanced options'):
            allow_maybe = st.checkbox(
                'Allow teams to decide to "maybe" thrombolyse.'
            )
            if allow_maybe:
                prob_maybe_min = st.number_input(
                    '"Maybe" minimum (%)',
                    min_value=0.0,
                    max_value=50.0,
                    value=33.3,
                    step=0.1,
                    format="%0.1f",
                    ) / 100.0
                prob_maybe_max = st.number_input(
                    '"Maybe" maximum (%)',
                    min_value=50.0,
                    max_value=100.0,
                    value=66.6,
                    step=0.1,
                    format="%0.1f",
                    ) / 100.0
                # Don't allow 50% exactly in case it messes with
                # inequalities later.
                if prob_maybe_max == 0.5:
                    prob_maybe_max += 1e-7
                if prob_maybe_min == 0.5:
                    prob_maybe_min -= 1e-7
            else:
                # Placeholder values.
                prob_maybe_min = 0.1
                prob_maybe_max = 0.9

    # DATA USED
    st.markdown('')  # Breathing room
    st.subheader('Data used to make the model')
    cols_data_etc = st.columns([1, 2])
    with cols_data_etc[0]:
        st.markdown(
            '''
            There are two data sets:
            ''',
            help=''.join([
                '🔍 - [What data is used?]',
                f'({path_to_details}what-data-is-used)',
            ])
        )
        st.markdown(
            '''            
            __🔮 Training__ (110,000 patients)  
            __🔮 Testing__ (10,000 patients)

            The patients cover England and Wales from 2016 to 2021
            and meet the conditions in this box.
            '''
        )

    with cols_data_etc[1]:
        container_data = st.container(border=True)
    with container_data:
        cols_data = st.columns(2)
    with cols_data[0]:
        st.markdown(
            '''
            ✨ Impossible data cleaned.  
            🚑 Arrived by ambulance.  
            🪚 Data limited to 10 details.  
            ⏰ Onset time known.  
            🩻 Onset to scan under 4 hours.  
            '''
            )
    with cols_data[1]:
        st.markdown(
            '''
            👥 Admission team had over 250 admissions.  
            💉 Admission team thrombolysed at least 10 patients.  
            '''
            )
    # ⏳🩻 Alternative emoji for onset to scan

    st.markdown('')  # Breathing room
    st.subheader('How we categorise the results')
    cols_method = st.columns([6, 4])
    with cols_method[0]:
        if allow_maybe:
            method_str = (
                '''
                | Probability | Decision |
                | --- | --- |
                | At least ''' + f'{100.0*prob_maybe_max:.1f}' + '''% | ✔️ would thrombolyse |
                | From ''' + f'{100.0*prob_maybe_min:.1f}' + '''% to ''' + f'{100.0*prob_maybe_max:.1f}' + '''% | ❓ might thrombolyse |
                | Below ''' + f'{100.0*prob_maybe_min:.1f}' + '''% | ❌ would not thrombolyse |
                '''
            )
        else:
            method_str = (
                '''
                | Probability | Decision |
                | --- | --- |
                | At least 50% | ✔️ would thrombolyse |
                | Below 50% | ❌ would not thrombolyse |
                '''
            )
        st.markdown(method_str)

    with cols_method[1]:
        st.markdown(
            '''
            :red[__Benchmark teams__] are more likely than average to choose
            thrombolysis.
            ''',
            help=''.join([
                '🔍 - [What are benchmark teams?]',
                f'({path_to_details}what-are-benchmark-teams)',
                '\n\n',
                '🔍 - [How are they picked?]',
                f'({path_to_details}how-are-the-benchmark-teams-picked)'
                ])
            )
        st.markdown(
            '''
            The :red[__benchmark decision__] is the
            option picked by most of the benchmark teams.
            '''
            )

    st.markdown('#')  # Breathing room
    st.header(':abacus: Results for this patient', divider='blue')
    st.markdown(''':blue[The patient details can be viewed and
                changed in the left sidebar.]''')

    # Draw some empty containers on the page.
    # They'll appear in this order, but we'll fill them in another order.
    st.markdown('')  # Breathing room
    st.subheader('How many teams would choose thrombolysis?')
    container_metrics = st.container(border=True)

    st.markdown('')  # Breathing room
    container_highlighted_summary = st.container()
    with container_highlighted_summary:
        st.subheader('What would your team do?')
        st.caption(
            '''
            To highlight stroke teams, select them in this box
            or click on them in the interactive bar chart.
            ''',
            help=''.join([
                '🔍 - [Why are the team names numbers?]'
                f'({path_to_details}why-are-teams-given-numbers)'
                '\n\n',
                '🔍 - [What is my team called?]'
                f'({path_to_details}what-is-my-team-called)'
                '\n\n',
                '🔍 - [How is the data anonymised?]'
                f'({path_to_details}how-is-the-data-anonymised)'
            ])
            )
        # Set up columns to put the highlighted teams' info in.
        # One of these columns will also contain the highlighted
        # teams input.
        cols_highlighted_summary = st.columns(4)
        with cols_highlighted_summary[0]:
            container_input_highlighted_teams = st.container()

    st.markdown(' ')  # Breathing room
    st.markdown('')  # Breathing room
    container_bar_chart = st.container()
    with container_bar_chart:
        st.subheader('How likely is thrombolysis for each team?')
        st.caption('To see the team names, hover or click on a bar.')

    st.markdown('#')  # Breathing room
    st.header(':dolls: Results for patient prototypes', divider='green')
    st.markdown(
        '''
        Patient prototypes are variations on the ideal candidate
        for thrombolysis. The ideal patient:
        '''
    )
    cols_ideal = st.columns(3)
    with cols_ideal[0]:
        st.markdown('''
        + arrives at the hospital quickly
        + has a moderately severe stroke
        '''
        )
    with cols_ideal[1]:
        st.markdown('''
        + has no prior disability
        + has a precisely-known onset time
        '''
        )
    with cols_ideal[2]:
        st.markdown('''
        + is aged under 80
        '''
        )
    container_proto = st.container()
    with container_proto:
        st.subheader('How likely is thrombolysis for patient prototypes?')
        st.markdown(
            '''
            We find the probability of each team giving thrombolysis
            to each of the patient prototypes.
            '''
            )

    container_outcomes = st.container()
    with container_outcomes:
        st.subheader('What would the patient outcomes be?')
        st.markdown(
            '''
            Most patients in the real data have data on their
            discharge disability. This is a modified
            Rankin scale (mRS) score between 0 (no disability) and 6
            (dead).
            '''
        )
        cols_outcomes = st.columns([0.35, 0.65])
        with cols_outcomes[1]:
            st.markdown(
                '''
                We train a second model to predict discharge disability.
                This model uses a subset of the patient data from the
                thrombolysis prediction model. The subset contains only
                patients who did not receive thrombectomy
                and whose discharge disability levels are known.

                We assume that the time to thrombolysis is
                the time from stroke onset to scan plus 15 minutes.
                '''
            )
            # # The results show the expected distribution of discharge
            # # disability levels for 100 similar patients who receive
            # # no treatment (left graph) or thrombolysis only (right graph).
            # '''
            # )
        with cols_outcomes[0]:
            st.markdown(
                '''
                | Disability | mRS scores |
                | --- | --- |
                | Independent | 0, 1, 2 |
                | Dependent | 3, 4, 5 |
                | Dead | 6 |
                '''
            )


    # ###########################
    # ########## SETUP ##########
    # ###########################
    from utilities_ml.inputs import setup_for_app
    from utilities_ml.fixed_params import n_benchmark_teams, \
        default_highlighted_team, display_name_of_default_highlighted_team
    (
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
        proto_display_names,
        X_outcomes,
    ) = setup_for_app(
        container_input_highlighted_teams,
        container_input_patient_details,
        )

    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################

    # Load in the model:
    from utilities_ml.fixed_params import ml_model_file
    model = utilities_ml.inputs.load_pretrained_model(ml_model_file)
    outcome_model = utilities_ml.inputs.load_outcomes_ml()
    # Main useful array:
    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                          highlighted_teams_list, benchmark_rank_list,
                          hb_teams_list, allow_maybe, prob_maybe_min,
                          prob_maybe_max)
    # Prototype patients:
    proto_results = utilities_ml.main_calculations.\
        predict_treatment_proto(df_proto, X_proto, model, allow_maybe,
                                prob_maybe_min, prob_maybe_max)
    # Outcomes:
    outcome_results = utilities_ml.main_calculations.\
        predict_outcomes_proto(df_proto, X_outcomes, outcome_model)


    # ###########################
    # ######### RESULTS #########
    # ###########################

    with container_metrics:
        # Print metrics for how many teams would thrombolyse:
        utilities_ml.container_metrics.main(
            sorted_results, n_benchmark_teams, allow_maybe)

    with container_highlighted_summary:
        highlighted_teams_colours = \
            st.session_state['highlighted_teams_colours']
        # Print results for highlighted teams.
        cols = cols_highlighted_summary
        i = 1
        for t, team in enumerate(hb_teams_input):
            if 'enchmark' not in team:
                try:
                    col = cols[i]
                except IndexError:
                    # Start a new row:
                    i = 1
                    col = cols[i]
                i += 1

                df_here = sorted_results[sorted_results['HB team'] == team]
                colour_here = highlighted_teams_colours[team]
                if team == default_highlighted_team:
                    team = display_name_of_default_highlighted_team
                # else:
                    # team = 'Team ' + team
                with col:
                    con = st.container(border=True)
                    with con:
                        write_markdown_in_colour(
                            '<strong> Team ' + team + '</strong>',
                            colour=colour_here)
                        prob_here = df_here['Probability_perc'].values[0]
                        thromb_here = df_here['Thrombolyse_str'].values[0]
                        if 'Yes' in thromb_here:
                            emoji_here = '✔️ '
                            extra_str = 'would '
                        elif 'No' in thromb_here:
                            emoji_here = '❌ '
                            extra_str = 'would not '
                        else:
                            emoji_here = '❓ '
                            extra_str = 'might '
                        st.markdown(
                            f'''
                            {prob_here:.2f}% chance

                            {emoji_here}{extra_str}thrombolyse
                            '''
                        )

    with container_bar_chart:
        # Top interactive bar chart:
        utilities_ml.container_bars.main(
            sorted_results,
            hb_teams_input,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            # use_plotly_events,
            allow_maybe,
            prob_maybe_min,
            prob_maybe_max
            )

    with container_proto:
        # Prototype patients bar chart.
        # For each prototype, average the benchmark decisions:
        # teams_bench_only = proto_results.loc[proto_results['HB team'].str.contains('enchmark'), 'Stroke team'].unique()
        # Pick out team names that contain a star:
        mask_bench = proto_results['HB team'].str.contains('\U00002605')
        teams_bench = proto_results.loc[mask_bench, 'Stroke team'].unique()
        rows_bench = []
        for proto_patient in proto_names:
            df_here = proto_results.loc[(
                (proto_results['Patient prototype'] == proto_patient) &
                (proto_results['Stroke team'].isin(teams_bench))
            )]
            prob_here = df_here['Probability'].mean()
            prob_perc_here = df_here['Probability_perc'].mean()
            thromb_here = df_here['Thrombolyse'].mode().values
            thromb_str_here = df_here['Thrombolyse_str'].mode().values
            # If there are multiple modes, arbitrarily pick the biggest
            # (= most likely to treat).
            if len(thromb_here) > 1:
                ind = np.where(thromb_here == max(thromb_here))[0]
                thromb_here = thromb_here[ind]
                thromb_str_here = thromb_str_here[ind]
            else:
                thromb_here = thromb_here[0]
                thromb_str_here = thromb_str_here[0]
            rows_bench.append([
                proto_patient, df_here['proto_display'].values[0],
                'Benchmark average', 'Benchmark average',
                prob_here, prob_perc_here, thromb_here, thromb_str_here,
            ])
        df_bench = pd.DataFrame(rows_bench, columns=proto_results.columns)
        # Now add in the highlighted teams:
        teams_highlighted = proto_results.loc[~proto_results['HB team'].str.contains('enchmark'), 'Stroke team'].unique()
        mask_highlighted = proto_results['Stroke team'].isin(teams_highlighted)
        df_proto_results = pd.concat((df_bench,
                                      proto_results.loc[mask_highlighted]))
        utilities_ml.container_proto.main(
            df_proto_results,
            proto_display_names,
            ['Benchmark average'] + hb_teams_input,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            # use_plotly_events,
            allow_maybe,
            prob_maybe_min,
            prob_maybe_max
            )

    with container_outcomes:
        # Create benchmark average results:
        cols = [c for c in outcome_results if (('treated' in c) | (c == 'improvement'))]
        rows_bench = []
        for proto_patient in proto_names:
            df_here = outcome_results.loc[(
                (outcome_results['Patient prototype'] == proto_patient) &
                (outcome_results['Stroke team'].isin(teams_bench))
            )]
            vals = [df_here[c].mean() for c in cols]
            rows_bench.append([
                proto_patient, df_here['proto_display'].values[0],
                'Benchmark average', 'Benchmark average',
                *vals
            ])
        df_bench = pd.DataFrame(rows_bench, columns=outcome_results.columns)
        # Now add in the highlighted teams:
        mask_highlighted = outcome_results['Stroke team'].isin(teams_highlighted)
        df_outcome_results = pd.concat(
            (df_bench, outcome_results.loc[mask_highlighted]))
        # Plot bars:
        with cols_outcomes[1]:
            proto_display_name = st.selectbox(
                'Prototype patient for outcome bar chart',
                proto_display_names
                )
        proto_name = df_outcome_results.loc[
            df_outcome_results['proto_display'] == proto_display_name,
            'Patient prototype'
            ].values[0]

        cols_plots = st.columns(2)
        teams_to_plot = ['Benchmark average'] + hb_teams_input
        # Remove benchmark entries:
        teams_to_plot = [t for t in teams_to_plot if
                         (('ench' not in t) | ('average' in t))]
        for t, team in enumerate(teams_to_plot):
            with cols_plots[t % 2]:
                utilities_ml.container_outcomes.main2(
                    df_outcome_results,
                    team,
                    proto_name,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team,
                    # use_plotly_events,
                    )


if __name__ == '__main__':
    main()
