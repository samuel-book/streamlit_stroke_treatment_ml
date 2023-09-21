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
from utilities_ml.fixed_params import write_markdown_in_colour
import utilities_ml.inputs
import utilities_ml.main_calculations
# Containers:
import utilities_ml.container_inputs
import utilities_ml.container_metrics
import utilities_ml.container_bars
import utilities_ml.container_waterfalls
import utilities_ml.container_combo_waterfall
import utilities_ml.container_results


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
    st.markdown(
        '''
        # :crystal_ball: Thrombolysis decisions

        The SAMueL-2 model gives the probability of
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
        We can use the model to give the same patient details
        to all stroke teams and decide how likely each team is
        to thrombolyse that patient.
        ''',
        help=''.join([
            '🔍 - [Which stroke teams are included?]',
            f'({path_to_details}which-teams-are-included)'
            ])
        )
    st.markdown(
        '''
        Stroke teams with a probability below 50% are unlikely to
        thrombolyse the patient, and other teams are
        likely to thrombolyse.
        ''',
        help=''.join([
            '🔍 - [What do the probabilities mean?]',
            f'({path_to_details}what-do-the-probabilities-mean)'
        ])
        )

    st.markdown('### How we categorise the results')
    cols_method = st.columns(3)
    with cols_method[0]:
        st.info(
            '''
            __Thrombolysis: yes or no?__

            If probability is at least 50%:

            ✔️ would thrombolyse

            If probability is below 50%:

            ❌ would not thrombolyse
            '''
            )
    with cols_method[1]:
        st.markdown('')  # To match offset of info/error boxes
        st.markdown('__Benchmark teams__')
        st.markdown(
            '''
            These teams are consistently more likely than average to
            thrombolyse any given patient.
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
            They are used to compare the treatment decisions
            of stroke teams with highly-thrombolysing units.
            '''
            )
    with cols_method[2]:
        st.error(
            '''
            __Benchmark decision__

            If at least half of the benchmark teams would thrombolyse:

            ✔️ would thrombolyse

            Otherwise:

            ❌ would not thrombolyse
            '''
            )

    # User inputs
    with st.sidebar:
        st.markdown(
            '## Patient details',
            help=''.join([
                '🔍 - [Which patient details are included?]',
                f'({path_to_details}which-features-are-used)',
                '\n\n',
                '🔍 - [Why do we model only ten features?]',
                f'({path_to_details}why-these-features)\n'
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

    st.markdown('-' * 50)
    st.markdown('## :abacus: Predictions for this patient')
    st.info(
        'The patient details can be viewed and changed in the left sidebar.',
        icon='ℹ️'
        )

    # Draw some empty containers on the page.
    # They'll appear in this order, but we'll fill them in another order.
    container_metrics = st.container()
    with container_metrics:
        st.markdown('### How many teams would thrombolyse this patient?')

    container_highlighted_summary = st.container()
    with container_highlighted_summary:
        st.markdown('## ')  # Breathing room
        st.markdown('### What would your team do?')
        st.caption(
            '''
            To highlight stroke teams, select them in this box
            or click on them in the interactive charts.
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

    container_bar_chart = st.container()
    with container_bar_chart:
        st.markdown('### What would each of the teams do?')
        st.caption('To see the team names, hover or click on a bar.')

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
        hb_teams_input
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
    # Main useful array:
    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                          highlighted_teams_list, benchmark_rank_list,
                          hb_teams_list)

    # ###########################
    # ######### RESULTS #########
    # ###########################

    with container_metrics:
        # Print metrics for how many teams would thrombolyse:
        utilities_ml.container_metrics.main(sorted_results, n_benchmark_teams)

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
                    write_markdown_in_colour(
                        '<strong> Team ' + team + '</strong>',
                        colour=colour_here)
                    prob_here = df_here['Probability_perc'].values[0]
                    thromb_here = df_here['Thrombolyse_str'].values[0]
                    if thromb_here == 'Yes':
                        emoji_here = '✔️ '
                        extra_str = ''
                    else:
                        emoji_here = '❌ '
                        extra_str = 'do not '
                    st.markdown(f'Probability: {prob_here:.2f}%')
                    st.markdown(emoji_here + extra_str + 'thrombolyse')
                    st.markdown('-' * 10)
                    # HTML horizontal rule is <hr> but appears in grey.

    with container_bar_chart:
        # Top interactive bar chart:
        utilities_ml.container_bars.main(
            sorted_results,
            hb_teams_input,
            use_plotly_events,
            default_highlighted_team,
            display_name_of_default_highlighted_team
            )

    # ----- The end! -----


if __name__ == '__main__':
    main()
