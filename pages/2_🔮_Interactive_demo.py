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
from utilities_ml.inputs import set_up_sidebar
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
    st.title(':crystal_ball: :rainbow[Thrombolysis decisions]')

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
        st.markdown(
            '''
            | Probability | Decision |
            | --- | --- |
            | At least 66.6% | ✔️ would thrombolyse |
            | From 33.3% to 66.6% | ❓ might thrombolyse |
            | Below 33.3% | ❌ would not thrombolyse |
            '''
            )

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

    # User inputs
    with st.sidebar:
        use_plotly_events, container_input_patient_details = (
            set_up_sidebar(path_to_details))

    st.markdown('#')  # Breathing room
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
        user_inputs_dict
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

    # Import training data group sizes.
    # UPDATE THIS PATH LATER FOR COMBO APP ---------------------------------------------
    import pandas as pd
    df_training_groups = pd.read_csv(f'./data_ml/train_group_sizes.csv')


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
            use_plotly_events,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            )


    # ############################
    # ######### ACCURACY #########
    # ############################

    # Model accuracy
    import pandas as pd
    import numpy as np

    # Test data accuracy.
    from utilities_ml.container_uncertainty import find_similar_test_patients, get_numbers_each_accuracy_band, find_accuracy, write_confusion_matrix, fudge_100_test_patients

    st.markdown('#')  # Breathing room
    st.markdown('#')  # Breathing room
    st.header('❓ Accuracy', divider='red')
    st.markdown(
        '''
        We can measure the accuracy of the model using the real-life
        🔮 __Testing data__.  
        We check whether the real-life treatment decision for each
        patient matches the model decision.
        '''
        )
    st.markdown(
        '''
        The real decision may be either
        thrombolysis or not.
        There is no "❓ might thrombolyse" option.

        | Probability | Decision |
        | --- | --- |
        | At least 66.6% | ✔️ would thrombolyse |
        | From 50.0% to 66.6% | ❓✔️ would thrombolyse |
        | From 33.3% to 50.0% | ❓❌ would not thrombolyse |
        | Below 33.3% | ❌ would not thrombolyse |
        '''
        )
    st.markdown(' ')  # Breathing room
    st.subheader('How often do the predictions match reality?')
    # Predicted probabilities and the true thrombolysis yes/no results
    # for "test data" patients. Two lists - one contains all test
    # patients, the other only patients who are similar to the selected
    # patient details.
    # TO DO - detail what "similar" means.
    all_probs, all_reals, similar_probs, similar_reals, mask_number = (
        find_similar_test_patients(user_inputs_dict))

    # How many patients like this were in the training data?
    n_train_all_patients = df_training_groups['number_of_patients'].sum()
    n_train_similar_patients = df_training_groups[df_training_groups['mask_number'] == mask_number]['number_of_patients'].values[0]

    # All test patients:
    # Calculations:
    all_pr_dict = get_numbers_each_accuracy_band(all_probs, all_reals)
    all_n_total = len(all_probs)

    if all_n_total > 0:
        all_pr_dict_100 = fudge_100_test_patients(all_pr_dict)
        all_acc = find_accuracy(all_pr_dict)
    else:
        all_acc = np.NaN

    # Similar test patients:
    # Calculations:
    similar_pr_dict = get_numbers_each_accuracy_band(
        similar_probs, similar_reals)
    similar_n_total = len(similar_probs)

    if similar_n_total > 0:
        similar_pr_dict_100 = fudge_100_test_patients(similar_pr_dict)
        similar_acc = find_accuracy(similar_pr_dict)
    else:
        similar_acc = np.NaN

    st.markdown(
        f'''
        The model's accuracy is __{all_acc:.1f}%__ for all patients
        and __{similar_acc:.1f}%__ for patients similar to the given details.

        The number of patients in the __🔮 Training data__ is how many
        examples the model had to learn from.
        The number of patients in the __🔮 Testing data__ is how many
        patients were used to calculate the accuracy rate.

        | | All patients | Similar to this patient |
        | --- | --- | --- |
        | 🔮 Training data | {n_train_all_patients:,} | {n_train_similar_patients:,} | 
        | 🔮 Testing data | {all_n_total:,} | {similar_n_total:,} |
        '''
    )
    st.markdown(' ')  # Breathing room

    # Confusion matrix.
    st.subheader('How similar are the predictions to reality?')
    st.markdown(
        '''
        We can show all of the combinations of predicted and
        real-life thrombolysis decisions using the confusion matrix below.
        '''
        )

    tabs_matrix = st.tabs(['Scaled to 100 patients', 'True numbers of patients'])
    with tabs_matrix[0]:
        cols_100 = st.columns(2, gap='large')
        with cols_100[0]:
            # All test patients, scaled to 100:
            st.markdown('All test patients (out of 100)')
            write_confusion_matrix(all_pr_dict_100)
        with cols_100[1]:
            if similar_n_total > 0:
                # Similar test patients, scaled to 100:
                st.markdown('Similar test patients (out of 100)')
                write_confusion_matrix(similar_pr_dict_100)
    with tabs_matrix[1]:
        cols_all = st.columns(2, gap='large')
        with cols_all[0]:
            # All test patients:
            st.markdown(f'All test patients (out of {all_n_total})')
            write_confusion_matrix(all_pr_dict)
        with cols_all[1]:
            if similar_n_total > 0:
                # Similar test patients:
                st.markdown(f'Similar test patients (out of {similar_n_total})')
                write_confusion_matrix(similar_pr_dict)


    # ----- The end! -----


if __name__ == '__main__':
    main()
