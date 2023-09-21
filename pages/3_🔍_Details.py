import streamlit as st
# import numpy as np

import utilities_ml.main_calculations
import utilities_ml.inputs
import utilities_ml.container_combo_waterfall
import utilities_ml.container_waterfalls

from utilities_ml.fixed_params import draw_sneaky_bar, model_version
from utilities_ml.inputs import write_text_from_file, \
    read_text_from_file, set_up_sidebar

# For compatibility with combo app,
# add an extra bit to the path if we need to.
try:
    # Try importing something as though we're running this from the
    # same directory as the landing page, Introduction.py.
    from utilities_ml.fixed_params import page_setup
    p = 'pages/text_for_pages/'

except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./streamlit_stroke_treatment_ml/')
    # Now the following import will work:
    from utilities_ml.fixed_params import page_setup


def main():
    page_setup()

    import sys
    if './streamlit_stroke_treatment_ml/' in sys.path:
        p = 'streamlit_stroke_treatment_ml/pages/text_for_pages/'
    else:
        p = 'pages/text_for_pages/'

    if './streamlit_stroke_treatment_ml/' in sys.path:
        path_to_details = './🔍_Details:_Thrombolysis_decisions#'
    else:
        path_to_details = './Details#'

    st.warning('Work in progress', icon='⚠️')

    st.markdown('# 🔍 Details')  # # 🧐 Details

    # Create a bunch of containers now
    # and later fill them with text and images.
    # To change the order displayed on the streamlit app page,
    # only need to change the order here - not elsewhere in this document.
    container_summary = st.container()
    # What the app shows:
    container_what_the_app_shows = st.container()
    container_stroke_teams = st.container()
    container_patient_data = st.container()
    container_prediction_model = st.container()
    container_shap_general = st.container()
    container_waterfall = st.container()
    container_wizard_hat = st.container()
    container_violin = st.container()
    container_shap_values = st.container()
    # How the app works:
    container_how_the_app_works = st.container()
    container_ssnap = st.container()
    container_create_prediction_model = st.container()
    container_create_shap_model = st.container()
    container_create_shap_values = st.container()
    container_benchmark = st.container()

    _ = """
    ###################################################################
    ######################## Table of contents #########################
    ###################################################################

    [0] Table of contents
    """
    with container_summary:
        # Table of contents:
        # (update this manually)
        write_text_from_file(f'{p}3_Details_toc.txt',
                             head_lines_to_skip=2)

    _ = """
    ###################################################################
    ############################# Summary #############################
    ###################################################################

    [0] Summary

    [1] What models are in this app?
    +--------------+
    |              |
    +--------------+
    """
    with container_what_the_app_shows:
        st.markdown('# What the app shows')
        write_text_from_file(f'{p}3_Notes.txt',
                             head_lines_to_skip=2)
        st.image(f'{p}images/app_overview_models.png')
        st.caption(''.join([
            'A cartoon to show which parts of the app\'s ',
            'demo page are created by each model.'
            ]))

    _ = """
    ###################################################################
    ########################## Stroke teams ###########################
    ###################################################################

    [0] 🏥 Stroke teams

    [1] ### Which teams are included?

    [2] ### Why are teams given numbers?

    [3] ### What is my team called?

    [4] ### How is the data anonymised?
    """
    with container_stroke_teams:
        stroke_teams_text = read_text_from_file(
            f'{p}3_Details_Stroke_teams.txt',
            head_lines_to_skip=2
            )
        for i, text in enumerate(stroke_teams_text):
            st.markdown(text)

    _ = """
    ###################################################################
    ########################## Patient data ###########################
    ###################################################################

    [0] 📈 Patient data

    [1] ### Which features are used?
    +------------+--------------------+
    +------------+--------------------+
    +------------+--------------------+
    +------------+--------------------+
    +------------+--------------------+
    +------------+--------------------+

    [2] ### Why these features?
    """
    with container_patient_data:
        patient_data_text = read_text_from_file(
            f'{p}3_Details_Patient_data.txt',
            head_lines_to_skip=2
            )
        for i, text in enumerate(patient_data_text):
            st.markdown(text)

    _ = """
    ###################################################################
    ######################## Prediction model #########################
    ###################################################################

    [0] 🔮 Prediction model

    [1] ### What do the probabilities mean?
    """
    with container_prediction_model:
        prediction_model_text = read_text_from_file(
            f'{p}3_Details_Prediction_model.txt',
            head_lines_to_skip=2
            )
        for i, text in enumerate(prediction_model_text):
            st.markdown(text)

    _ = """
    ###################################################################
    ########################### SHAP demo #############################
    ###################################################################

    Tabs for:
      Highlighted teams /
      All Teams /
      Shifts for highlighted teams /
      Max/min/median teams
    """
    with container_shap_values:
        st.markdown('## ⚖️ SHAP values for this patient')
        st.info(''.join([
            'The patient details can be viewed and changed ',
            'in the left sidebar.'
            ]), icon='ℹ️')
        container_input_highlighted_teams = st.container()

        # User inputs
        with st.sidebar:
            use_plotly_events, container_input_patient_details = (
                set_up_sidebar(path_to_details))

        from utilities_ml.inputs import setup_for_app
        from utilities_ml.fixed_params import \
            default_highlighted_team, display_name_of_default_highlighted_team, \
            explainer_file, starting_probabilities
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

        # Load in the model:
        from utilities_ml.fixed_params import ml_model_file
        model = utilities_ml.inputs.load_pretrained_model(ml_model_file)
        # Main useful array:
        sorted_results = utilities_ml.main_calculations.\
            predict_treatment(X, model, stroke_teams_list,
                              highlighted_teams_list, benchmark_rank_list,
                              hb_teams_list)

        container_shapley_probs = st.container()
        with container_shapley_probs:

            container_shap_explanation = st.container()
            with container_shap_explanation:
                # Keep this explanation outside of a .txt file so that the
                # format string for starting probability can be added.
                st.markdown(''.join([
                    'Here we have visualised the process with waterfall plots.'
                ]))

            # Set up tabs:
            tabs_waterfall = st.tabs([
                'Highlighted teams',
                'All teams',
                'Shifts for highlighted teams',
                'Max/min/median teams',
                ])

            # Use this string in the first two tabs:
            waterfall_explanation_str = ''.join([
                    'The features are ordered from largest negative effect on ',
                    'probability to largest positive effect. ',
                    'The 9 largest features are shown individually and the rest ',
                    'are condensed into the "other features" bar. ',
                    'This bar mostly contains the effect of the patient _not_ ',
                    'attending the other stroke teams.'
                ])

            with tabs_waterfall[1]:
                # Combo waterfall explanation:
                st.markdown(
                    ''.join([
                        'The following chart shows the waterfall charts for all ',
                        'teams. Instead of red and blue bars, each team has ',
                        'a series of scatter points connected by lines. ',
                        'The features are ordered with the most agreed on features ',
                        'at the top, and the ones with more variation lower down. '
                    ]),
                    help=''.join([
                        '🔍 - [How do I read this plot?]'
                        f'({path_to_details}how-do-you-read-a-wizard-s-hat-plot)'
                    ])
                )

            with tabs_waterfall[2]:
                # Box plots explanation:
                st.markdown(
                    '''
                    Here the range of shifts in probability caused by
                    each feature are shown as violin plots.
                    The highlighted teams are overlaid as scatter circles.
                    ''',
                    help=''.join([
                        '🔍 - [How do I read a violin plot?]',
                        f'({path_to_details}how-do-you-read-a-violin-plot)'
                        ])
                )

        # explainer = utilities_ml.inputs.load_explainer()
        explainer_probability = utilities_ml.inputs.load_explainer_probability(
            explainer_file
            )

        # This is down here so that the starting probability is ok.
        with container_shap_explanation:
            st.markdown(
                '''
                + Before the model looks at any of the patient\'s details,
                the patient starts with a base probability''' +
                f' of {100.0*starting_probabilities:.2f}%.',
                help=''.join([
                    '🔍 - [Why this base probability?]',
                    f'({path_to_details}why-this-base-probability)'
                ])
                )
            st.markdown(
                '''

                + The model then looks at the value of each feature of
                the patient in turn, and adjusts this probability upwards
                or downwards. The size of the adjustment depends on which
                stroke team the patient is being assessed by.

                + The final probability is found when all of the features
                have been considered.
                ''')

        (
            indices_highlighted,
            shap_values_probability_extended_highlighted,
            indices_highlighted,
            df_waterfalls,
            final_probs,
            patient_data_waterfall,
            grid_cat_sorted,
            grid_cat_bench,
            grid_cat_nonbench,
            headers,
            shap_values_probability_extended_high_mid_low,
            indices_high_mid_low
        ) = utilities_ml.main_calculations.calculations_for_shap_values(
            sorted_results,
            explainer_probability,
            X,
            hb_teams_input,
            headers_X,
            starting_probabilities
            )

        with container_shapley_probs:

            with tabs_waterfall[0]:
                # Individual waterfalls for highlighted teams.
                if len(indices_highlighted) < 1:
                    # Nothing to see here
                    st.write('No teams are highlighted.')
                else:
                    # Individual waterfalls for the highlighted teams.
                    st.markdown(
                        waterfall_explanation_str,
                        help=''.join([
                            '🔍 - [How do I read a waterfall plot?]',
                            f'({path_to_details}how-do-you-read-a-waterfall-plot)'
                            ]))
                    draw_sneaky_bar()
                    utilities_ml.container_waterfalls.show_waterfalls_highlighted(
                        shap_values_probability_extended_highlighted,
                        indices_highlighted,
                        sorted_results,
                        default_highlighted_team,
                        display_name_of_default_highlighted_team,
                        model_version
                        )

            with tabs_waterfall[1]:
                # Combo waterfall (all teams).
                draw_sneaky_bar()
                utilities_ml.container_combo_waterfall.plot_combo_waterfalls(
                    df_waterfalls,
                    sorted_results,
                    final_probs,
                    patient_data_waterfall,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team,
                    use_plotly_events
                    )
                st.caption('''
                    If no graph appears above this message,
                    try changing the patient details
                    to reload the app.
                    ''')

            with tabs_waterfall[2]:
                # Box plots:
                utilities_ml.container_combo_waterfall.box_plot_of_prob_shifts(
                    grid_cat_sorted,
                    grid_cat_bench,
                    grid_cat_nonbench,
                    headers,
                    sorted_results,
                    hb_teams_input,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team,
                    starting_probabilities
                    )

            with tabs_waterfall[3]:
                # Individual waterfalls for the teams with the
                # max / median / min probabilities of thrombolysis.
                st.markdown(waterfall_explanation_str, help=''.join([
                    '🔍 - [How do I read a waterfall plot?]',
                    f'({path_to_details}how-do-you-read-a-waterfall-plot)'
                    ]))
                draw_sneaky_bar()
                utilities_ml.container_waterfalls.show_waterfalls_max_med_min(
                    shap_values_probability_extended_high_mid_low,
                    indices_high_mid_low,
                    sorted_results,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team,
                    model_version
                    )

    _ = """
    ###################################################################
    ########################## General SHAP ###########################
    ###################################################################

    [0] ## ⚖️ SHAP values

    [1] ### Why this base probability?

    [2] ### What are the general effects?
    """
    with container_shap_general:
        write_text_from_file(f'{p}3_Details_SHAP.txt', head_lines_to_skip=2)

        feats_dict = {
            'Anticoagulants': 'afib_anticoagulant',
            'Age': 'age',
            'Arrival to scan time': 'arrival_to_scan_time',
            'Infarction': 'infarction',
            'Onset during sleep': 'onset_during_sleep',
            'Onset to arrival time': 'onset_to_arrival_time',
            'Precise onset known': 'precise_onset_known',
            'Prior disability': 'prior_disability',
            'Stroke severity': 'stroke_severity'
        }
        feat_pretty = st.selectbox(
            'Select a feature to display',
            options=list(feats_dict.keys())
            )
        feat = feats_dict[feat_pretty]
        image_name = f'04_xgb_10_features_thrombolysis_shap_boxplot_{feat}.jpg'
        st.image(f'{p}images/{image_name}')

        # Write some information about that image.
        # First define a string with the info for each image,
        # and then use another dictionary to pick out the relevant string.
        afib_anticoagulant_str = '''
    When the patient is on anticoagulants, the "feature value" at the bottom is 1.
    The average SHAP value is below -1.5 when the patient is on anticoagulants.
    This means that the patient is considerably less likely to be treated
    than if they were not on blood-thinning medication.
    '''
        age_str = '''
    The average SHAP value decreases as age increases.
    For patients below the age of 80, the SHAP value is positive and so
    there is a small increase in the chance of thrombolysis.
    For patients above the age of 80, the SHAP value is negative and becomes
    more and more negative with increasing age. This reduces the chance of these
    older patients receiving thrombolysis.
    '''
        arrival_to_scan_time_str = '''
    The average SHAP value decreases as the time between arrival at hospital
    and receiving a brain scan increases.
    On average, every extra 30 minutes decreases the SHAP value by around 0.5.
    That means that after every hour on average, the chance of receiving
    thrombolysis becomes three times smaller.
    '''
        infarction_str = '''
    When the "feature value" at the bottom is 1, the patient has an infarction
    which is also called a blood clot. When the patient does not have a blood clot,
    the average SHAP value is very negative at around minus 9. This means that
    the chance of being given thrombolysis becomes very small. This reflects how
    in real life, clot-busting drugs should not be given when there is no clot.
    '''
        onset_during_sleep_str = '''
    When the "feature value" at the bottom is 1, the patient\'s stroke began
    while they were asleep. This means that the exact time that the stroke
    started is less accurately known, and so the chance of their being treated
    with thrombolysis is reduced considerably. The average SHAP value in this
    case is below minus 1.
    '''
        onset_to_arrival_time_str = '''
    The average SHAP value decreases as the time between the start of the stroke and
    the arrival at hospital increases. Patients arriving before 120 minutes are
    slightly more likely to receive treatment. Patients arriving after 120 minutes
    are slightly less likely to receive treatment.
    '''
        precise_onset_known_str = '''
    When the "feature value" at the bottom is 0, the time that the stroke started
    is not known precisely. The average SHAP value in this case is almost minus 1.
    This makes the patient somewhat less likely to receive thrombolysis.
    '''
        prior_disability_str = '''
    As the prior disability score increases, the average SHAP value decreases.
    This means that patients who had a higher level of disability before the
    stroke began are somewhat less likely to be treated than patients with
    no pre-stroke disability.
    '''
        stroke_severity_str = '''
    In this chart, each separate score for stroke severity has its own box.
    The average SHAP value is negative for low scores below 5
    and for high scores above 32. For scores between those values,
    the average SHAP value is positive. This means that the most mild and the
    most severe strokes are somewhat less likely to be treated with thrombolysis,
    and the most moderate strokes are somewhat more likely to be treated
    with thrombolysis.
    '''

        feats_str_dict = {
            'Anticoagulants': afib_anticoagulant_str,
            'Age': age_str,
            'Arrival to scan time': arrival_to_scan_time_str,
            'Infarction': infarction_str,
            'Onset during sleep': onset_during_sleep_str,
            'Onset to arrival time': onset_to_arrival_time_str,
            'Precise onset known': precise_onset_known_str,
            'Prior disability': prior_disability_str,
            'Stroke severity': stroke_severity_str
        }
        st.markdown(feats_str_dict[feat_pretty])

    _ = """
    ###################################################################
    ######################## Waterfall plots ##########################
    ###################################################################

    [0] ## ⛲ Waterfall plots

    [1] ### How do you read a waterfall plot?
    +--------------+
    |              |
    +--------------+
    [2] ### What does this example show?
    """
    with container_waterfall:
        waterfall_text = read_text_from_file(
            f'{p}3_Details_Waterfall.txt',
            head_lines_to_skip=2
            )
        st.markdown(waterfall_text[0])
        st.markdown(waterfall_text[1])
        st.image(f'{p}images/waterfall.png')
        st.caption('An example waterfall plot.')
        st.markdown(waterfall_text[2])
        with st.expander('Detailed look'):
            st.markdown(waterfall_text[3].strip('###'))

    _ = """
    ###################################################################
    ###################### Wizard's hat plots #########################
    ###################################################################

    [0] ## 🧙‍♀️ Wizard hat plots

    [1] ### How do you read a wizard's hat plot?
    +--------------+
    |              |
    +--------------+
    [2] ### What does this example show?
    """
    with container_wizard_hat:
        wizard_text = read_text_from_file(
            f'{p}3_Details_Wizard_Hat.txt',
            head_lines_to_skip=2
            )
        st.markdown(wizard_text[0])
        st.markdown(wizard_text[1])
        st.image(f'{p}images/wizard.png')
        st.caption('An example wizard\'s hat plot.')
        st.markdown(wizard_text[2])

    _ = """
    ###################################################################
    ######################## Violin plots #############################
    ###################################################################

    [0] ## 🎻 Violin plots

    [1] ### How do you read a violin plot?

    +----+  [2] ### What does this example show?
    |    |
    |    |
    +----+
    [3] What do the circle markers mean?
    """
    with container_violin:
        violin_text = read_text_from_file(
            f'{p}3_Details_Violin.txt',
            head_lines_to_skip=2
            )
        st.markdown(violin_text[0])
        st.markdown(violin_text[1])
        cols_violin = st.columns(2)
        with cols_violin[0]:
            st.image(f'{p}images/violin.png')
            st.caption(''.join([
                'Three example violin plots. ',
                'The top row shows the range of values for all teams. ',
                'The middle row shows only benchmark teams, and ',
                'the bottom row shows only non-benchmark teams. ',
                'The red circle highlights the value of one particular team.'
                ]))
        with cols_violin[1]:
            st.markdown(violin_text[2])
        st.markdown(violin_text[3])

    _ = """
    ###################################################################
    ###################### How the app works ##########################
    ###################################################################

    [0] # How the app works
    """
    with container_how_the_app_works:
        st.markdown('-'*50)
        st.markdown('# How the app works')

    _ = """
    ###################################################################
    ########################## SSNAP data #############################
    ###################################################################

    [0] ## 🧮 The data

    [1] ### What data do we use?
    +--------------+
    |              |
    +--------------+
    [2] ### What data goes into the models?
    """
    with container_ssnap:
        ssnap_text = read_text_from_file(
            f'{p}3_Details_SSNAP.txt',
            head_lines_to_skip=2
            )
        st.markdown(ssnap_text[0])
        st.markdown(ssnap_text[1])
        st.image(f'{p}images/process_data_split.png')
        st.markdown(ssnap_text[2])
    # st.markdown('# 🗃️ 🗂️ 📁 ')

    _ = """
    ###################################################################
    ################### Create prediction model #######################
    ###################################################################

    [0] ## 🔮 Prediction model
    +--------------+
    |              |
    +--------------+
    [1] ### Creating the decision model
    +--------------+
    |              |
    +--------------+
    [2] ### How do we know it works?
    +--------------+
    |              |
    +--------------+
    """
    with container_create_prediction_model:
        create_prediction_text = read_text_from_file(
            f'{p}3_Details_Create_prediction_model.txt',
            head_lines_to_skip=2
            )
        st.markdown(create_prediction_text[0])
        st.image(f'{p}images/process_decision_model.png')
        st.markdown(create_prediction_text[1])
        st.image(f'{p}images/process_create_decision_model.png')
        st.markdown(create_prediction_text[2])

    _ = """
    ###################################################################
    ###################### Create SHAP model ##########################
    ###################################################################

    [0] ## ⚖️ SHAP model
    [1] ### Creating the SHAP explainer model
    +--------------+
    |              |
    +--------------+
    """
    with container_create_shap_model:
        create_shap_model_text = read_text_from_file(
            f'{p}3_Details_Create_SHAP_model.txt',
            head_lines_to_skip=2
            )
        st.markdown(create_shap_model_text[0])
        st.markdown(create_shap_model_text[1])
        st.image(f'{p}images/process_create_shap_models.png')

    _ = """
    ###################################################################
    ######################### SHAP values #############################
    ###################################################################

    [0] ## ⚖️ SHAP values
    +--------------+
    |              |
    +--------------+
    [1] ### Creating a waterfall plot
    +--------------+
    |              |
    +--------------+
    [2] ### Creating a wizard's hat plot
    +--------------+
    |              |
    +--------------+
    """
    with container_create_shap_values:
        create_shap_values_text = read_text_from_file(
            f'{p}3_Details_Create_SHAP_values.txt',
            head_lines_to_skip=2
            )
        st.markdown(create_shap_values_text[0])
        st.image(f'{p}images/process_shap_model.png')
        st.markdown(create_shap_values_text[1])
        st.image(f'{p}images/process_waterfall.png')
        st.markdown(create_shap_values_text[2])
        st.image(f'{p}images/process_combo_waterfall.png')

    _ = """
    ###################################################################
    ####################### Benchmark teams ###########################
    ###################################################################

    [0] ## 🎯 Benchmark teams

    [1] ### What are benchmark teams?

    +----+  [2] ### How are the benchmark teams picked?
    |    |
    |    |
    +----+
    """
    with container_benchmark:
        bench_text = read_text_from_file(
            f'{p}3_Details_Benchmark_teams.txt',
            head_lines_to_skip=2
            )
        st.markdown(bench_text[0])
        st.markdown(bench_text[1])
        cols_bench = st.columns(2)
        with cols_bench[0]:
            st.image(f'{p}images/flowchart_define_benchmarks.png')
        with cols_bench[1]:
            st.markdown(bench_text[2])


if __name__ == '__main__':
    main()
