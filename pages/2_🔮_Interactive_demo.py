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
from utilities_ml.fixed_params import \
    plain_str, bench_str, draw_sneaky_bar, \
    write_markdown_in_colour
import utilities_ml.inputs
import utilities_ml.main_calculations
# Containers:
import utilities_ml.container_inputs
import utilities_ml.container_metrics
import utilities_ml.container_bars
import utilities_ml.container_waterfalls
import utilities_ml.container_combo_waterfall
import utilities_ml.container_results

from utilities_ml.plot_utils import remove_old_colours_for_highlights, \
                                    choose_colours_for_highlights


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# :crystal_ball: Thrombolysis decisions')

    st.markdown('''
        The SAMueL-2 model gives the probability of
        any stroke team thrombolysing any patient.
        ''')

    # Draw the image with the basic model summary.
    try:
        st.image('./utilities_ml/SAMueL2_model_wide.png')
    except (FileNotFoundError,
            st.runtime.media_file_storage.MediaFileStorageError):
        # Add an extra bit to the path for the combo app.
        st.image('./streamlit_stroke_treatment_ml/' +
                 'utilities_ml/SAMueL2_model_wide.png')

    st.markdown('''
        We can use the model to give the same patient details
        to all stroke teams and decide how likely each team is
        to thrombolyse that patient.

        Stroke teams with a probability below 50% are unlikely to
        thrombolyse the patient, and other teams are
        likely to thrombolyse.
        ''')

    st.markdown('### How we categorise the results')
    cols_method = st.columns(3)

    with cols_method[1]:
        st.error(
            '''
            __Benchmark teams__

            These teams are consistently more likely than average to
            thrombolyse any given patient.

            They are used to compare the treatment decisions
            of stroke teams with highly-thrombolysing units.
            '''
            )
    with cols_method[0]:
        st.info(
            '''
            __Thrombolysis: yes or no?__

            If probability is at least 50%:

            :heavy_check_mark: thrombolyse

            If probability is below 50%:

            :x: do not thrombolyse
            '''
            )

    with cols_method[2]:
        st.error(
            '''
            __Benchmark decision__

            If at least half of the benchmark teams would thrombolyse:

            :heavy_check_mark: thrombolyse

            Otherwise:

            :x: do not thrombolyse
            '''
            )
        # If under half of the benchmark teams would thrombolyse:

    st.markdown('-' * 50)
    st.markdown('''
        ## :abacus: Predictions for this patient
        ''')
    st.info(''.join([
        'The patient details can be viewed and changed ',
        'in the left sidebar.'
        ]), icon='ℹ️')

    # Draw some empty containers on the page.
    # They'll appear in this order, but we'll fill them in another order.
    container_metrics = st.container()
    with container_metrics:
        st.markdown('## ')  # Breathing room
        st.markdown(''.join([
            '### ',  # :abacus: ',
            'How many teams would thrombolyse this patient?'
            ]))

    container_highlighted_summary = st.container()
    with container_highlighted_summary:
        st.markdown('## ')  # Breathing room
        st.markdown(''.join([
            '### ',  # :chart_with_upwards_trend: ',
            'What would your team do?'
            ]))

    container_bar_chart = st.container()
    with container_bar_chart:
        st.markdown('## ')  # Breathing room
        st.markdown(''.join([
            '### ',  # :chart_with_upwards_trend: ',
            'What would each of the teams do?'
            ]))
        st.caption('To see the team names, hover or click on a bar.')

    container_shapley_probs = st.container()
    with container_shapley_probs:
        st.markdown('## ')  # Breathing room
        st.markdown('-' * 50)

        container_shap_explanation = st.container()
        with container_shap_explanation:
            # Keep this explanation outside of a .txt file so that the
            # format string for starting probability can be added.
            st.markdown('## :scales: How does the model work?')
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
            st.markdown(''.join([
                'The following chart shows the waterfall charts for all ',
                'teams. Instead of red and blue bars, each team has ',
                'a series of scatter points connected by lines. ',
                'The features are ordered with the most agreed on features ',
                'at the top, and the ones with more variation lower down. '
            ]))

        with tabs_waterfall[2]:
            # Box plots explanation:
            st.markdown('''
                Here the range of shifts in probability caused by
                each feature are shown as violin plots.
                The highlighted teams are overlaid as scatter circles.
                ''')

    # ###########################
    # ########## SETUP ##########
    # ###########################

    # ----- Build the X array -----
    # All patient detail widgets go in the sidebar:
    with st.sidebar:
        st.markdown('## Patient details')
        user_inputs_dict = utilities_ml.container_inputs.user_inputs()
        # Write an empty header to give breathing room at the bottom:
        # st.markdown('# ')

        # Decide whether to use new or old model:
        st.markdown('-'*50)
        st.markdown('## Advanced options')
        # model_version = st.radio(
        #     'Model version',
        #     ['SAMueL-1: December 2022',
        #      'SAMueL-2: April 2023'],
        #     index=1  # Initial selection
        # )
        model_version = 'SAMueL-2: August 2023'

    # if 'SAMueL-1' in model_version:
    stroke_teams_file = 'stroke_teams.csv'
    ml_model_file = 'model.p'
    explainer_file = 'shap_explainer_probability.p'

    # Stroke team column heading for the model
    stroke_team_col = 'stroke_team_id'

    # Benchmark teams:
    benchmark_filename = 'benchmark_codes.csv'
    benchmark_team_column = 'stroke_team_id'

    # Default highlighted team:
    default_highlighted_team = '42' # 'LECHF1024T'
    display_name_of_default_highlighted_team = str(default_highlighted_team) # '"St Elsewhere"'

    starting_probabilities = 0.3481594278820853 # 0.42355026099306586

    # else:
    #     stroke_teams_file = 'stroke_teams_samuel2_anon.csv'
    #     ml_model_file = 'thrombolysis_xgb_model_anonymised_2017_2019.pkl'
    #     explainer_file = \
    #         'thrombolysis_xgb_explainer_anonymised_probability_2017_2019.pkl'

    #     # Stroke team column heading for the model
    #     stroke_team_col = 'stroke team'

    #     # Benchmark teams:
    #     benchmark_filename = 'benchmark_codes.csv'
    #     benchmark_team_column = 'Hospital code'

    #     # Default highlighted team:
    #     default_highlighted_team = 'team_100'
    #     display_name_of_default_highlighted_team = 'team_100'

    #     starting_probabilities = 0.3314

    # This is down here so that the starting probability is ok.
    with container_shap_explanation:
        st.markdown(
            '''
            + Before the model looks at any of the patient\'s details,
            the patient starts with a base probability''' +
            f' of {100.0*starting_probabilities:.2f}%.' + '''

            + The model then looks at the value of each feature of
            the patient in turn, and adjusts this probability upwards
            or downwards. The size of the adjustment depends on which
            stroke team the patient is being assessed by.

            + The final probability is found when all of the features
            have been considered.
            ''')

    # List of stroke teams that this patient will be sent to:
    stroke_teams_list = utilities_ml.inputs.read_stroke_teams_from_file(
        stroke_teams_file
    )

    # Build these into a 2D DataFrame:
    X, headers_X = utilities_ml.inputs.build_X(
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

    # ----- Highlighted teams -----
    # The user can select teams to highlight on various plots.
    # The teams are selected using either a streamlit input widget
    # in the following container, or by clicking directly on
    # certain plots.
    # Receive the user inputs now and show this container now:
    with container_highlighted_summary:
        st.caption(''.join([
            'To highlight stroke teams, ',
            'select them in this box ',
            'or click on them in the interactive charts.'
        ]))

        cols_highlighted_summary = st.columns(4)
        with cols_highlighted_summary[0]:
            # Pick teams to highlight on the bar chart:
            highlighted_teams_input = utilities_ml.container_inputs.\
                highlighted_teams(
                    stroke_teams_list,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team
                    )

    # Columns for highlighted teams and highlighted+benchmark (HB),
    # and a shorter list hb_teams_input with just the unique values
    # from hb_teams_list in the order that the highlighted teams
    # were added to the highlighted input list.
    highlighted_teams_list, hb_teams_list, hb_teams_input = \
        utilities_ml.inputs.find_highlighted_hb_teams(
            stroke_teams_list,
            inds_benchmark,
            highlighted_teams_input,
            default_highlighted_team,
            display_name_of_default_highlighted_team
            )

    # Find colour lists for plotting (saved to session state):
    remove_old_colours_for_highlights(hb_teams_input)
    choose_colours_for_highlights(hb_teams_input)

    # Load in the model and explainers separately so each can be cached:
    model = utilities_ml.inputs.load_pretrained_model(
        ml_model_file
        )
    # explainer = utilities_ml.inputs.load_explainer()
    explainer_probability = utilities_ml.inputs.load_explainer_probability(
        explainer_file
        )

    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################

    # Main useful array:
    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                          highlighted_teams_list, benchmark_rank_list,
                          hb_teams_list)

    # ----- Shapley probabilities -----
    # Make Shapley values for all teams:
    shap_values_probability_extended_all, shap_values_probability_all = \
        utilities_ml.main_calculations.find_shapley_values(
            explainer_probability, X)

    # Make separate arrays of the Shapley values for certain teams.
    # Get indices of highest, most average, and lowest probability teams.
    index_high = sorted_results.iloc[0]['Index']
    index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
    index_low = sorted_results.iloc[-1]['Index']
    indices_high_mid_low = [index_high, index_mid, index_low]
    # Get indices of highlighted teams:
    indices_highlighted = []
    for team in hb_teams_input:
        if plain_str not in team and bench_str not in team:
            # If it's not the default benchmark or non-benchmark
            # team label, then add this index to the list:
            ind_team = sorted_results['Index'][
                sorted_results['HB team'] == team].values[0]
            indices_highlighted.append(ind_team)

    # Shapley values for the high/mid/low indices:
    shap_values_probability_extended_high_mid_low = \
        shap_values_probability_extended_all[indices_high_mid_low]
    # Shapley values for the highlighted indices:
    shap_values_probability_extended_highlighted = \
        shap_values_probability_extended_all[indices_highlighted]

    # ----- Other grids and dataframes for Shap probabilities:
    # Get big SHAP probability grid:
    grid, grid_cat_sorted, stroke_team_2d, headers = \
        utilities_ml.main_calculations.make_heat_grids(
            headers_X,
            sorted_results['Stroke team'],
            sorted_results['Index'],
            shap_values_probability_all
            )
    # These grids have teams in the same order as sorted_results.

    # Pick out the subset of benchmark teams:
    inds_bench = np.where(
        sorted_results['Benchmark rank'].to_numpy() <= n_benchmark_teams)[0]
    inds_nonbench = np.where(
        sorted_results['Benchmark rank'].to_numpy() > n_benchmark_teams)[0]
    # Make separate grids of just the benchmark or non-benchmark teams:
    grid_cat_bench = grid_cat_sorted[:, inds_bench]
    grid_cat_nonbench = grid_cat_sorted[:, inds_nonbench]

    # Make a list of the input patient data for labelling
    # features+values on e.g. the combined waterfall plot.
    # Pull out the feature values:
    patient_data_waterfall = X.iloc[0][:9].to_numpy()
    # Add empty value for stroke team attended:
    patient_data_waterfall = np.append(patient_data_waterfall, '')
    # Find which values are 0/1 choice and can be changed to no/yes:
    if 'SAMueL-1' in model_version:
        features_yn = [
            'Infarction',
            'Precise onset time',
            'Use of AF anticoagulants',
            'Onset during sleep',
        ]
    else:
        features_yn = [
            'infarction',
            'precise onset known',
            'use of AF anticoagulants',
            'onset during sleep'
        ]
    for feature in features_yn:
        i = np.where(np.array(headers_X) == feature)[0]
        # Annoying nested list to pacify DeprecationWarning for
        # checking for element of empty array.
        if patient_data_waterfall[i].size > 0:
            if patient_data_waterfall[i] > 0:
                patient_data_waterfall[i] = 'Yes'
            else:
                patient_data_waterfall[i] = 'No'
        else:
            patient_data_waterfall[i] = 'No'
    # Resulting list format e.g.:
    #     [15, 'Yes', 15, 'Yes', 0, 'No', 90, 'No', 72.5, '']
    # where headers_X provides the feature names to match the values.

    # Make dataframe for combo waterfalls:
    # patient_data_waterfall is returned here with the order of the
    # values switched to match the order the features are plotted in
    # in the combo waterfall.
    df_waterfalls, final_probs, patient_data_waterfall = \
        utilities_ml.main_calculations.make_waterfall_df(
            grid_cat_sorted,
            headers,
            sorted_results['Stroke team'],
            sorted_results['Highlighted team'],
            sorted_results['HB team'],
            patient_data_waterfall,
            base_values=starting_probabilities
            )
    # Columns of df_waterfalls:
    #     Sorted rank
    #     Stroke team
    #     Probabilities
    #     Prob shift
    #     Prob final
    #     Features
    #     Highlighted team
    #     HB team

    # ###########################
    # ######### RESULTS #########
    # ###########################

    # Add an option for removing plotly_events()
    # which doesn't play well on skinny screens / touch devices.
    with st.sidebar:
        # st.markdown('-' * 50)
        st.markdown('##')  # Adds a gap before this option.
        if st.checkbox('Disable interactive plots'):
            use_plotly_events = False
        else:
            use_plotly_events = True
        st.caption(''.join([
            'The clickable plots sometimes appear strange ',
            'on small screens and touch devices, ',
            'so select this option to convert them to normal plots.'
        ]))

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
                    # st.markdown('-'*50)
                    # cols = st.columns(4)
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
                        emoji_here = ':heavy_check_mark: '
                        extra_str = ''
                    else:
                        emoji_here = ':x: '
                        extra_str = 'do not '
                    st.markdown(f'Probability: {prob_here:.2f}%')
                    st.markdown(emoji_here + extra_str + 'thrombolyse')
                    st.markdown('-' * 50)
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

    with container_shapley_probs:

        with tabs_waterfall[0]:
            # Individual waterfalls for highlighted teams.
            if len(indices_highlighted) < 1:
                # Nothing to see here
                st.write('No teams are highlighted.')
            else:
                # Individual waterfalls for the highlighted teams.
                st.markdown(waterfall_explanation_str)
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
            st.markdown(waterfall_explanation_str)
            draw_sneaky_bar()
            utilities_ml.container_waterfalls.show_waterfalls_max_med_min(
                shap_values_probability_extended_high_mid_low,
                indices_high_mid_low,
                sorted_results,
                default_highlighted_team,
                display_name_of_default_highlighted_team,
                model_version
                )

    # ----- The end (usually)! -----

    # #################################
    # ######### SANITY CHECKS #########
    # #################################
    show_sanity_check_plots = False
    if show_sanity_check_plots is False:
        # Do nothing
        pass
    else:
        # Plots for testing and sanity checking:
        # Imshow grid of all SHAP probability values:
        utilities_ml.container_results.plot_heat_grid_full(grid)
        # Same imshow grid, but the one-hot-encoded stroke teams are
        # compressed into two rows: "this team" and "other teams".
        utilities_ml.container_results.plot_heat_grid_compressed(
            grid_cat_sorted,
            sorted_results['Stroke team'],
            headers,
            stroke_team_2d
            )
        # Line chart equivalent of compressed imshow grid:
        utilities_ml.container_results.\
            plot_all_prob_shifts_for_all_features_and_teams(
                headers,
                grid_cat_sorted
                )

        # Write statistics (median/std/min/max probability shift)
        # for each feature:
        st.markdown('All teams: ')
        inds_std = utilities_ml.container_results.write_feature_means_stds(
            grid_cat_sorted, headers, return_inds=True)
        st.markdown('Benchmark teams: ')
        utilities_ml.container_results.write_feature_means_stds(
            grid_cat_bench, headers, inds=inds_std)

        # Write sentences about the biggest probability shifts:
        utilities_ml.container_results.print_changes_info(
            grid_cat_sorted, headers, stroke_team_2d)

    # ----- The end! -----


if __name__ == '__main__':
    main()
