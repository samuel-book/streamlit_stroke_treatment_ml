"""
Streamlit app template.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import numpy as np

# Add an extra bit to the path if we need to.
# Try importing something as though we're running this from the same
# directory as the landing page.
try:
    from utilities_ml.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./streamlit_stroke_treatment_ml/')

# Custom functions:
from utilities_ml.fixed_params import page_setup, starting_probabilities, \
    plain_str, bench_str
# from utilities_ml.inputs import \
#     write_text_from_file
import utilities_ml.inputs
import utilities_ml.main_calculations
# Containers:
import utilities_ml.container_inputs
import utilities_ml.container_metrics
import utilities_ml.container_bars
import utilities_ml.container_waterfalls
import utilities_ml.container_combo_waterfall
import utilities_ml.container_results
# import utilities_ml.container_details

from utilities_ml.plot_utils import remove_old_colours_for_highlights, \
                                 choose_colours_for_highlights

def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Title:
    st.markdown('# Interactive demo')
    st.markdown(''.join([
        'To use this demo, '
        'change the patient details in the left sidebar.'
        ]))

    st.markdown(''.join([
        # 'The line at 50% is the cut-off for thrombolysis. ',
        'The model returns the probability of each team thrombolysing ',
        'this patient. '
        'Stroke teams with a probability below 50% are unlikely to ',
        'thrombolyse the patient, and other teams are ',
        'likely to thrombolyse. ',
        'We record teams below 50% as :x: not thrombolysing this patient ',
        'and other teams as :heavy_check_mark: thrombolysing this patient.'
        ]))

    # Draw some empty containers on the page.
    # They'll appear in this order, but we'll fill them in another order.
    container_metrics = st.container()
    with container_metrics:
        st.markdown(''.join([
            '### How many stroke teams _would_ thrombolyse this patient?'
            ]))

    container_bar_chart = st.container()
    with container_bar_chart:
        st.markdown(''.join([
            '### Probability of thrombolysis from each team'
            ]))

    container_shapley_probs = st.container()
    with container_shapley_probs:
        st.markdown('### Probability waterfalls')
        st.markdown(''.join([
            'We can look at how the model decides on the probability ',
            'of thrombolysis. ',
            'Before the model looks at any of the ',
            'patient\'s details, the patient starts with a base probability',
            f' of {100.0*starting_probabilities:.2f}%',
            '. '
            'The model then looks at the value of each feature of the patient ',
            'in turn, and adjusts this probability upwards or downwards.'
        ]))
        st.markdown(''.join([
            'The process can be visualised as a waterfall plot.'
        ]))


    # ###########################
    # ########## SETUP ##########
    # ###########################

    with st.sidebar:
        st.markdown('# Patient details')
        user_inputs_dict = utilities_ml.container_inputs.user_inputs()
        # Write an empty header to give breathing room at the bottom:
        st.markdown('# ')
    stroke_teams_list = utilities_ml.inputs.read_stroke_teams_from_file()


    # Build these into a 2D DataFrame:
    X, headers_X, headers_synthetic = utilities_ml.inputs.\
        build_X(user_inputs_dict, stroke_teams_list)

    # Load in the model and explainers separately so each can be cached:
    model = utilities_ml.inputs.load_pretrained_model()
    explainer = utilities_ml.inputs.load_explainer()
    explainer_probability = utilities_ml.inputs.load_explainer_probability()


    # ----- Benchmark teams -----
    benchmark_df = utilities_ml.inputs.import_benchmark_data()
    # Make list of benchmark rank:
    # (original data is sorted alphabetically by stroke team)
    benchmark_rank_list = \
        benchmark_df.sort_values('stroke_team')['Rank'].to_numpy()
    # Indices of benchmark data at the moment:
    inds_benchmark = np.where(benchmark_rank_list <= 30)[0]


    # ----- Highlighted teams -----

    # Receive the user inputs now and show this container now:
    with container_bar_chart:
        st.markdown(''.join([
            'To highlight stroke teams on the following charts, ',
            'select them in this box or click on them in the charts.'
        ]))
        # Pick teams to highlight on the bar chart:
        highlighted_teams_input = utilities_ml.container_inputs.\
            highlighted_teams(stroke_teams_list)
    # Update the "Highlighted teams" column:
    # Label benchmarks:
    # table[np.where(table[:, 7] <= 30), 6] = 'Benchmark'
    highlighted_teams_list = np.array(
        ['-' for team in stroke_teams_list], dtype=object)
    # Combo highlighted and benchmark:
    hb_teams_list = np.array(
        [plain_str for team in stroke_teams_list], dtype=object)
    hb_teams_list[inds_benchmark] = bench_str
    # Put in selected Highlighteds (overwrites benchmarks):
    # inds_highlighted = []
    hb_teams_input = [plain_str, bench_str]
    for team in highlighted_teams_input:
        ind_t = np.argwhere(stroke_teams_list == team)[0][0]
        # inds_highlighted.append(ind_t)
        highlighted_teams_list[ind_t] = team
        if ind_t in inds_benchmark:
            team = team + ' \U00002605'
        hb_teams_list[ind_t] = team
        hb_teams_input.append(team)


    st.session_state['hb_teams_input'] = hb_teams_input

    # Find colour lists for plotting (saved to session state):
    remove_old_colours_for_highlights(hb_teams_input)
    choose_colours_for_highlights(hb_teams_input)


    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################

    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                        highlighted_teams_list, benchmark_rank_list,
                        hb_teams_list)

    # Pull out the feature values:
    patient_data_waterfall = X.iloc[0][:9].to_numpy()
    # Add empty value for stroke team attended:
    patient_data_waterfall = np.append(patient_data_waterfall, '')

    # Find which values are 0/1 choice and can be changed to no/yes:
    features_yn = [
        'Infarction',
        'Precise onset time',
        'Use of AF anticoagulants',
        'Onset during sleep',
    ]

    for feature in features_yn:
        i = np.where(np.array(headers_X) == feature)[0]
        patient_data_waterfall[i] = 'Yes' if patient_data_waterfall[i] > 0 else 'No'


    # Get indices of highest, most average, and lowest probability teams.
    index_high = sorted_results.iloc[0]['Index']
    index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
    index_low = sorted_results.iloc[-1]['Index']
    indices_high_mid_low = [index_high, index_mid, index_low]

    # Get indices of highlighted teams:
    indices_highlighted = []
    for team in hb_teams_input:
        if '-' not in team and 'Benchmark' not in team:
            ind_team = sorted_results['Index'][
                sorted_results['HB team'] == team].values[0]
            indices_highlighted.append(ind_team)

    # Find Shapley values only for the important indices:
    (shap_values_probability_extended_high_mid_low,
    shap_values_probability_high_mid_low) = \
        utilities_ml.main_calculations.find_shapley_values(
            explainer_probability, X.iloc[indices_high_mid_low])

    if len(indices_highlighted) > 0:
        (shap_values_probability_extended_highlighted,
        shap_values_probability_highlighted) = \
            utilities_ml.main_calculations.find_shapley_values(
                explainer_probability, X.iloc[indices_highlighted])
    else:
        shap_values_probability_extended_highlighted = None
        shap_values_probability_highlighted = None


    # Make Shapley values for all indices:
    (shap_values_probability_extended_all,
        shap_values_probability_all) = \
        utilities_ml.main_calculations.find_shapley_values(
            explainer_probability, X)

    # Stuff for displaying SHAP probabilities:

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
    inds_bench = np.where(sorted_results['Benchmark rank'].to_numpy() <= 30)[0]
    inds_nonbench = np.where(sorted_results['Benchmark rank'].to_numpy() > 30)[0]

    grid_cat_bench = grid_cat_sorted[:, inds_bench]
    grid_cat_nonbench = grid_cat_sorted[:, inds_nonbench]


    # Make dataframe for combo waterfalls:
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


    # ###########################
    # ######### RESULTS #########
    # ###########################

    with container_metrics:
        # Print metrics for how many teams would thrombolyse:
        utilities_ml.container_metrics.main(sorted_results)

    with container_bar_chart:
        utilities_ml.container_bars.main(sorted_results)

    with container_shapley_probs:
        # Set up tabs:
        tabs_waterfall = st.tabs([
            'Max/min/median teams',
            'Highlighted teams',
            'All teams',
            'Shifts for highlighted teams'
            ])

        # Use this string in the first two tabs:
        waterfall_explanation_str = ''.join([
                'The features are ordered from largest negative effect on ',
                'probability to largest positive effect. ',
                'The 9 largest features are shown individually and the rest ',
                'are condensed into the "132 other features" bar. ',
                'This bar mostly contains the effect of the patient _not_ ',
                'attending the other stroke teams.'
            ])

        with tabs_waterfall[0]:
            # Individual waterfalls for the teams with the
            # max / median / min probabilities of thrombolysis.
            st.markdown(waterfall_explanation_str)
            utilities_ml.container_waterfalls.show_waterfalls_max_med_min(
                shap_values_probability_extended_high_mid_low,
                indices_high_mid_low,
                sorted_results
                )

        # Highlighted teams
        with tabs_waterfall[1]:
            if len(indices_highlighted) < 1:
                # Nothing to see here
                st.write('No teams are highlighted.')
            else:
                # Individual waterfalls for the highlighted teams.
                st.markdown(waterfall_explanation_str)
                utilities_ml.container_waterfalls.show_waterfalls_highlighted(
                    shap_values_probability_extended_highlighted,
                    indices_highlighted,
                    sorted_results
                    )

        with tabs_waterfall[2]:
            # Combo waterfall:
            st.markdown(''.join([
                'The following chart shows the waterfall charts for all ',
                'teams. Instead of red and blue bars, each team has ',
                'a series of scatter points connected by lines. ',
                'The features are ordered with the most agreed on features ',
                'at the top, and the ones with more variation lower down. '
            ]))
            utilities_ml.container_combo_waterfall.plot_combo_waterfalls(
                df_waterfalls,
                sorted_results,
                final_probs,
                patient_data_waterfall
                )

        with tabs_waterfall[3]:
            # Box plots:
            utilities_ml.container_combo_waterfall.box_plot_of_prob_shifts(
                grid_cat_sorted,
                grid_cat_bench,
                grid_cat_nonbench,
                headers,
                sorted_results
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