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
# Garbage collection to help reduce memory creep:
import gc

# Custom functions:
from utilities.fixed_params import page_setup
# from utilities.inputs import \
#     write_text_from_file
import utilities.inputs
import utilities.main_calculations
# Containers:
import utilities.container_inputs
import utilities.container_results
# import utilities.container_details

from utilities.plot_utils import remove_old_colours_for_highlights, \
                                 choose_colours_for_highlights

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

container_team_probs = st.container()
with container_team_probs:
    st.markdown(''.join([
        '### Probability of thrombolysis from each team'
        ]))

# # Draw a blue information box:
# st.info(
#     ':information_source: ' +
#     'For acronym reference, see the introduction page.'
#     )
# # Intro text:
# write_text_from_file('pages/text_for_pages/2_Intro_for_demo.txt',
#                      head_lines_to_skip=2)


# ###########################
# ########## SETUP ##########
# ###########################

with st.sidebar:
    st.markdown('# Patient details')
    user_inputs_dict = utilities.container_inputs.user_inputs()
    # Write an empty header to give breathing room at the bottom:
    st.markdown('# ')
stroke_teams_list = utilities.inputs.read_stroke_teams_from_file()


# Build these into a 2D DataFrame:
X, headers_X, headers_synthetic = utilities.inputs.\
    build_X(user_inputs_dict, stroke_teams_list)

# Load in the model and explainers separately so each can be cached:
model = utilities.inputs.load_pretrained_model()
explainer = utilities.inputs.load_explainer()
explainer_probability = utilities.inputs.load_explainer_probability()


# ----- Benchmark teams -----
benchmark_df = utilities.inputs.import_benchmark_data()
# Make list of benchmark rank:
# (original data is sorted alphabetically by stroke team)
benchmark_rank_list = benchmark_df.sort_values('stroke_team')['Rank'].to_numpy()
# Indices of benchmark data at the moment:
inds_benchmark = np.where(benchmark_rank_list <= 30)[0]


# ----- Highlighted teams -----

bench_str = 'Benchmark \U00002605'
plain_str = '-'

# Receive the user inputs now and show this container now:
with container_team_probs:
    st.markdown(''.join([
        'To highlight stroke teams on the following charts, ',
        'select them in this box or click on them in the charts.'
    ]))
    # Pick teams to highlight on the bar chart:
    highlighted_teams_input = utilities.container_inputs.\
        highlighted_teams(stroke_teams_list)
# Update the "Highlighted teams" column:
# Label benchmarks:
# table[np.where(table[:, 7] <= 30), 6] = 'Benchmark'
highlighted_teams_list = np.array(
    ['-' for team in stroke_teams_list], dtype=object)
# Combo highlighted and benchmark:
hb_teams_list = np.array(
    ['-' for team in stroke_teams_list], dtype=object)
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

# highlighted_teams_list = highlighted_teams_list
# # If just need yes/no, the following works.
# # It doesn't return indices in the same order as Highlighted_teams.
# bool_Highlighteds = np.in1d(stroke_teams_list, Highlighted_teams)
# table[:, 6][bool_Highlighteds] = 'Yes'

# # Mark the benchmarks in the highlighted names. Add a star.
# # Star: '\U00002605'
# hb_teams_list = np.copy(highlighted_teams_list)
# hb_teams_list[inds_benchmark] = '\U00002605 ' + hb_teams_list[inds_benchmark]

# inds_benchmark_not_highlighted = np.where(hb_teams_list == '\U00002605 -')
# hb_teams_list[inds_benchmark_not_highlighted] = bench_str

# # Remove repeats:
# highlighted_teams_input_extras = [plain_str, bench_str] + highlighted_teams_input #np.unique(hb_teams_list)
# st.write(highlighted_teams_input_extras)
# ind_bench_str = np.where(highlighted_teams_input_extras == bench_str)[0][0]
# ind_plain_str = np.where(highlighted_teams_input_extras == plain_str)[0][0]
# inds_hite = sorted([ind_bench_str, ind_plain_str])
# inds_other = []
# ind_prev = 0
# for ind in inds_hite + [len(highlighted_teams_input_extras)]:
#     inds_other += range(ind_prev, ind)
#     ind_prev = ind + 1
# inds_hite += inds_other[::-1]
# highlighted_teams_input_extras = highlighted_teams_input_extras[inds_hite]
# st.session_state['highlighted_teams_extras'] = highlighted_teams_input_extras


# Find colour lists for plotting (saved to session state):
remove_old_colours_for_highlights(hb_teams_input)
choose_colours_for_highlights(hb_teams_input)

# highlighted_teams_input_extras.remove('-')


# ##################################
# ########## CALCULATIONS ##########
# ##################################

sorted_results = utilities.main_calculations.\
    predict_treatment(X, model, stroke_teams_list,
                      highlighted_teams_list, benchmark_rank_list,
                      hb_teams_list)

# Get indices of highest, most average, and lowest probability teams.
index_high = sorted_results.iloc[0]['Index']
index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
index_low = sorted_results.iloc[-1]['Index']
indices_high_mid_low = [index_high, index_mid, index_low]

# Get indices of highlighted teams:
# --- These ways are faster...
# (attempted a Pandas-esque way to do this
# but it looks worse than numpy where)
# indices_highlighted = sorted_results['Index'].loc[
#     ~sorted_results['Highlighted team'].str.contains('-')]
# indices_highlighted = sorted_results[
#     sorted_results['Highlighted team'].isin(highlighted_teams_input)]
# --- but this way retains the order that highlighted teams were added:
indices_highlighted = []
for team in hb_teams_input:
    if '-' not in team and 'Benchmark' not in team:
        ind_team = sorted_results['Index'][
            sorted_results['HB team'] == team].values[0]
        indices_highlighted.append(ind_team)

# Find Shapley values only for the important indices:
(shap_values_probability_extended_high_mid_low,
 shap_values_probability_high_mid_low) = \
    utilities.main_calculations.find_shapley_values(
        explainer_probability, X.iloc[indices_high_mid_low])

if len(indices_highlighted) > 0:
    (shap_values_probability_extended_highlighted,
     shap_values_probability_highlighted) = \
        utilities.main_calculations.find_shapley_values(
            explainer_probability, X.iloc[indices_highlighted])
else:
    shap_values_probability_extended_highlighted = None
    shap_values_probability_highlighted = None


# Make Shapley values for all indices:
(shap_values_probability_extended_all,
    shap_values_probability_all) = \
    utilities.main_calculations.find_shapley_values(
        explainer_probability, X)

# ###########################
# ######### RESULTS #########
# ###########################
# st.header('Results')

with container_metrics:
    # Print metrics for how many teams would thrombolyse:
    utilities.container_results.show_metrics_benchmarks(sorted_results)


# Draw a plot in this function:
utilities.container_results.main(
    sorted_results,
    shap_values_probability_extended_high_mid_low,
    shap_values_probability_extended_highlighted,
    indices_high_mid_low,
    indices_highlighted,
    headers_X,
    explainer_probability, X,
    shap_values_probability_extended_all,
    shap_values_probability_all
    )


# # TESTS - size of objects
# from sys import getsizeof
# things = [
#     user_inputs_dict,
#     X,
#     sorted_results,
#     stroke_teams_list,
#     shap_values_probability_extended_high_mid_low,
#     shap_values_probability_high_mid_low,
#     shap_values_probability_extended_highlighted,
#     shap_values_probability_highlighted,
#     benchmark_rank_list,
#     highlighted_teams_list,
#     indices_high_mid_low,
#     indices_highlighted,
#     headers_X,
#     headers_synthetic,
#     explainer_probability,
#     model,
#     explainer
# ]
#
#
# for thing in things:
#     st.write(type(thing), getsizeof(thing))
#     st.write(' ')


# ###########################
# ######### DETAILS #########
# ###########################
# st.write('-'*50)
# st.header('Details of the calculation')
# st.write('The following bits detail the calculation.')

# with st.expander('Some details'):
#     # Draw the equation in this function:
#     utilities.container_details.main(animal, feature, row_value)

# Garbage collection to reduce gradual memory creep
gc.collect()

# ----- The end! -----
