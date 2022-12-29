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


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Title:
st.markdown('# Interactive demo')
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
benchmark_rank_list = tuple(benchmark_df.sort_values('stroke_team')['Rank'])

# ----- Highlighted teams -----
# Pick teams to highlight on the bar chart:
highlighted_teams_input = utilities.container_inputs.\
    highlighted_teams(stroke_teams_list)
# Update the "Highlighted teams" column:
# Label benchmarks:
# table[np.where(table[:, 7] <= 30), 6] = 'Benchmark'
highlighted_teams_list = np.array(
    ['-' for team in stroke_teams_list], dtype=object)
# Put in selected Highlighteds (overwrites benchmarks):
for team in highlighted_teams_input:
    ind_t = np.where(stroke_teams_list == team)
    highlighted_teams_list[ind_t] = team
highlighted_teams_list = tuple(highlighted_teams_list)
# # If just need yes/no, the following works.
# # It doesn't return indices in the same order as Highlighted_teams.
# bool_Highlighteds = np.in1d(stroke_teams_list, Highlighted_teams)
# table[:, 6][bool_Highlighteds] = 'Yes'


# ##################################
# ########## CALCULATIONS ##########
# ##################################

sorted_results = utilities.main_calculations.\
    predict_treatment(X, model, stroke_teams_list,
                      highlighted_teams_list, benchmark_rank_list)

# Get indices of highest, most average, and lowest probability teams.
index_high = sorted_results.iloc[0]['Index']
index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
index_low = sorted_results.iloc[-1]['Index']
indices_high_mid_low = [index_high, index_mid, index_low]

# Get indices of highlighted teams:
# (attempted a Pandas-esque way to do this
# but it looks worse than numpy where)
indices_highlighted = sorted_results['Index'].loc[
    ~sorted_results['Highlighted team'].str.contains('-')]

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


# ###########################
# ######### RESULTS #########
# ###########################
st.header('Results')
# Draw a plot in this function:
utilities.container_results.main(
    sorted_results,
    shap_values_probability_extended_high_mid_low,
    shap_values_probability_extended_highlighted,
    indices_high_mid_low,
    indices_highlighted,
    headers_X,
    explainer_probability, X
    # shap_values_probability
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
