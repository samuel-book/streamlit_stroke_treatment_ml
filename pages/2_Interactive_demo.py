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
# st.header('Setup')
# st.subheader('Inputs')

with st.sidebar:
    st.markdown('# Patient details')
    user_inputs_dict = utilities.container_inputs.user_inputs()
    # Write an empty header to give breathing room at the bottom:
    st.markdown('# ')
stroke_teams_list = utilities.inputs.read_stroke_teams_from_file()

benchmark_df = utilities.inputs.import_benchmark_data()

# Pick favourites:
highlighted_teams_input = utilities.container_inputs.\
    highlighted_teams(stroke_teams_list)

# Build these into a 2D DataFrame:
synthetic = utilities.inputs.build_dataframe_from_inputs(
        user_inputs_dict, stroke_teams_list, highlighted_teams_input,
        benchmark_df)

# synthetic = utilities.inputs.import_patient_data()

# Make a copy of this data that is ready for the model.
# The same data except the Stroke Team column is one-hot-encoded.
X = utilities.inputs.one_hot_encode_data(synthetic)

# Store the column names:
headers_synthetic = synthetic.columns
headers_X = X.columns

# Load in the model and explainers separately so each can be cached:
model = utilities.inputs.load_pretrained_model()
explainer = utilities.inputs.load_explainer()
explainer_probability = utilities.inputs.load_explainer_probability()


# ##################################
# ########## CALCULATIONS ##########
# ##################################

sorted_results = utilities.main_calculations.\
    predict_treatment(X, synthetic, model)

# Get index of highest probability team
index_high = sorted_results.iloc[0]['Index']
index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
index_low = sorted_results.iloc[-1]['Index']
indices_high_mid_low = [index_high, index_mid, index_low]

# Get indices of highlighted teams:
# (attempted a Pandas-esque way to do this
# but it looks worse than numpy where)
indices_highlighted = sorted_results['Index'].loc[
    ~sorted_results['Highlighted team'].str.contains('-')]

# Find Shapley values:
shap_values_probability_extended, shap_values_probability = \
    utilities.main_calculations.find_shapley_values(explainer_probability, X)


# ###########################
# ######### RESULTS #########
# ###########################
st.header('Results')
# Draw a plot in this function:
utilities.container_results.main(
    sorted_results,
    shap_values_probability_extended,
    indices_high_mid_low,
    indices_highlighted,
    headers_X
    )


# ###########################
# ######### DETAILS #########
# ###########################
# st.write('-'*50)
# st.header('Details of the calculation')
# st.write('The following bits detail the calculation.')

# with st.expander('Some details'):
#     # Draw the equation in this function:
#     utilities.container_details.main(animal, feature, row_value)

# ----- The end! -----
