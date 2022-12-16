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
from utilities.inputs import \
    write_text_from_file
# Containers:
import utilities.container_inputs
import utilities.container_results
import utilities.container_details


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Title:
st.markdown('# Interactive demo')
# Draw a blue information box:
st.info(
    ':information_source: ' +
    'For acronym reference, see the introduction page.'
    )
# Intro text:
write_text_from_file('pages/text_for_pages/2_Intro_for_demo.txt',
                     head_lines_to_skip=2)


# ###########################
# ########## SETUP ##########
# ###########################
st.header('Setup')
st.subheader('Inputs')

# Draw the input selection boxes in this function:
animals_df, animal, feature, row_value = utilities.container_inputs.main()


# ###########################
# ######### RESULTS #########
# ###########################
st.header('Results')
# Draw a plot in this function:
utilities.container_results.main(row_value, animal, feature)


# ###########################
# ######### DETAILS #########
# ###########################
st.write('-'*50)
st.header('Details of the calculation')
st.write('The following bits detail the calculation.')

with st.expander('Some details'):
    # Draw the equation in this function:
    utilities.container_details.main(animal, feature, row_value)

# ----- The end! -----
