"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st

from utilities.inputs import import_animal_data


def main():
    # ----- Import the animal data -----
    animals_df = import_animal_data(
        './data/animal_df.csv',
        index_col='Animal'
        )
    # Display the data:
    st.write('This data has been imported from a file: ')
    st.dataframe(animals_df)

    # ----- User selection parameters -----
    st.write('By changing the user input, we can pick a different value: ')

    animal = st.selectbox(
        'Pick an animal',
        ['Rats', 'Bats', 'Wombats', 'Cats']
        )

    feature = st.selectbox(
        'Pick a feature',
        ['Length', 'Height', 'Squeak', 'Girth']
        )

    # ----- Print the selected value. -----
    # First take the requested row.
    try:
        row_data = animals_df.loc[animal]
    except KeyError:
        # Write an error message:
        st.error(f':exclamation: {animal} are not in the data file.')
        # Stop the script.
        st.stop()

    # Then look for the requested feature.
    try:
        row_value = row_data[feature]
    except KeyError:
        # Write an error message:
        st.error(f':exclamation: {feature} is not in the data file.')
        # Stop the script.
        st.stop()

    # If the choice was valid, print the result:
    st.write(f'The {feature} of {animal} is {row_value:2d}.')

    return animals_df, animal, feature, row_value
