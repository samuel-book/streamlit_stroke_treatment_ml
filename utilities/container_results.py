"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import some fixed parameters from our file:
from utilities.fixed_params import x_min, x_max, colours_plot


def main(row_value, animal, feature):
    st.write('We will draw a simple plot using our chosen value.')

    # Define the arrays to plot:
    # The x_min and x_max values are defined in fixed_params
    # so that they can be reused in multiple places.
    x_values = np.arange(x_min, x_max)
    y_values = (x_values/row_value)**2.0

    # Make the plot using matplotlib:
    fig, ax = plt.subplots()
    # The colours_plot list is also defined in fixed_params.
    ax.plot(x_values, y_values, color=colours_plot[0])
    ax.set_ylim(-5, 90)
    ax.set_title(f'{animal} {feature}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Call to streamlit:
    st.pyplot(fig)
