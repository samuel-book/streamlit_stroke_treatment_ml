"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Import local package
from utilities import waterfall
# Force package to be reloaded
importlib.reload(waterfall);


def main(sorted_results, shap_values_probability_extended,
         indices_to_show):
    plot_sorted_probs(sorted_results)
    for i in indices_to_show:
        plot_shap_waterfall(shap_values_probability_extended[i])


def plot_sorted_probs(sorted_results):

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    x_chart = range(len(sorted_results))
    ax.bar(x_chart, sorted_results['Probability'], width=0.5)
    ax.plot([0, len(sorted_results)], [0.5, 0.5], c='k')
    ax.axes.get_xaxis().set_ticks([])
    ax.set_xlabel('Stroke team')
    ax.set_ylabel('Probability of giving patient thrombolysis')

    ax.set_ylim(0, 1)
    st.pyplot(fig)
    plt.close(fig)


def plot_shap_waterfall(shap_values):
    fig = waterfall.waterfall(
        shap_values,
        show=False, max_display=10, y_reverse=True
        )
    st.pyplot(fig)
    plt.close(fig)