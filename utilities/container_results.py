"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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

    # x_chart = range(len(sorted_results))

    fig = px.bar(
        sorted_results,
        x='Sorted rank',
        y='Probability_perc',
        custom_data=['Stroke team']
        )

    # ax.axes.get_xaxis().set_ticks([])

    # Figure title:
    # fig.update_layout(title_text='Example text', title_x=0.5)
    # Change axis:
    fig.update_yaxes(range=[0.0, 100.0])
    # fig.update_xaxes(range=[0, time_list_yr[-1]],
    #                  constrain='domain')  # For aspect ratio.
    # Update ticks:
    fig.update_xaxes(tick0=0, dtick=10)
    # fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(tick0=0, dtick=25.0)
    # Set axis labels:
    fig.update_xaxes(title_text=
        f'Rank out of {sorted_results.shape[0]} stroke teams')
    fig.update_yaxes(
        title_text='Probability of giving patient thrombolysis (%)')

    # Hover settings:
    # Make it so cursor can hover over any x value to show the
    # label of the survival line for (x,y), rather than needing to
    # hover directly over the line:
    fig.update_layout(hovermode='x')
    # Show the probability with two decimal places:
    fig.update_traces(
        hovertemplate=(
            '%{customdata[0]}' +
            '<br>'
            '%{y:>.2f}%' +
            '<extra></extra>'
            )
        )

    # Add horizontal line at prob=0.5:
    # fig.add_hline(y=50.0)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def plot_sorted_probs_matplotlib(sorted_results):

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