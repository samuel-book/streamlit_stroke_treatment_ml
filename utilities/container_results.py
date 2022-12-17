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
importlib.reload(waterfall)


def main(sorted_results, shap_values_probability_extended,
         indices_high_mid_low, indices_favourites):
    plot_sorted_probs(sorted_results)

    with st.expander('SHAP for max, middle, min'):
        headers = [
            'Maximum probability',
            'Middle probability',
            'Minimum probability'
            ]
        for i_here, i in enumerate(indices_high_mid_low):
            title = (
            '### ' + headers[i_here] + ': Team ' +
            sorted_results['Stroke team'].loc[i]
            )
            st.markdown(title)
            # st.write(i, shap_values_probability_extended[i])
            plot_shap_waterfall(shap_values_probability_extended[i])

    if len(indices_favourites) > 0:
        with st.expander('SHAP for favourite teams'):
            for i in indices_favourites:
                title = '### Team ' + sorted_results['Stroke team'].loc[i]
                st.markdown(title)
                # st.write(i, shap_values_probability_extended[i])
                plot_shap_waterfall(shap_values_probability_extended[i])


def plot_sorted_probs(sorted_results):
    base_values = 0.2995270168908044

    # x_chart = range(len(sorted_results))

    fig = px.bar(
        sorted_results,
        x='Sorted rank',
        y='Probability_perc',
        custom_data=['Stroke team'],
        color='Favourite team'
        )

    # ax.axes.get_xaxis().set_ticks([])

    # Figure title:
    # fig.update_layout(title_text='Example text', title_x=0.5)
    # Change axis:
    fig.update_yaxes(range=[0.0, 100.0])

    # Update ticks:
    # fig.update_xaxes(tick0=0, dtick=10)

    xmax = sorted_results.shape[0]
    # If leaving a space at the end for text:
    # xmax = xmax*1.25

    fig.update_xaxes(range=[0.0, xmax+1])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, sorted_results.shape[0], 10),
        # ticktext=['0%', '25%', 'Default value, 30%',
        #             '50%', '75%', '100%']
        ))

    # fig.update_xaxes(showticklabels=False)
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 25, base_values*100.0, 50, 75, 100],
        ticktext=['0%', '25%', f'{base_values*100.0:.2f}%',
                    '50%', '75%', '100%']
        ))
    # fig.update_yaxes(tick0=0, dtick=25.0)
    # Set axis labels:
    fig.update_xaxes(title_text=
        f'Rank out of {sorted_results.shape[0]} stroke teams')
    fig.update_yaxes(
        title_text='Probability of giving patient thrombolysis')

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

    # Add horizontal line at prob=0.3, the SHAP starting point:
    fig.add_hline(y=base_values*100.0)#,
    #             annotation_text='Starting probability')
    # fig.add_hline(y=base_values*100.0,
    #               annotation_text=f'{base_values*100.0:.2f}%',
    #               annotation_position='bottom right')

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
    # # Access the axis limits with this: 
    # current_xlim = plt.xlim()
    # st.write(current_xlim)
    # # Update:
    # plt.xlim(-0.5, 1.5) # doesn't work fully as expected
    st.pyplot(fig)
    plt.close(fig)
