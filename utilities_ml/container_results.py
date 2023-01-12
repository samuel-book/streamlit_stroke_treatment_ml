"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import importlib
import pandas as pd
# import matplotlib

# For creating SHAP waterfall in response to click:
import utilities_ml.main_calculations
# For clickable plotly events:
from streamlit_plotly_events import plotly_events

# For matplotlib plots:
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# Import local package
from utilities_ml import waterfall
# Force package to be reloaded
importlib.reload(waterfall)


def main(sorted_results,
         shap_values_probability_extended_high_mid_low,
         shap_values_probability_extended_highlighted,
         indices_high_mid_low, indices_highlighted, headers_X,
         explainer_probability, X,
         shap_values_probability_extended_all,
         shap_values_probability_all
         ):

    pass


def plot_heat_grid_full(grid):
    vlim = np.max(np.abs(grid))
    fig = px.imshow(
        grid,
        color_continuous_scale='rdbu_r',
        range_color=(-vlim, vlim),
        aspect='auto'
        )
    fig.update_xaxes(showticklabels=False)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def plot_heat_grid_compressed(
        grid_cat_sorted, stroke_team_list, headers, stroke_team_2d
        ):
    vlim = np.abs(np.max(np.abs(grid_cat_sorted)))

    fig = px.imshow(
        grid_cat_sorted,
        x=np.arange(1, len(stroke_team_list)+1),
        y=headers,
        labels=dict(
            y='Feature',
            x=f'Rank out of {len(stroke_team_list)} stroke teams',
            color='Effect on probability (%)'
        ),
        color_continuous_scale='rdbu_r',
        range_color=(-vlim, vlim),
        aspect='auto'
        )

    # I don't understand why this step is necessary, but it is:
    stroke_team_cd = np.dstack((stroke_team_2d, stroke_team_2d))

    # Add this data to the figure:
    fig.update(data=[{'customdata': stroke_team_cd}])

    # Update the hover message with the stroke team:
    fig.update_traces(hovertemplate=(
        'Team %{customdata[0]}' +
        '<br>' +
        'Rank: %{x}' +
        '<br>' +
        'Feature: %{y}' +
        '<br>' +
        'Effect on probability: %{z:.2f}%'
        ))
    # fig.update_xaxes(showticklabels=False)

    xmax = grid_cat_sorted.shape[1]
    fig.update_xaxes(range=[0.0, xmax+1])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, grid_cat_sorted.shape[1], 10),
        # ticktext=['0%', '25%', 'Default value, 30%',
        #             '50%', '75%', '100%']
        ))

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def print_changes_info(grid_cat_sorted, headers, stroke_team_2d):
    # Print the biggest positive and negative changes:
    biggest_prob_change_pve = np.max(grid_cat_sorted)
    biggest_prob_change_nve = np.min(grid_cat_sorted)
    inds_biggest_prob_change_pve = np.where(
        grid_cat_sorted == biggest_prob_change_pve)
    inds_biggest_prob_change_nve = np.where(
        grid_cat_sorted == biggest_prob_change_nve)
    feature_biggest_prob_change_pve = headers[inds_biggest_prob_change_pve[0]]
    feature_biggest_prob_change_nve = headers[inds_biggest_prob_change_nve[0]]
    team_biggest_prob_change_pve = \
        stroke_team_2d[inds_biggest_prob_change_pve]
    team_biggest_prob_change_nve = \
        stroke_team_2d[inds_biggest_prob_change_nve]

    st.markdown(''.join([
        'The biggest shift upwards in probability is ',
        f'{biggest_prob_change_pve:.2f}% ',
        'from the feature ',
        f'{feature_biggest_prob_change_pve}',
        ' for the stroke team ',
        f'{team_biggest_prob_change_pve}',
        ' (rank ',
        f'{inds_biggest_prob_change_pve[1]+1}'
        ').'
    ]))

    st.markdown(''.join([
        'The biggest shift downwards in probability is ',
        f'{biggest_prob_change_nve:.2f}% ',
        'from the feature ',
        f'{feature_biggest_prob_change_nve}',
        ' for the stroke team ',
        f'{team_biggest_prob_change_nve}',
        ' (rank ',
        f'{inds_biggest_prob_change_nve[1]+1}'
        ').'
    ]))


def plot_all_prob_shifts_for_all_features_and_teams(
        headers, grid_cat_sorted
        ):
    """
    Line chart
    """
    fig = go.Figure()
    sorted_rank_arr = np.arange(1, grid_cat_sorted.shape[1]+1)
    for i, feature in enumerate(headers):
        fig.add_trace(go.Scatter(
            x=sorted_rank_arr,
            y=grid_cat_sorted[i, :],
            mode='markers',
            name=feature))

    # Update titles and labels:
    fig.update_layout(
        title='Effect on probability by feature',
        xaxis_title='Stroke team by rank',
        yaxis_title='Effect on probability (%)',
        legend_title='Feature'
        )

    # # When hovering, highlight all features' points for chosen x:
    # fig.update_layout(hovermode='x unified')
    fig.update_xaxes(showspikes=True, spikesnap='cursor', spikemode='across',
                     spikethickness=1)

    # Remove hover message:
    fig.update_traces(hovertemplate='<extra></extra>')

    # Move legend to bottom
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.2,
        xanchor="right",
        x=1
    ))

    # Make the figure taller:
    fig.update_layout(height=750)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def write_feature_means_stds(grid_cat_sorted, headers, inds=[], return_inds=False):

    # Round the values now to save mucking about with df formatting.
    std_list = np.std(grid_cat_sorted, axis=1)
    ave_list = np.median(grid_cat_sorted, axis=1)
    max_list = np.max(grid_cat_sorted, axis=1)
    min_list = np.min(grid_cat_sorted, axis=1)

    # Find each team where this is max?

    lists = [headers, ave_list, std_list, max_list, min_list]
    if len(inds) > 0:
        inds_std = inds
    else:
        # Sort from lowest to highest standard deviation:
        # (sorting in pandas also sorts the index column. This will look
        # confusing.)
        inds_std = np.argsort(std_list)
    for i, data in enumerate(lists):
        lists[i] = data[inds_std]

    # Make a dataframe of these values:
    lists = np.array(lists, dtype=object)
    data_for_df = np.transpose(np.vstack(lists))
    headers_for_df = [
            'Feature',
            'Median shift (%)',
            'Standard deviation of shift (%)',
            'Biggest shift (%)',
            'Smallest shift (%)'
            ]
    df = pd.DataFrame(
        data_for_df,
        columns=headers_for_df
    )

    # Write to streamlit with two decimal places:
    f = '{:.2f}'
    style_dict = {
        'Feature': None,
        'Median shift (%)': f,
        'Standard deviation of shift (%)': f,
        'Biggest shift (%)': f,
        'Smallest shift (%)': f
    }
    st.dataframe(df.style.format(style_dict))

    if return_inds == True:
        return inds_std

