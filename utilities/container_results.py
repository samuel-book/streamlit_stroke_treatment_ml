"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import importlib
import pandas as pd

# For matplotlib plots:
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

from utilities.main_calculations import convert_explainer_01_to_noyes

# Import local package
from utilities import waterfall
# Force package to be reloaded
importlib.reload(waterfall)


def main(sorted_results, 
         shap_values_probability_extended_high_mid_low,
         shap_values_probability_extended_highlighted,
         indices_high_mid_low, indices_highlighted, headers_X,
        #  shap_values_probability
         ):
    show_metrics_benchmarks(sorted_results)

    plot_sorted_probs(sorted_results)

    with st.expander('SHAP for max, middle, min'):
        headers = [
            'Maximum probability',
            'Middle probability',
            'Minimum probability'
            ]
        for i_here, i in enumerate(indices_high_mid_low):
            # Find the data:
            sv = shap_values_probability_extended_high_mid_low[i_here]
            # Change integer 0/1 to str no/yes for display:
            sv_to_display = convert_explainer_01_to_noyes(sv)
            
            # Write to streamlit:
            title = (
                '## ' + headers[i_here] + ': \n ### Team ' +
                sorted_results['Stroke team'].loc[i]
            )
            sorted_rank = sorted_results['Sorted rank'].loc[i]
            # Final probability:
            final_prob = sorted_results['Probability'].loc[i]
            st.markdown(title)
            st.markdown(f'Rank: {sorted_rank} of {sorted_results.shape[0]}')
            # Plot:
            plot_shap_waterfall(sv_to_display, final_prob)

    # Optional extra expander for highlights:
    if len(indices_highlighted) > 0:
        with st.expander('SHAP for highlighted teams'):
            for i_here, i in enumerate(indices_highlighted):
                # Find the data:
                sv = shap_values_probability_extended_highlighted[i_here]
                # Change integer 0/1 to str no/yes for display:
                sv_to_display = convert_explainer_01_to_noyes(sv)
                # Final probability:
                final_prob = sorted_results['Probability'].loc[i]

                # Write to streamlit:
                title = '### Team ' + sorted_results['Stroke team'].loc[i]
                sorted_rank = sorted_results['Sorted rank'].loc[i]
                st.markdown(title)
                st.markdown(
                    f'Rank: {sorted_rank} of {sorted_results.shape[0]}')
                # Plot:
                plot_shap_waterfall(sv_to_display, final_prob)

    # if st.checkbox('Testing:'):
    #     st.markdown('# Testing below')
    #     plot_heat_grid(
    #         shap_values_probability_extended,
    #         headers_X, sorted_results['Stroke team'],
    #         sorted_results['Index'], shap_values_probability
    #         )


def plot_sorted_probs(sorted_results):
    base_values = 0.2995270168908044

    # sorted_results = sorted_results_orig.copy(deep=True)

    # x_chart = range(len(sorted_results))

    fig = px.bar(
        sorted_results,
        x='Sorted rank',
        y='Probability_perc',
        custom_data=['Stroke team', 'Thrombolyse_str', 'Benchmark'],
        color='Highlighted team',
        text='Benchmark',
        # color_discrete_map= {'Benchmark': 'yellow'}
        )

    # Update text at top of bar chart:
    fig.update_traces(
        textfont_size=20,
        textposition='outside',
        cliponaxis=False,
        # marker_line=dict(width=2, color='black')
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
    # fig.update_layout(yaxis=dict(
    #     tickmode='array',
    #     tickvals=[0, 25, base_values*100.0, 50, 75, 100],
    #     ticktext=['0%', '25%', f'{base_values*100.0:.2f}%',
    #                 '50%', '75%', '100%']
    #     ))
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
            '<br>' +
            'Thrombolysis: %{customdata[1]}' +
            '<br>' + 
            '%{customdata[2]}'
            '<extra></extra>'
            )
        )

    # Add horizontal line at prob=0.5, the decision to thrombolyse:
    fig.add_hline(y=50.0)
    # # Add horizontal line at prob=0.3, the SHAP starting point:
    # fig.add_hline(y=base_values*100.0)#,
    # #             annotation_text='Starting probability')
    # # fig.add_hline(y=base_values*100.0,
    # #               annotation_text=f'{base_values*100.0:.2f}%',
    # #               annotation_position='bottom right')

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)
    st.write(''.join([
        'The line at 50% is the cut-off for thrombolysis. ',
        'Stroke teams with a probability below the line will not ',
        'thrombolyse the patient, and teams on or above the line will.'
        ]))
    st.write(''.join([
        'Currently benchmark teams are marked with the world\'s tiniest ',
        'stars, but this will be changed to something easier to see.'
        ]))


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


def plot_shap_waterfall_matplotlib(shap_values):
    with _lock:
        fig = waterfall.waterfall(
            shap_values,
            show=False, max_display=10, y_reverse=True, rank_absolute=False
            )
        # # Access the axis limits with this:
        # current_xlim = plt.xlim()
        # st.write(current_xlim)
        # # Update:
        # plt.xlim(-0.5, 1.5) # doesn't work fully as expected

        st.pyplot(fig)

        plt.close(fig)
        # Delete the figure - attempt to help memory usage:
        # del fig


def plot_shap_waterfall(shap_values, final_prob, n_to_show=9):

    # Make lists of all of the features:
    shap_probs = shap_values.values
    feature_names = np.array(shap_values.feature_names)

    # Collect the input patient data:
    patient_data = shap_values.data
    # Adjust some values to add units for showing on the plot:
    extra_bits = {
        'Onset-to-arrival time':' minutes',
        'Arrival-to-scan time':' minutes',
        'Stroke severity':' (out of 42)',
        'Prior disability level':' (mRS)',
        'Age':' years',
    }
    for i_here, feature in enumerate(extra_bits.keys()):
        i = np.where(feature_names == feature)[0][0]
        patient_data[i] = f'{patient_data[i]}' + extra_bits[feature]

    base_values = 0.2995270168908044
    base_values_perc = base_values * 100.0

    final_prob_perc = final_prob * 100.0


    # Sort by decreasing probability (magnitude, not absolute):
    inds = np.argsort(np.abs(shap_probs))

    # Show these ones individually:
    shap_probs_to_show = shap_probs[inds][-n_to_show:]
    feature_names_to_show = feature_names[inds][-n_to_show:]
    patient_data_to_show = patient_data[inds][-n_to_show:]

    # Merge these into one bar:
    shap_probs_to_hide = shap_probs[inds][:-n_to_show]
    shap_probs_hide_sum = np.sum(shap_probs_to_hide)
    n_features_hidden = len(feature_names) - len(feature_names_to_show)
    feature_name_hidden = f'{n_features_hidden} other features'
    data_name_hidden = 'N/A'

    # Add the merged hidden features to the main lists:
    n_to_show += 1
    shap_probs_to_show = np.append(shap_probs_hide_sum, shap_probs_to_show)
    feature_names_to_show = np.append(feature_name_hidden, feature_names_to_show)
    patient_data_to_show = np.append(data_name_hidden, patient_data_to_show)

    # Save a copy of shap probabilities in terms of percentage:
    shap_probs_perc = shap_probs_to_show * 100.0

    # "measures" list says whether each step in the waterfall is a
    # shift or a new total.
    measures = ['relative'] * (len(shap_probs_to_show))

    fig = go.Figure(go.Waterfall(
        orientation='h',  # horizontal
        measure=measures,
        y=feature_names_to_show,
        x=shap_probs_perc,
        base=base_values_perc,
        decreasing={'marker':{'color':'#008bfa'}},
        increasing={'marker':{'color':'#ff0050'}}
    ))


    # For some reason, custom_data needs to be columns rather than rows:
    custom_data = np.stack( (shap_probs_perc, patient_data_to_show), axis=-1)
    # Add the custom data:
    fig.update_traces(customdata=custom_data, selector=dict(type='waterfall'))

    # Flip y-axis so bars are read from top to bottom.
    fig['layout']['yaxis']['autorange'] = 'reversed'

    # When hovering, show bar at this y value:
    fig.update_layout(hovermode='y')



    # Update the hover message with the stroke team:
    fig.update_traces(hovertemplate=(
        # 'Team %{customdata[0]}' +
        # '<br>' +
        # 'Rank: %{x}' +
        # '<br>' +
        'Feature value: %{customdata[1]}' +
        '<br>' +
        'Effect on probability: %{customdata[0]:.2f}%' +
        # Need the following line to remove default "trace" bit:
        '<extra></extra>'
        ))

    # Write the size of each bar within the bar:
    fig.update_traces(text=np.round(shap_probs_perc,2), selector=dict(type='waterfall'))
    fig.update_traces(textposition='inside', selector=dict(type='waterfall'))
    fig.update_traces(texttemplate='%{text:+}%', selector=dict(type='waterfall'))

    # Set axis labels:
    fig.update_xaxes(title_text=' <br>Probability of thrombolysis (%)')
    fig.update_yaxes(title_text='Feature')
    # fig.update_layout(title='Team name')

    # Add start and end prob annotations:
    fig.add_annotation(x=base_values_perc, y=-0.4,
                text=f'Start probability: {base_values_perc:.2f}%',
                showarrow=True,
                yshift=1,
                ax=0  # Make arrow vertical - a = arrow, x = x-shift.
    )
    fig.add_annotation(x=final_prob_perc, y=n_to_show-0.6,
                text=' <br>'+f'End probability: {final_prob_perc:.2f}%',
                showarrow=True,
                # yshift=-100,
                ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
                ay=35,  # Make the arrow sit below the final bar
    )

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


def show_metrics_benchmarks(sorted_results):
    # Benchmark teams:
    # sorted_results['Benchmark rank']

    inds_benchmark = sorted_results['Benchmark rank'] <= 30

    results_all = sorted_results
    results_benchmark = sorted_results.loc[inds_benchmark]
    results_non_benchmark = sorted_results.loc[~inds_benchmark]

    # Number of entries that would thrombolyse:
    n_thrombolyse_all = results_all.Thrombolyse.sum()
    n_thrombolyse_benchmark = results_benchmark.Thrombolyse.sum()
    n_thrombolyse_non_benchmark = results_non_benchmark.Thrombolyse.sum()

    # Total number of entries:
    n_all = len(results_all)
    n_benchmark = len(results_benchmark)
    n_non_benchmark = len(results_non_benchmark)

    # Percentage of entries that would thrombolyse:
    perc_thrombolyse_all = 100.0 * n_thrombolyse_all / n_all
    perc_thrombolyse_benchmark = 100.0 * n_thrombolyse_benchmark / n_benchmark
    perc_thrombolyse_non_benchmark = 100.0 * n_thrombolyse_non_benchmark / n_non_benchmark

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            'All stroke teams',
            f'{perc_thrombolyse_all:.0f}%'
            )
        st.write(f'{n_thrombolyse_all} of {n_all} stroke teams would thrombolyse.')

    with cols[1]:
        st.metric(
            'Benchmark stroke teams',
            f'{perc_thrombolyse_benchmark:.0f}%'
            )
        st.write(f'{n_thrombolyse_benchmark} of {n_benchmark} stroke teams would thrombolyse.')

    with cols[2]:
        st.metric(
            'Non-benchmark stroke teams',
            f'{perc_thrombolyse_non_benchmark:.0f}%'
            )
        st.write(f'{n_thrombolyse_non_benchmark} of {n_non_benchmark} stroke teams would thrombolyse.')

    # Write benchmark decision:
    extra_str = '' if perc_thrombolyse_benchmark >= 50.0 else 'do not '
    st.markdown('__Benchmark decision:__ ' + extra_str + 'thrombolyse this patient.')


def plot_heat_grid(shap_values_probability_extended, headers,
                   stroke_team_list, sorted_inds, shap_values_probability):
    # Experiment
    n_teams = len(shap_values_probability_extended)
    n_features = len(shap_values_probability_extended[0].values)
    # grid = np.zeros((n_features, n_teams))

    # # Don't fill this grid in the same order as the sorted bar chart.
    # # Rely on picking out the diagonal later.
    # for i, team in enumerate(shap_values_probability_extended):
    #     values = shap_values_probability_extended[i].values
    #     grid[:, i] = values

    grid = np.transpose(shap_values_probability)


    vlim = np.max(np.abs(grid))
    fig = px.imshow(
        grid,
        # x=stroke_team_list,
        # y=headers,
        # labels=dict(
            # x='Feature',
            # y='Stroke team',
            # color='Effect on probability (%)'
        # ),
        color_continuous_scale='rdbu_r',
        range_color=(-vlim, vlim),
        aspect='auto'
        )
    fig.update_xaxes(showticklabels=False)

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)


    # Expect most of the mismatched one-hot-encoded hospitals to make
    # only a tiny contribution to the SHAP. Moosh them down into one
    # column instead.

    # Have 9 features other than teams. Index 9 is the first team.
    ind_first_team = 9

    # Make a new grid and copy over most of the values:
    grid_cat = np.zeros((ind_first_team + 1, n_teams))
    grid_cat[:ind_first_team, :] = grid[:ind_first_team, :]

    # For the remaining column, loop over to pick out the value:
    for i, sorted_ind in enumerate(sorted_inds):
        row = i + ind_first_team
        grid_cat[ind_first_team, i] = grid[row, i]

    # Multiply values by 100 to get probability in percent:
    grid_cat *= 100.0

    # Sort the values into the same order as sorted_results:
    grid_cat_sorted = grid_cat[:, sorted_inds]

    vlim = np.abs(np.max(np.abs(grid_cat)))

    headers = np.append(headers[:9], 'Stroke Team')

    # df_cat = pd.DataFrame(
    #     grid_cat.T,
    #     columns=headers
    # )


    # fig, ax = plt.subplots()
    # grid_map = ax.imshow(grid_cat, vmin=-vlim, vmax=vlim, cmap='RdBu_r')
    # plt.colorbar(grid_map)

    # ax.set_xlabel('Team')
    # ax.set_ylabel('Feature')

    # st.pyplot(fig)

    # headers = headers[:ind_first_team]

    # fig = px.imshow(grid_cat, color_continuous_scale='rdbu_r')
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

    # 2D grid of stroke_teams:
    stroke_team_2d = np.tile(
        stroke_team_list, len(headers)).reshape(grid_cat_sorted.shape)
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


    # Print the biggest positive and negative changes:
    biggest_prob_change_pve = np.max(grid_cat_sorted)
    biggest_prob_change_nve = np.min(grid_cat_sorted)
    inds_biggest_prob_change_pve = np.where(grid_cat_sorted==biggest_prob_change_pve)
    inds_biggest_prob_change_nve = np.where(grid_cat_sorted==biggest_prob_change_nve)
    feature_biggest_prob_change_pve = headers[inds_biggest_prob_change_pve[0]]
    feature_biggest_prob_change_nve = headers[inds_biggest_prob_change_nve[0]]
    team_biggest_prob_change_pve = stroke_team_2d[inds_biggest_prob_change_pve]
    team_biggest_prob_change_nve = stroke_team_2d[inds_biggest_prob_change_nve]

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


    # Line chart
    # (definitely move to its own function after testing done)
    # for row in range(grid_cat_sorted.shape[0]):
    df_cat = pd.DataFrame(
        grid_cat_sorted.T,
        columns=headers
    )
    # Add index column:
    df_cat['Sorted rank'] = np.arange(1, len(df_cat)+1)

    # st.write(df_cat)
    # fig = px.line(
    #     df_cat,
    #     x='Sorted rank',
    #     y='Arrival-to-scan time',
    #     color='Stroke team',
    #     # line_dash = 'continent'
        # )

    import plotly.graph_objects as go
    fig = go.Figure()
    sorted_rank_arr = np.arange(1, len(df_cat)+1)
    for i, feature in enumerate(headers):
        fig.add_trace(go.Scatter(x=sorted_rank_arr, y=grid_cat_sorted[i, :],
                            mode='lines',
                            name=feature))


    fig.update_layout(
        title='Effect on probability by feature',
        xaxis_title='Stroke team by rank',
        yaxis_title='Effect on probability (%)',
        legend_title='Feature'
        )

    # # When hovering, highlight all features' points for chosen x:
    # fig.update_layout(hovermode='x unified')

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
    # Changing width in the same way doesn't work when we write to
    # streamlit later with use_container_width=True.
    # Set aspect ratio:
    # fig.update_yaxes(
    #     scaleanchor='x',
    #     scaleratio=2.0,
    #     constrain='domain'
    # )

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)
