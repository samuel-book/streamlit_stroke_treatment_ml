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

# For clickable plotly events:
from streamlit_plotly_events import plotly_events

# For matplotlib plots:
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# For creating SHAP waterfall in response to click:
import utilities.main_calculations

# Import local package
from utilities import waterfall
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
    show_metrics_benchmarks(sorted_results)

    st.write(''.join([
        'The line at 50% is the cut-off for thrombolysis. ',
        'Stroke teams with a probability below the line will not ',
        'thrombolyse the patient, and teams on or above the line will.'
        ]))
    st.write(''.join([
        'Currently benchmark teams are marked with the world\'s tiniest ',
        'stars, but this will be changed to something easier to see.'
        ]))
    selected_bar = plot_sorted_probs(sorted_results)

    st.markdown('__New:__ click on a bar and get its waterfall plot below. This will probably be removed because it is slow and ruins the colours.')

    try:
        rank_selected = selected_bar[0]['x']
        prob_selected = selected_bar[0]['y']
        # Find where this bar is in the results dataframe:
        ind_selected = sorted_results['Index'].loc[sorted_results['Sorted rank'] == rank_selected].values[0]
        # Pull out some useful bits:
        # Find Shapley values:
        (shap_values_probability_extended_selected,
         shap_values_probability_selected) = \
            utilities.main_calculations.find_shapley_values(
                explainer_probability, X.iloc[[ind_selected, ind_selected]])

        # Find the data:
        sv = shap_values_probability_extended_selected[0]
        # Change integer 0/1 to str no/yes for display:
        sv_to_display = utilities.main_calculations.convert_explainer_01_to_noyes(sv)

        # Final probability:
        final_prob = sorted_results['Probability'].loc[ind_selected]

        # Write to streamlit:
        title = '### Team ' + sorted_results['Stroke team'].loc[ind_selected]
        st.markdown(title)
        st.markdown(
            f'Rank: {rank_selected} of {sorted_results.shape[0]}')
        # Plot:
        plot_shap_waterfall(sv_to_display, final_prob)

    except IndexError:
        # Do nothing if no bar is selected.
        pass

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
            sv_to_display = utilities.main_calculations.convert_explainer_01_to_noyes(sv)

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
                sv_to_display = utilities.main_calculations.\
                    convert_explainer_01_to_noyes(sv)
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

    st.write(''.join([
        'Calculating all of the waterfall plots takes a few seconds '
        'once this box is ticked: '
        ]))
    if st.checkbox('Combine all waterfalls'):
        st.markdown('## All waterfalls')

        grid, grid_cat_sorted, stroke_team_2d, headers = make_heat_grids(
            headers_X, sorted_results['Stroke team'], sorted_results['Index'],
            shap_values_probability_all)

        # plot_heat_grid_full(grid)
        # plot_heat_grid_compressed(
        #     grid_cat_sorted, sorted_results['Stroke team'], headers,
        #     stroke_team_2d)
        print_changes_info(grid_cat_sorted, headers, stroke_team_2d)

        # plot_all_prob_shifts_for_all_features_and_teams(
        #     headers, grid_cat_sorted)

        df_waterfalls = make_waterfall_df(
                grid_cat_sorted,
                headers,
                sorted_results['Stroke team'],
                sorted_results['Highlighted team'],
                base_values=0.2995270168908044
                )
        plot_combo_waterfalls(df_waterfalls, sorted_results['Stroke team'])

        # Write statistics:
        write_feature_means_stds(grid_cat_sorted, headers)
    else:
        pass


def plot_sorted_probs(sorted_results):

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
    # # Non-interactive version:
    # st.plotly_chart(fig, use_container_width=True)
    # Clickable version:
    selected_bar = plotly_events(fig, click_event=True)
    # selected_bars.append(selected_bar)

    # # try:
    # if 1==1:
    #     rank_selected = selected_bar[0]['x']
    #     prob_selected = selected_bar[0]['y']
    #     all_bars = fig.data[0]
    #     st.write(all_bars)
    #     c = list(all_bars.marker.color)
    #     i = rank_selected - 1
    #     c[i] = 'white'
    #     with fig.batch_update():
    #         all_bars.marker.color = c

    # except IndexError:
    #     # Don't change anything.
    #     pass

    return selected_bar


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
        'Onset-to-arrival time': ' minutes',
        'Arrival-to-scan time': ' minutes',
        'Stroke severity': ' (out of 42)',
        'Prior disability level': ' (mRS)',
        'Age': ' years',
    }
    for feature in extra_bits.keys():
        i = np.where(feature_names == feature)[0][0]
        patient_data[i] = f'{patient_data[i]}' + extra_bits[feature]

    # Start probability:
    base_values = shap_values.base_values
    base_values_perc = base_values * 100.0
    # End probability:
    final_prob_perc = final_prob * 100.0

    def sort_lists(
            shap_probs, feature_names, patient_data, n_to_show,
            sort_by_magnitude=True, merged_cell_loc='top'
            ):
        """
        sort_by_magnitude =
            True  - in order from smallest magnitude to largest
                    magnitude (mix -ve and +ve).
            False - in order from smallest to largest.
        merged_cell_at_top =
            True  - merged row at the top.
            False - merged row before first positive row.
            (This only makes sense when sort_by_magnitude=False).
        """
        # Sort by increasing probability (magnitude, not absolute):
        inds = np.argsort(np.abs(shap_probs))

        # Show these ones individually:
        shap_probs_to_show = shap_probs[inds][-n_to_show:]
        feature_names_to_show = feature_names[inds][-n_to_show:]
        patient_data_to_show = patient_data[inds][-n_to_show:]

        if sort_by_magnitude == False:
            # Sort again to put into increasing probability
            # (absolute, not magnitude):
            inds_abs = np.argsort(shap_probs_to_show)
            # Show these ones individually:
            shap_probs_to_show = shap_probs_to_show[inds_abs]
            feature_names_to_show = feature_names_to_show[inds_abs]
            patient_data_to_show = patient_data_to_show[inds_abs]

        # Merge these into one bar:
        shap_probs_to_hide = shap_probs[inds][:-n_to_show]
        shap_probs_hide_sum = np.sum(shap_probs_to_hide)
        n_features_hidden = len(feature_names) - len(feature_names_to_show)
        feature_name_hidden = f'{n_features_hidden} other features'
        data_name_hidden = 'N/A'

        if merged_cell_loc == 'top' or merged_cell_loc == 'ordered':
            # Add the merged hidden features to the main lists:
            shap_probs_to_show = np.append(
                shap_probs_hide_sum, shap_probs_to_show)
            feature_names_to_show = np.append(
                feature_name_hidden, feature_names_to_show)
            patient_data_to_show = np.append(
                data_name_hidden, patient_data_to_show)
            if merged_cell_loc == 'ordered':
                # Put the merged cell in size order.
                if sort_by_magnitude == False:
                    # Sort by increasing probability (magnitude, not absolute):
                    inds = np.argsort(shap_probs_to_show)
                else:
                    # Sort by increasing probability (magnitude, not absolute):
                    inds = np.argsort(np.abs(shap_probs_to_show))
                # Show these ones individually:
                shap_probs_to_show = shap_probs_to_show[inds]
                feature_names_to_show = feature_names_to_show[inds]
                patient_data_to_show = patient_data_to_show[inds]

        else:
            # Find where the first positive change in probability is:
            ind_pos = np.where(shap_probs_to_show > 0)[0]
            try:
                # Insert the merged cell data just before the first
                # positive bar.
                shap_probs_to_show = np.insert(
                    shap_probs_to_show, ind_pos[0], shap_probs_hide_sum)
                feature_names_to_show = np.insert(
                    feature_names_to_show, ind_pos[0], feature_name_hidden)
                patient_data_to_show = np.insert(
                    patient_data_to_show, ind_pos[0], data_name_hidden)
            except IndexError:
                # No positive values in the list. Whack this at the end.
                # Add the merged hidden features to the main lists:
                shap_probs_to_show = np.append(
                    shap_probs_to_show, shap_probs_hide_sum)
                feature_names_to_show = np.append(
                    feature_names_to_show, feature_name_hidden)
                patient_data_to_show = np.append(
                    patient_data_to_show, data_name_hidden)

        return shap_probs_to_show, feature_names_to_show, patient_data_to_show

    # Use the function to sort the displayed rows:
    shap_probs_to_show, feature_names_to_show, patient_data_to_show = \
        sort_lists(
            shap_probs, feature_names, patient_data, n_to_show,
            sort_by_magnitude=False, merged_cell_loc='ordered'
            )

    # Add one to n_to_show to account for the merged "all other features" row.
    n_to_show += 1

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



    # Change axis:
    # fig.update_xaxes(range=[0, 100])

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


    # # Set aspect ratio:
    # fig.update_yaxes(
    #     scaleanchor='x',
    #     scaleratio=4.0,
    #     # constrain='domain'
    # )
    # Make the figure taller:
    # fig.update_layout(height=750, width=750)

    # fig.update_scenes(aspectratio_x=4.0)

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


def make_heat_grids(headers, stroke_team_list, sorted_inds,
                    shap_values_probability):
    # Experiment
    n_teams = shap_values_probability.shape[0]
    # n_features = len(shap_values_probability_extended[0].values)
    grid = np.transpose(shap_values_probability)

    # Expect most of the mismatched one-hot-encoded hospitals to make
    # only a tiny contribution to the SHAP. Moosh them down into one
    # column instead.

    # Have 9 features other than teams. Index 9 is the first team.
    ind_first_team = 9

    # Make a new grid and copy over most of the values:
    grid_cat = np.zeros((ind_first_team + 2, n_teams))
    grid_cat[:ind_first_team, :] = grid[:ind_first_team, :]

    # For the remaining column, loop over to pick out the value:
    for i, sorted_ind in enumerate(sorted_inds):
        row = i + ind_first_team
        # Pick out the value we want:
        value_of_matching_stroke_team = grid[row, i]
        # Add the wanted value to the new grid:
        grid_cat[ind_first_team, i] = value_of_matching_stroke_team
        # Take the sum of all of the team values:
        value_of_merged_stroke_teams = np.sum(grid[ind_first_team:, i])
        # Subtract the value we want:
        value_of_merged_stroke_teams -= value_of_matching_stroke_team
        # And store this as a merged "all other teams" value:
        grid_cat[ind_first_team+1, i] = value_of_merged_stroke_teams

    # Multiply values by 100 to get probability in percent:
    grid_cat *= 100.0

    # Sort the values into the same order as sorted_results:
    grid_cat_sorted = grid_cat[:, sorted_inds]

    headers = np.append(headers[:9], 'This stroke team')
    headers = np.append(headers, 'Other stroke teams')


    # 2D grid of stroke_teams:
    stroke_team_2d = np.tile(
        stroke_team_list, len(headers)).reshape(grid_cat_sorted.shape)
    return grid, grid_cat_sorted, stroke_team_2d, headers


def plot_heat_grid_full(grid):
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


def plot_heat_grid_compressed(grid_cat_sorted, stroke_team_list, headers, stroke_team_2d):
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


def write_feature_means_stds(grid_cat_sorted, headers):  #, stroke_team_list):

    # Round the values now to save mucking about with df formatting.
    std_list = np.std(grid_cat_sorted, axis=1)
    ave_list = np.median(grid_cat_sorted, axis=1)
    max_list = np.max(grid_cat_sorted, axis=1)
    min_list = np.min(grid_cat_sorted, axis=1)

    # Find each team where this is max?

    lists = [headers, ave_list, std_list, max_list, min_list]
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
        'Feature':None,
        'Median shift (%)':f,
        'Standard deviation of shift (%)':f,
        'Biggest shift (%)':f,
        'Smallest shift (%)':f
    }
    st.dataframe(df.style.format(style_dict))


def make_waterfall_df(
        grid_cat_sorted, headers, stroke_team_list, highlighted_team_list,
        base_values=0.2995270168908044
        ):
    base_values_perc = 100.0 * base_values

    grid_waterfall = np.copy(grid_cat_sorted)
    # Sort the grid in order of increasing standard deviation:
    inds_std = np.argsort(np.std(grid_waterfall, axis=1))
    grid_waterfall = grid_waterfall[inds_std, :]
    features_waterfall = headers[inds_std]

    # Add a row for the starting probability:
    grid_waterfall = np.vstack(
        (np.zeros(grid_waterfall.shape[1]), grid_waterfall))
    # Make a cumulative probability line for each team:
    grid_waterfall_cumsum = np.cumsum(grid_waterfall, axis=0)
    # Add the starting probability to all values:
    grid_waterfall_cumsum += base_values_perc
    # Keep final probabilities separate:
    final_probs_list = grid_waterfall_cumsum[-1, :]
    # Feature names:
    features_waterfall = np.append('Base probability', features_waterfall)
    # features_waterfall = np.append(features_waterfall, 'Final probability')

    # Get the grid into a better format for the data frame:
    # Column containing shifts in probabilities for each feature:
    column_probs_shifts = grid_waterfall.T.ravel()
    # Column containing cumulative probabilities (for x axis):
    column_probs_cum = grid_waterfall_cumsum.T.ravel() 
    # Column of feature names:
    column_features = np.tile(features_waterfall, len(stroke_team_list))
    # Column of the rank:
    a = np.arange(1, len(stroke_team_list)+1)
    column_sorted_rank = np.tile(a, len(features_waterfall))\
        .reshape(len(features_waterfall), len(a)).T.ravel()
    # Column of stroke teams:
    column_stroke_team = np.tile(stroke_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(stroke_team_list)).T.ravel()
    # Column of highlighted teams:
    column_highlighted_teams = np.tile(highlighted_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(highlighted_team_list)).T.ravel()
    # Column of final probability of thrombolysis:
    column_probs_final = np.tile(final_probs_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(final_probs_list)).T.ravel()

    # Put this into a data frame:
    df_waterfalls = pd.DataFrame()
    df_waterfalls['Sorted rank'] = column_sorted_rank
    df_waterfalls['Stroke team'] = column_stroke_team
    df_waterfalls['Probabilities'] = column_probs_cum
    df_waterfalls['Prob shift'] = column_probs_shifts
    df_waterfalls['Prob final'] = column_probs_final
    df_waterfalls['Features'] = column_features
    df_waterfalls['Highlighted team'] = column_highlighted_teams
    return df_waterfalls


def plot_combo_waterfalls(df_waterfalls, stroke_team_list):
    fig = go.Figure()


    drawn_blank_legend_line = 0
    # Keep a list of the added data in order of zorder:
    fig_data_list = []
    for i, team in enumerate(stroke_team_list):
        df_team = df_waterfalls[df_waterfalls['Stroke team'] == team]
        if team == df_team['Highlighted team'].iloc[0]:
            leggy = True
            colour = None
            opacity = 1.0
            width = 2.0
        else:
            if drawn_blank_legend_line > 0:
                leggy = False
            else:
                leggy = True
                drawn_blank_legend_line += 1
                # name = 
            colour = 'grey'
            opacity = 0.25
            width = 1.0
            
        fig.add_trace(go.Scatter(
            x=df_team['Probabilities'],
            y=df_team['Features'],
            mode='lines+markers',
            line=dict(width=width, color=colour), opacity=opacity,
            # c=df_waterfalls['Stroke team'],
            customdata=np.stack((
                df_team['Stroke team'], 
                df_team['Prob shift'],
                df_team['Prob final']
                ), axis=-1),
            name=df_team['Highlighted team'].iloc[0],
            showlegend=leggy
            ))

        # Update data list to put highlights with highest zorder:
        if team == df_team['Highlighted team'].iloc[0]:
            fig_data_list.append(fig.data[i])
        else:
            fig_data_list = [fig.data[i]] + fig_data_list

    # Shuffle data into the wanted zorder:
    fig.data = fig_data_list

    # Update x axis limits:
    xmin = df_waterfalls['Probabilities'].min() - 2
    xmax = df_waterfalls['Probabilities'].max() + 2
    xmin = xmin if xmin < 0.0 else 0.0
    xmax = xmax if xmax > 100.0 else 100.0
    fig.update_xaxes(range=[xmin, xmax])


    # Titles and labels:
    fig.update_layout(
        title='Waterfalls for all stroke teams',
        xaxis_title='Probability of thrombolysis (%)',
        yaxis_title='Feature',
        legend_title='Highlighted team'
        )

    # Update the hover text:
    fig.update_traces(
        hovertemplate=(
            'Stroke team: %{customdata[0]}' +
            '<br>' +
            'Effect of %{y}: %{customdata[1]:>+.2f}%' +
            '<br>' +
            'Final probability: %{customdata[2]:>.2f}%' +
            '<extra></extra>'
            )
        )

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

    # Flip y-axis so bars are read from top to bottom.
    fig['layout']['yaxis']['autorange'] = 'reversed'

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True)
