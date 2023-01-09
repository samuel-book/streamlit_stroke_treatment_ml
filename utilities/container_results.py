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
import matplotlib

# For creating SHAP waterfall in response to click:
import utilities.main_calculations
# For clickable plotly events:
from streamlit_plotly_events import plotly_events

# For matplotlib plots:
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

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

    plot_sorted_probs(sorted_results)

    # Move this to its own container 
    base_values=0.2995270168908044

    st.markdown('### Probability waterfalls')
    st.markdown(''.join([
        'We can look at how the model decides on the probability ',
        'of thrombolysis. Before the model looks at any of the ',
        'patient\'s details, the patient starts with a base probability ',
        f'of {100.0*base_values:.2f}%. ',
        'The model then looks at the value of each feature of the patient ',
        'in turn, and adjusts this probability upwards or downwards.'
    ]))
    st.markdown(''.join([
        'The process can be visualised as a waterfall plot.'
    ]))
    # Get big SHAP probability grid:
    grid, grid_cat_sorted, stroke_team_2d, headers = make_heat_grids(
        headers_X, sorted_results['Stroke team'], sorted_results['Index'],
        shap_values_probability_all)

    # These grids have teams in the same order as sorted_results.
    # Pick out the subset of benchmark teams:
    inds_bench = np.where(sorted_results['Benchmark rank'].to_numpy() <= 30)[0]
    inds_nonbench = np.where(sorted_results['Benchmark rank'].to_numpy() > 30)[0]

    grid_cat_bench = grid_cat_sorted[:, inds_bench]
    grid_cat_nonbench = grid_cat_sorted[:, inds_nonbench]


    # plot_heat_grid_full(grid)
    # plot_heat_grid_compressed(
    #     grid_cat_sorted, sorted_results['Stroke team'], headers,
    #     stroke_team_2d)

    # plot_all_prob_shifts_for_all_features_and_teams(
    #     headers, grid_cat_sorted)

    # Make dataframe for combo waterfalls:
    df_waterfalls, final_probs = make_waterfall_df(
            grid_cat_sorted,
            headers,
            sorted_results['Stroke team'],
            sorted_results['Highlighted team'],
            sorted_results['HB team'],
            base_values=0.2995270168908044
            )

    tabs_waterfall = st.tabs(
        ['Max/min/median teams', 'Highlighted teams', 'All teams', 'Shifts for highlighted teams'])
    with tabs_waterfall[2]:
        st.markdown(''.join([
            'The following chart shows the waterfall charts for all ',
            'teams. Instead of red and blue bars, each team has ',
            'a series of scatter points connected by lines. ',
            'The features are ordered with the most agreed on features ',
            'at the top, and the ones with more variation lower down. '
        ]))
        plot_combo_waterfalls(df_waterfalls, sorted_results, final_probs)

    with tabs_waterfall[3]:
        # # Write statistics:
        # st.markdown('All teams: ')
        # inds_std = write_feature_means_stds(grid_cat_sorted, headers, return_inds=True)

        # st.markdown('Benchmark teams: ')
        # # write_feature_means_stds(grid_cat_bench, headers, inds=inds_std)

        # Round the values now to save mucking about with df formatting.
        std_list = np.std(grid_cat_sorted, axis=1)
        # Sort from lowest to highest standard deviation:
        # (sorting in pandas also sorts the index column. This will look
        # confusing.)
        inds_std = np.argsort(std_list)
        # Box plot:
        box_plot_of_prob_shifts(grid_cat_sorted, grid_cat_bench, grid_cat_nonbench, headers, sorted_results, inds_std)

        # print_changes_info(grid_cat_sorted, headers, stroke_team_2d)

    waterfall_explanation_str = ''.join([
            'The features are ordered from largest negative effect on ',
            'probability to largest positive effect. ',
            'The 9 largest features are shown individually and the rest ',
            'are condensed into the "132 other features" bar. ',
            'This bar mostly contains the effect of the patient _not_ ',
            'attending the other stroke teams.'
        ])

    with tabs_waterfall[0]:
        st.markdown(waterfall_explanation_str)
        titles = [
            'Maximum probability',
            'Median probability',
            'Minimum probability'
            ]
        for i_here, i in enumerate(indices_high_mid_low):
            # Find the data:
            sv = shap_values_probability_extended_high_mid_low[i_here]
            # Change integer 0/1 to str no/yes for display:
            sv_to_display = utilities.main_calculations.\
                convert_explainer_01_to_noyes(sv)

            # Write to streamlit:
            sorted_rank = sorted_results['Sorted rank'].loc[i]
            # Final probability:
            final_prob = sorted_results['Probability'].loc[i]

            title = '__' + titles[i_here] + ' of thrombolysis__'
            team_info = (
                'Team ' +
                sorted_results['Stroke team'].loc[i] +
                f' (Rank {sorted_rank} of {sorted_results.shape[0]})'
            )
            st.markdown(title)
            # st.markdown(team_info)
            # Plot:
            plot_shap_waterfall(sv_to_display, final_prob, team_info)

    # Highlighted teams
    with tabs_waterfall[1]:
        if len(indices_highlighted) < 1:
            st.write('No teams are highlighted.')
        else:
            st.markdown(waterfall_explanation_str)
            for i_here, i in enumerate(indices_highlighted):
                # Find the data:
                sv = shap_values_probability_extended_highlighted[i_here]
                # Change integer 0/1 to str no/yes for display:
                sv_to_display = utilities.main_calculations.\
                    convert_explainer_01_to_noyes(sv)
                # Final probability:
                final_prob = sorted_results['Probability'].loc[i]

                # Write to streamlit:
                # title = 'Team ' + sorted_results['Stroke team'].loc[i]
                sorted_rank = sorted_results['Sorted rank'].loc[i]
                team_info = (
                    'Team ' +
                    sorted_results['Stroke team'].loc[i] +
                    f' (Rank {sorted_rank} of {sorted_results.shape[0]})'
                )
                # st.markdown(team_info)
                # st.markdown(
                #     f'Rank: {sorted_rank} of {sorted_results.shape[0]}')
                # Plot:
                plot_shap_waterfall(sv_to_display, final_prob, team_info)


def plot_sorted_probs(sorted_results):

    # Add the bars to the chart in the same order as the highlighted
    # teams list. Otherwise by default the bars would be added in the
    # order of sorted rank, and adding a new highlighted team could
    # change the colours of the existing teams.
    # (Currently removing a team does shuffle the colours but I don't
    # see an easy fix to that.)
    # Make the ordered list of things to add:
    highlighted_teams_list = st.session_state['hb_teams_input']
    # highlighted_teams_list = np.append(['-', '\U00002605' + ' (Benchmark)'], highlighted_teams_list)
    # Store the colours used in here:
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']

    fig = go.Figure()
    for i, leg_entry in enumerate(highlighted_teams_list):
        # Take the subset of the big dataframe that contains the data
        # for this highlighted team:
        results_here = sorted_results[
            sorted_results['HB team'] == leg_entry]
        # Choose the colour of this bar:
        colour = highlighted_teams_colours[leg_entry]
        # Add bar(s) to the chart for this highlighted team:
        fig.add_trace(go.Bar(
            x=results_here['Sorted rank'],
            y=results_here['Probability_perc'],
            # Extra data for hover popup:
            customdata=np.stack([
                results_here['Stroke team'],
                results_here['Thrombolyse_str'],
                # results_here['Benchmark']
                ], axis=-1),
            # Name for the legend:
            name=leg_entry,
            # Set bars colours:
            marker=dict(color=colour)
            ))

    # Figure title:
    # Change axis:
    fig.update_yaxes(range=[0.0, 100.0])
    xmax = sorted_results.shape[0]
    fig.update_xaxes(range=[0.0, xmax+1])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, sorted_results.shape[0], 10),
        ))

    # Update titles and labels:
    fig.update_layout(
        # title='Effect on probability by feature',
        xaxis_title=f'Rank out of {sorted_results.shape[0]} stroke teams',
        yaxis_title='Probability of giving<br>thrombolysis',
        legend_title='Highlighted team'
        )

    # Hover settings:
    # Make it so cursor can hover over any x value to show the
    # label of the survival line for (x,y), rather than needing to
    # hover directly over the line:
    fig.update_layout(hovermode='x')
    # Update the information that appears on hover:
    fig.update_traces(
        hovertemplate=(
            # Stroke team:
            '%{customdata[0]}' +
            '<br>' +
            # Probability to two decimal places:
            '%{y:>.2f}%' +
            '<br>' +
            # Yes/no whether to thrombolyse:
            'Thrombolysis: %{customdata[1]}' +
            '<br>' +
            # Yes/no whether it's a benchmark team:
            # '%{customdata[2]}'
            '<extra></extra>'
            )
        )

    # Add horizontal line at prob=0.5, the decision to thrombolyse:
    fig.add_hline(y=50.0, line=dict(color='black'))
    # Update y ticks to match this 50% line:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 20, 40, 50, 60, 80, 100],
        ))

    # Reduce size of figure by adjusting margins:
    fig.update_layout(
        margin=dict(    
            # l=50,
            # r=50,
            b=80,
            t=20,
            # pad=4
        ),
        height=250
        )
    # fig.update_xaxes(automargin=True)

    # Write to streamlit:
    # # Non-interactive version:
    # st.plotly_chart(fig, use_container_width=True)

    # Clickable version:
    # Write the plot to streamlit, and store the details of the last
    # bar that was clicked:
    selected_bar = plotly_events(fig, click_event=True, key='bars',
        override_height=250)#, override_width='50%')
    try:
        # Pull the details out of the last bar that was changed
        # (moved to or from the "highlighted" list due to being
        # clicked on) out of the session state:
        last_changed_bar = st.session_state['last_changed_bar']
    except KeyError:
        # Invent some nonsense. It doesn't matter whether it matches
        # the default value of selected_bar before anything is clicked.
        last_changed_bar = [0]
    callback_bar(selected_bar, last_changed_bar, 'last_changed_bar', sorted_results)
    # return selected_bar


def callback_bar(selected_bar, last_changed_bar, last_changed_str, sorted_results):
    """
    # When the script is re-run, this value of selected bar doesn't
    # change. So if the script is re-run for another reason such as
    # a streamlit input widget being changed, then the selected bar
    # is remembered and it looks indistinguishable from the user
    # clicking the bar again. To make sure we only make updates if
    # the user actually clicked the bar, compare the current returned
    # bar details with the details of the last bar *before* it was
    # changed.
    # When moved to or from the highlighted list, the bar is drawn
    # as part of a different trace and so its details such as
    # curveNumber change.
    # When the user clicks on the same bar twice in a row, we do want
    # the bar to change. For example, a non-highlighted bar might have
    # curveNumber=0, then when clicked changes to curveNumber=1.
    # When clicked again, the last_changed_bar has curveNumber=0,
    # but selected_bar has curveNumber=1. The bar details have changed
    # and so the following loop still happens despite x and y being
    # the same.
    """
    if selected_bar != last_changed_bar:
        # If the selected bar doesn't match the last changed bar,
        # then we need to update the graph and store the latest
        # clicked bar.
        try:
            # If a bar has been clicked, then the following line
            # will not throw up an IndexError:
            rank_selected = selected_bar[0]['x']
            # Find which team this is:
            team_selected = sorted_results['Stroke team'].loc[
                sorted_results['Sorted rank'] == rank_selected].values[0]
            # Copy the current highlighted teams list
            highlighted_teams_list_updated = \
                st.session_state['highlighted_teams']
            # Check if the newly-selected team is already in the list.
            if team_selected in highlighted_teams_list_updated:
                # Remove this team from the list.
                highlighted_teams_list_updated.remove(team_selected)
            else:
                # Add the newly-selected team to the list.
                highlighted_teams_list_updated.append(team_selected)
            # Add this new list to the session state so that
            # streamlit can access it immediately on the next re-run.
            st.session_state['highlighted_teams_with_click'] = \
                highlighted_teams_list_updated

            # Keep a copy of the bar that we've just changed,
            # and put it in the session state so that we can still
            # access it once the script is re-run:
            st.session_state[last_changed_str] = selected_bar.copy()

            # Re-run the script to get immediate feedback in the
            # multiselect input widget and the graph colours:
            st.experimental_rerun()

        except IndexError:
            # Nothing has been clicked yet, so don't change anything.
            pass


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


def plot_shap_waterfall(shap_values, final_prob, title='', n_to_show=9):

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

        if sort_by_magnitude is False:
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
                if sort_by_magnitude is False:
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
        decreasing={'marker': {'color': '#008bfa'}},
        increasing={'marker': {'color': '#ff0050'}}
    ))

    # For some reason, custom_data needs to be columns rather than rows:
    custom_data = np.stack((shap_probs_perc, patient_data_to_show), axis=-1)
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
    fig.update_traces(text=np.round(shap_probs_perc, 2),
                      selector=dict(type='waterfall'))
    fig.update_traces(textposition='inside', selector=dict(type='waterfall'))
    fig.update_traces(texttemplate='%{text:+}%',
                      selector=dict(type='waterfall'))

    # Set axis labels:
    fig.update_xaxes(title_text=' <br>Probability of thrombolysis (%)')
    fig.update_yaxes(title_text='Feature')
    fig.update_layout(title=title)

    # Add start and end prob annotations:
    fig.add_annotation(
        x=base_values_perc,
        y=-0.4,
        text=f'Start probability: {base_values_perc:.2f}%',
        showarrow=True,
        yshift=1,
        ax=0  # Make arrow vertical - a = arrow, x = x-shift.
        )
    fig.add_annotation(
        x=final_prob_perc,
        y=n_to_show-0.6,
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
    perc_thrombolyse_non_benchmark = (
        100.0 * n_thrombolyse_non_benchmark / n_non_benchmark)
    
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            f'All {n_all} teams',
            f'{perc_thrombolyse_all:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_all} teams '
            ':x:' + f' {n_all - n_thrombolyse_all} teams'
            ]))

    with cols[1]:
        st.metric(
            f'{n_benchmark} Benchmark teams',
            f'{perc_thrombolyse_benchmark:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_benchmark} teams '
            ':x:' + f' {n_benchmark - n_thrombolyse_benchmark} teams'
            ]))


    with cols[2]:
        st.metric(
            f'{n_non_benchmark} Non-benchmark teams',
            f'{perc_thrombolyse_non_benchmark:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_non_benchmark} teams '
            ':x:' + f' {n_non_benchmark - n_thrombolyse_non_benchmark} teams'
            ]))


    # Write benchmark decision:
    extra_str = '' if perc_thrombolyse_benchmark >= 50.0 else ' do not'
    decision_emoji = ':heavy_check_mark:' if perc_thrombolyse_benchmark >= 50.0 else ':x:'
    st.markdown(''.join([
        '__Benchmark decision:__ ',
        decision_emoji,
        extra_str,
        ' thrombolyse this patient.'
        ]))


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


def make_waterfall_df(
        grid_cat_sorted, headers, stroke_team_list, highlighted_team_list,
        hb_team_list, base_values=0.2995270168908044
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
    column_highlighted_teams = np.tile(
        highlighted_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(highlighted_team_list)).T.ravel()
    # Column of highlighted/benchmark teams:
    column_hb_teams = np.tile(
        hb_team_list, len(features_waterfall))\
        .reshape(len(features_waterfall), len(hb_team_list)).T.ravel()
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
    df_waterfalls['HB team'] = column_hb_teams
    return df_waterfalls, final_probs_list


def plot_combo_waterfalls(df_waterfalls, sorted_results, final_probs):
    """
    Add the elements to the chart in order so that the last thing
    added ends up on the top. Add the unhighlighted teams first,
    then the highlighted teams in the same order as the user input.
    This will make the highlighted colours here the same as in
    previous plots.
    """
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']

    # Find the indices of the non-highlighted teams:
    # inds_order = list(set(np.arange(0, len(stroke_team_list))).difference(indices_highlighted))

    bench_str = 'Benchmark \U00002605'
    plain_str = '-'

    # Getting muddled with pandas indexing so switch to numpy:
    # index_list = sorted_results['Index'].to_numpy()
    hb_list = sorted_results['HB team'].to_numpy()
    stroke_team_list = sorted_results['Stroke team'].to_numpy()

    inds_plain = np.where(hb_list == plain_str)[0]
    inds_bench = np.where(hb_list == bench_str)[0]
    # inds_highlighted = np.where(
    #     (hb_list != plain_str) & (hb_list != bench_str))[0]
    
    # # inds_plain = sorted_results['Index'].loc[sorted_results['HB team'] == plain_str].values
    # # inds_bench = sorted_results['Index'].loc[sorted_results['HB team'] == bench_str].values
    # # inds_order = np.concatenate((inds_plain, inds_bench))

    # # inds_highlighted = sorted_results['Index'][~inds_order].values
    highlighted_teams_input = st.session_state['highlighted_teams']#.copy()

    # Do this way to retain order of input:
    inds_highlighted = []
    for team in highlighted_teams_input:
        # ind_h = sorted_results['Index'].loc[sorted_results['Highlighted team'] == team].values[0]
        ind_h = np.where(sorted_results['Highlighted team'].to_numpy() == team)[0][0]
        inds_highlighted.append(ind_h)

    # # hb_team_list = hb_team_list.to_numpy()
    # # inds_plain = np.where(hb_team_list == plain_str)[0]
    # # inds_bench = np.where(hb_team_list == bench_str)[0]
    # # inds_highlighted = np.where(
    # #     (hb_team_list != plain_str) & (hb_team_list != bench_str))[0]

    # Need the astype(int) as if any of the lists are empty,
    # all of the elements will be converted to float.
    inds_order = np.concatenate((inds_plain, inds_bench, inds_highlighted)).astype(int)
    # st.write(inds_order)

    
    y_vals = np.arange(len(set(df_waterfalls['Features']))+1)*0.5 # for features + final prob

    pretty_jitter = False
    if pretty_jitter == True:
        # # Generate random jitter values for the final probability row
        y_jigg = make_pretty_jitter_offsets(final_probs)


    fig = go.Figure()
    drawn_blank_legend_line = 0
    drawn_bench_legend_line = 0
    for i, ind in enumerate(inds_order):
        # team = stroke_team_list.iloc[ind]
        # team = stroke_team_list.loc[ind]
        team = stroke_team_list[ind]
        # st.write('df', i, ind, team, index_list[i], index_list[ind], sorted_results['Sorted rank'].loc[i], sorted_results['Sorted rank'].loc[ind])
        df_team = df_waterfalls[df_waterfalls['Stroke team'] == team]
        # st.write(df_team)
        if team == df_team['Highlighted team'].iloc[0]:
            leggy = True
            # colour = #None
            opacity = 1.0
            width = 2.0
        else:
            if df_team['HB team'].iloc[0] == '-':
                if drawn_blank_legend_line > 0:
                    leggy = False
                else:
                    leggy = True
                    drawn_blank_legend_line += 1
            else:
                if drawn_bench_legend_line > 0:
                    leggy = False
                else:
                    leggy = True
                    drawn_bench_legend_line += 1
            # colour = 'grey'
            opacity = 0.25
            width = 1.0
        colour = highlighted_teams_colours[df_team['HB team'].iloc[0]]

        # Draw the waterfall
        fig.add_trace(go.Scatter(
            x=df_team['Probabilities'],
            y=y_vals, #df_team['Features'],
            mode='lines+markers',
            line=dict(width=width, color=colour), opacity=opacity,
            # c=df_waterfalls['Stroke team'],
            customdata=np.stack((
                df_team['Stroke team'],
                df_team['Prob shift'],
                df_team['Prob final'],
                df_team['Sorted rank']
                ), axis=-1),
            name=df_team['HB team'].iloc[0],
            showlegend=leggy
            ))

        if pretty_jitter == True:
            # Draw an extra marker for the final probability.
            show_legend_dot = False if i < len(inds_order)-1 else True
            df_short = df_team.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[df_short['Probabilities']],
                y=[y_vals[-1]+y_jigg[ind]], #df_team['Features'],
                mode='markers',
                # line=dict(width=width, color=colour), 
                # opacity=opacity,
                marker=dict(color=colour, size=2),
                # c=df_waterfalls['Stroke team'],
                customdata=np.stack((
                    [df_short['Stroke team']],
                    # ['N/A'],
                    [df_short['Prob final']],
                    [df_short['Sorted rank']]
                    ), axis=-1),
                name='Final Probability', #df_short['HB team'],
                showlegend=show_legend_dot
                ))
            # st.write(df_short['Probabilities'], y_jigg[ind], colour, df_short['HB team'])


            
    # fig.add_trace(go.Scatter(x=[10, 30], y=[9, 8.8], mode='lines+markers'))

    # # Add a box plot of the final probability values.
    # fig.add_trace(go.Box(
    #     x=final_probs,
    #     y0=y_vals[-1],
    #     name='Final Probability',
    #     boxpoints='all', # can also be outliers, or suspectedoutliers, or False
    #     jitter=1.0, # add some jitter for a better separation between points
    #     pointpos=0, # relative position of points wrt box
    #     line=dict(color='rgba(0,0,0,0)'),  # Set box and whisker outline to invisible
    #     fillcolor='rgba(0,0,0,0)',  # Set box and whisker fill to invisible
    #     marker=dict(color='black', size=2),
    #     customdata=np.stack((
    #         sorted_results['Stroke team'],
    #         # ['N/A'],
    #         sorted_results['Probability']*100.0,
    #         sorted_results['Sorted rank']
    #         ), axis=-1),
    #     ))
    # Add a box plot of the final probability values.
    fig.add_trace(go.Box(
        x=final_probs,
        y0=y_vals[-1],
        name='Final probability',
        boxpoints=False, # can also be outliers, or suspectedoutliers, or False
        # jitter=1.0, # add some jitter for a better separation between points
        # pointpos=0, # relative position of points wrt box
        # line=dict(color='rgba(0,0,0,0)'),  # Set box and whisker outline to invisible
        # fillcolor='rgba(0,0,0,0)',  # Set box and whisker fill to invisible
        line_color='black',
        # marker=dict(color='black', size=2)    
        customdata=np.stack((
            sorted_results['Stroke team'],
            # ['N/A'],
            sorted_results['Probability']*100.0,
            sorted_results['Sorted rank']
            ), axis=-1),
        ))

    # # # Add a violin plot
    # fig.add_trace(go.Violin(
    #     x=final_probs,
    #     # box_visible=True,
    #     line_color='black',
    #     meanline_visible=True,
    #     # box_visible=True,
    #     # fillcolor='lightseagreen',
    #     # opacity=0.6,
    #     y0=y_vals[-1],
    #     orientation='h',
    #     name='Final Probability',
    #     points=False  # 'all'
    #     ))

    # Update x axis limits:
    xmin = df_waterfalls['Probabilities'].min() - 2
    xmax = df_waterfalls['Probabilities'].max() + 2
    xmin = xmin if xmin < 0.0 else 0.0
    xmax = xmax if xmax > 100.0 else 100.0
    fig.update_xaxes(range=[xmin, xmax])


    # Titles and labels:
    fig.update_layout(
        # title='Waterfalls for all stroke teams',
        xaxis_title='Probability of thrombolysis (%)',
        yaxis_title='Feature',
        legend_title='Highlighted team'
        )
    fig.update_layout(
        yaxis = dict(
            tickmode='array',
            tickvals=np.append(y_vals, 'Final probability'),
            ticktext=np.append(df_team['Features'], 'Final probability')
        )
    )

    # Update the hover text for the lines:
    fig.update_traces(
        hovertemplate=(
            'Stroke team: %{customdata[0]}' +
            '<br>' +
            'Effect of %{y}: %{customdata[1]:>+.2f}%' +
            '<br>' +
            'Final probability: %{customdata[2]:>.2f}%' +
            '<br>' +
            'Rank: %{customdata[3]} of ' +
            f'{len(stroke_team_list)}' + ' teams' +
            '<extra></extra>'
            )
        )
    # Update the hover text for the final probabilities:
    fig.update_traces(
        hovertemplate=(
            'Stroke team: %{customdata[0]}' +
            '<br>' +
            'Final probability: %{customdata[1]:>.2f}%' +
            '<br>' +
            'Rank: %{customdata[2]} of ' +
            f'{len(stroke_team_list)}' + ' teams' +
            '<extra></extra>'
            ),
        selector={'name':'Final Probability'}
        )
    # Explicitly set hover mode (else Streamlit sets this to 'x')
    fig.update_layout(hovermode='closest')

    # Flip y-axis so bars are read from top to bottom.
    fig['layout']['yaxis']['autorange'] = 'reversed'

    # Increase margin size:
    # fig.update_layout(
    #     margin=dict(l=50))#, r=20, t=20, b=20),
    # Reduce size of figure by adjusting margins:
    fig.update_layout(
        margin=dict(
            l=150,
            r=150,
            b=80,
            t=20,
            # pad=4
        ),
        height=600,
        # width=300
        )
    fig.update_yaxes(automargin=True)


    # Move legend to side
    fig.update_layout(legend=dict(
        orientation='v', #'h',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.03
    ))
    # fig.update_layout(legend=dict(
    #     orientation='v', #'h',
    #     yanchor='top',
    #     y=-0.1,
    #     xanchor="right",
    #     x=1
    # ))

    # Remove y=0 line:
    fig.update_yaxes(zeroline=False)

    # Write to streamlit:
    # st.plotly_chart(fig, use_container_width=True)
    # Clickable version:
    # Write the plot to streamlit, and store the details of the last
    # bar that was clicked:
    selected_waterfall = plotly_events(
        fig, click_event=True, key='waterfall_combo',
        override_height=600, override_width='100%')
    
    try:
        # Pull the details out of the last bar that was changed
        # (moved to or from the "highlighted" list due to being
        # clicked on) out of the session state:
        last_changed_waterfall = st.session_state['last_changed_waterfall']
    except KeyError:
        # Invent some nonsense. It doesn't matter whether it matches
        # the default value of selected_bar before anything is clicked.
        last_changed_waterfall = [0]

    callback_waterfall(selected_waterfall, last_changed_waterfall, 'last_changed_waterfall', inds_order, stroke_team_list, pretty_jitter=pretty_jitter)


def callback_waterfall(selected_waterfall, last_changed_waterfall, last_changed_str, inds_order, stroke_team_list, pretty_jitter=False):
    """
    # When the script is re-run, this value of selected bar doesn't
    # change. So if the script is re-run for another reason such as
    # a streamlit input widget being changed, then the selected bar
    # is remembered and it looks indistinguishable from the user
    # clicking the bar again. To make sure we only make updates if
    # the user actually clicked the bar, compare the current returned
    # bar details with the details of the last bar *before* it was
    # changed.
    # When moved to or from the highlighted list, the bar is drawn
    # as part of a different trace and so its details such as
    # curveNumber change.
    # When the user clicks on the same bar twice in a row, we do want
    # the bar to change. For example, a non-highlighted bar might have
    # curveNumber=0, then when clicked changes to curveNumber=1.
    # When clicked again, the last_changed_waterfall has curveNumber=0,
    # but selected_waterfall has curveNumber=1. The bar details have changed
    # and so the following loop still happens despite x and y being
    # the same.
    """
    if selected_waterfall != last_changed_waterfall:
        # If the selected bar doesn't match the last changed bar,
        # then we need to update the graph and store the latest
        # clicked bar.
        try:
            # If a bar has been clicked, then the following line
            # will not throw up an IndexError:
            curve_selected = selected_waterfall[0]['curveNumber']
            if pretty_jitter == True:
                # Have to divide this by two because every other curve
                # is a dot in the "final probability" row.
                curve_selected = int(curve_selected*0.5)
            ind = inds_order[curve_selected]
            # Find which team this is:
            team_selected = stroke_team_list[ind]

            # Copy the current highlighted teams list
            highlighted_teams_list_updated = \
                st.session_state['highlighted_teams']
            # Check if the newly-selected team is already in the list.
            if team_selected in highlighted_teams_list_updated:
                # Remove this team from the list.
                highlighted_teams_list_updated.remove(team_selected)
            else:
                # Add the newly-selected team to the list.
                highlighted_teams_list_updated.append(team_selected)
            # Add this new list to the session state so that
            # streamlit can access it immediately on the next re-run.
            st.session_state['highlighted_teams_with_click'] = \
                highlighted_teams_list_updated

            # Keep a copy of the bar that we've just changed,
            # and put it in the session state so that we can still
            # access it once the script is re-run:
            st.session_state[last_changed_str] = selected_waterfall.copy()

            # Re-run the script to get immediate feedback in the
            # multiselect input widget and the graph colours:
            st.experimental_rerun()

        except IndexError:
            # Nothing has been clicked yet, so don't change anything.
            pass


def box_plot_of_prob_shifts(grid, grid_bench, grid_non_bench, headers, sorted_results, inds=[]):

    # Default order:
    #  0 Arrival-to-scan time,
    #  1 Infarction,
    #  2 Stroke severity,
    #  3 Precise onset time,
    #  4 Prior disability level,
    #  5 Use of AF anticoagulants,
    #  6 Onset-to-arrival time,
    #  7 Onset during sleep,
    #  8 Age,
    #  9 This stroke team,
    # 10 Other stroke teams.
    # Put it into the same order as in the input sidebar:
    inds = [4, 2, 0, 6, 8, 1, 3, 5, 7, 9, 10]

    # Sort data:
    if len(inds) > 0:
        grid = grid[inds, :]
        grid_bench = grid_bench[inds, :]
        grid_non_bench = grid_non_bench[inds, :]
        headers = headers[inds]
    # # # Quick plot demo
    # # df = pd.DataFrame(
    # #     grid_cat_sorted.T,
    # #     # columns=headers
    # # )
    # # st.write(df)


    # Find min/max/average value for each feature and grid:
    ave_list = np.median(grid, axis=1)
    max_list = np.max(grid, axis=1)
    min_list = np.min(grid, axis=1)
    ave_list_bench = np.median(grid_bench, axis=1)
    max_list_bench = np.max(grid_bench, axis=1)
    min_list_bench = np.min(grid_bench, axis=1)
    ave_list_non_bench = np.median(grid_non_bench, axis=1)
    max_list_non_bench = np.max(grid_non_bench, axis=1)
    min_list_non_bench = np.min(grid_non_bench, axis=1)

    highlighted_teams = st.session_state['highlighted_teams']
    n_stroke_teams = grid.shape[1]


    y_vals = [0, 1, 2]  #headers #np.arange(0, len(grid))
    # Where to scatter the team markers:
    y_offsets_scatter = np.linspace(0.2, -0.2, len(highlighted_teams))

    row_headers = [
        'Team',
        f'Rank for this feature (of {n_stroke_teams} teams)',
        'Effect on probability (%)'
    ]

    for i, column in enumerate(grid):
        feature = headers[i]
        # Values for this feature:
        feature_values = grid[i]
        # Store effects for highlighted teams in here:
        effect_vals = []

        cols = st.columns(2)
        with cols[0]:
            st.markdown('### ' + feature)
            table = []
            for team in highlighted_teams:
                team_for_table = sorted_results['HB team'][
                    sorted_results['Stroke team'] == team].to_numpy()[0]
                
                # Find where this team is in the overall list:
                rank_overall = sorted_results['Sorted rank'][
                    sorted_results['Stroke team'] == team]
                # Find the effect of this value for this feature:
                effect_val_perc = grid[i][rank_overall-1][0]
                effect_vals.append(effect_val_perc)
                # effect_val_perc = 100.0 * effect_val
                # Find how it compares with the other features
                rank_here = np.where(
                    np.sort(feature_values) == effect_val_perc)[0][0]
                row0 = [team_for_table, rank_here, effect_val_perc]
                table.append(row0)

            df = pd.DataFrame(table, columns=row_headers)
            st.table(df)

        effect_diffs = []
        for t, team in enumerate(highlighted_teams):
            # Find the difference between this effect_val and
            # the min/max/average values for this feature
            # and this benchmark category.

            # Can definitely shorten this code a lot when tidying
            # (build grid, slice, reshape result?)
            v = effect_vals[t]
            diff_ave = v - ave_list[i]
            diff_max = v - max_list[i]
            diff_min = v - min_list[i]
            diff_ave_bench = v - ave_list_bench[i]
            diff_max_bench = v - max_list_bench[i]
            diff_min_bench = v - min_list_bench[i]
            diff_ave_non_bench = v - ave_list_non_bench[i]
            diff_max_non_bench = v - max_list_non_bench[i]
            diff_min_non_bench = v - min_list_non_bench[i]

            effect_diffs.append([
                [diff_max, diff_min, diff_ave],
                [diff_max_bench, diff_min_bench, diff_ave_bench],
                [diff_max_non_bench, diff_min_non_bench, diff_ave_non_bench],
            ])

                
        with cols[1]:
            fig = go.Figure()
            # plotly_colours = px.colors.qualitative.Plotly
            box_colour = 'black'  # plotly_colours[0]

            # Draw the box plots:
            fig.add_trace(go.Box(
                x=grid[i],
                y0=y_vals[0],
                name='All',
                line=dict(color=box_colour),
                boxpoints=False,
                hoveron='points'  # Switch off the hover label
                ))
            fig.add_trace(go.Box(
                x=grid_bench[i],
                y0=y_vals[1],
                name='Benchmark',
                line=dict(color=box_colour),
                boxpoints=False,
                hoveron='points'  # Switch off the hover label
                ))
            fig.add_trace(go.Box(
                x=grid_non_bench[i],
                y0=y_vals[2],
                name='Not benchmark',
                line=dict(color=box_colour),
                boxpoints=False,
                hoveron='points'  # Switch off the hover label
                ))

            # Setup for box:
            fig.update_layout(boxgap=0.01)#, boxgroupgap=1)
            fig.update_traces(width=0.5)

            # Mark the highlighted teams:
            for t, team in enumerate(highlighted_teams):
                # Index in the big array:
                ind = np.where(sorted_results['Stroke team'].values == team)[0]
                hb_team = sorted_results['HB team'].values[ind][0]
                colour = st.session_state['highlighted_teams_colours'][hb_team]
                team_effect = effect_vals[t]

                # fig.add_vline(x=team_effect, line=dict(color=colour))
                # Add a separate marker for each grid (all, bench, non-bench)
                extra_strs = [' (All)', ' (Benchmark)', '(Non-benchmark)']
                for y, y_val in enumerate(y_vals):
                    diff_max = effect_diffs[t][y][0]
                    diff_min = effect_diffs[t][y][1]
                    diff_ave = effect_diffs[t][y][2]
                    # # Make arrow strings depending on diff
                    # arr_max = '\U00002191' if diff_max > 0 else '\U00002193'
                    # arr_min = '\U00002191' if diff_min > 0 else '\U00002193'
                    # arr_ave = '\U00002191' if diff_ave > 0 else '\U00002193'
                    custom_data = np.stack([
                        # effect_diffs[t][y],  # Differences
                        # [arr_max, arr_min, arr_ave],  # Arrow strings
                        # effect_diffs[t][y],
                        [round(diff_max, 2)], [round(diff_min, 2)], [round(diff_ave, 2)],
                        # [arr_max], [arr_min], [arr_ave]
                        [hb_team]
                    ], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=[team_effect],
                        y=[y_val + y_offsets_scatter[t]],
                        mode='markers',
                        name=team + extra_strs[y],
                        marker=dict(color=colour,
                            line=dict(color='black', width=1.0)),
                        customdata=custom_data
                        ))
            # Setup for markers:
            # # Hover settings:
            # # Make it so cursor can hover over any x value to show the
            # # label of the survival line for (x,y), rather than needing to
            # # hover directly over the line:
            # fig.update_layout(hovermode='y')
            # # Update the information that appears on hover:
            fig.update_traces(
                hovertemplate=(
                    'Team %{customdata[3]}' +
                    '<br>' +
                    'Effect: %{x:.2f}%' +
                    '<br>' +
                    '%{customdata[0]:+}% from max' +
                    '<br>' +
                    '%{customdata[1]:+}% from min' +
                    '<br>' +
                    '%{customdata[2]:+}% from median' +
                    # '<br>' +
                    # # Stroke team:
                    # '%{customdata[0]}' +
                    # '<br>' +
                    # # Probability to two decimal places:
                    # '%{y:>.2f}%' +
                    # '<br>' +
                    # # Yes/no whether to thrombolyse:
                    # 'Thrombolysis: %{customdata[1]}' +
                    # '<br>' +
                    # # Yes/no whether it's a benchmark team:
                    # '%{customdata[2]}'
                    # Remove everything in the second box:
                    '<extra></extra>'
                    )
                )



            # Setup:

            # Change the background colour of stuff in the frame
            # (stuff outside the frame is paper_bgcolor)
            # fig.update_layout(plot_bgcolor='black')
            

            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_vals,
                    ticktext=['All', 'Benchmark', 'Not benchmark']
                ))
            fig.update_yaxes(tickangle=90)
            fig.update_layout(showlegend=False)
            # fig.update_traces(orientation='h')
            # fig.update_layout(height=350)

            # Flip y-axis so boxes are read from top to bottom.
            fig['layout']['yaxis']['autorange'] = 'reversed'



            
            # Update titles and labels:
            fig.update_layout(
                # title='Effect on probability by feature',
                xaxis_title='Shift in probability (%)',
                # yaxis_title='Probability of giving<br>thrombolysis',
                # legend_title='Highlighted team'
                )

            # Reduce size of figure by adjusting margins:
            fig.update_layout(
                margin=dict(    
                    l=0,
                    r=0,
                    b=0,
                    t=40,
                    pad=0
                ),
                height=350
                )
            # st.write(fig.data[0].hovertemplate)

            # Write to streamlit:
            st.plotly_chart(fig, use_container_width=True)


def make_pretty_jitter_offsets(final_probs):
    # Coordinates of all points:
    all_coords = np.transpose(
        np.vstack((final_probs, np.zeros_like(final_probs))))
    # Minimum distance between points:
    d_min = 1.0  # % in x axis
    # # Weight the jitter by distance from the nearest point.
    # # (assuming that final_probs is sorted)
    diff_x = np.append(0.0, np.abs(np.diff(final_probs)))
    inds_close = np.where(diff_x < d_min)[0]
    # Store the jiggled y-values in here:
    y_jigg = []
    jigg_dir = 1

    for ind in range(len(final_probs)):
        # st.write('ind', ind)
        if ind in inds_close:
            # Find all inds within d_min in x of this point.
            x_here = all_coords[ind][0]
            y_here = all_coords[ind][1]
            x_max = x_here + d_min
            inds_close_here = np.where((final_probs >= x_here) & (final_probs < x_max))[0]
            # Remove itself from this list:
            ind_here = np.where(inds_close_here == ind)
            inds_close_here = np.delete(inds_close_here, ind_here)

            # st.write(x_here)#, y_here, x_max, inds_close_here)
            # if len(inds_close_here) == 0:
            #     y_jigg.append(0)
            # else:
            # Work out how much to jiggle this value in y 
            # st.write('While loop:')
            # count = 0
            # Find current distance between this point and
            # the other close points.
            coords_nearby_list = np.copy(all_coords)[inds_close_here]
            inds_still_close_here = np.copy(inds_close_here)
            count = 0
            
            if len(inds_still_close_here) < 1:
                success = 1
            else:
                success = 0
            while success < 1:
                # st.write('    ', len(inds_close_here), inds_close_here)

                dist_list = np.sqrt(
                    (coords_nearby_list[:, 0] - x_here)**2.0 + 
                    (coords_nearby_list[:, 1] - y_here)**2.0
                )
                ind_nearest = np.where(dist_list == np.min(dist_list))
                coords_nearest = coords_nearby_list[ind_nearest]
                x_next = coords_nearest[0][0]
                # Find out current difference between this point and
                # next nearest in x:
                # x_next = coords_nearby_list[-2, 0]
                x_diff = x_here - x_next
                # st.write('xdiff', x_diff)
                y_here += jigg_dir * np.sqrt(d_min**2.0 - x_diff**2.0)
                # y_here = jigg_dir * count * d_min*0.1
                # st.write(dist_list)
                # st.write(dist_list[0, :])
                inds_still_close_here = np.where(dist_list <= d_min)[0]
                # st.write(inds_close_here)
                # count += 1
                if len(inds_still_close_here) < 1:
                    success = 1
                elif count > 100:
                    # Looping for too long. Bow out now.
                    success = 1

            all_coords[ind][1] = y_here
            # Set the next point to be moved to move in the other direction.
            jigg_dir *= -1
        else:
            y_here = 0
        # Add the y value to the list:
        y_jigg.append(y_here)
        # st.write('y_here', y_here)
        # st.text(y_here)
        # st.text(type(y_here))

    # Squash y_jigg down into a smaller y space:
    # y_jigg = np.array(y_jigg) * 0.6*(y_vals[-1] / 100.0)
    y_jigg = 0.3 * np.array(y_jigg) / np.max(np.abs(y_jigg))

    return y_jigg
