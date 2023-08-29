"""
All of the content for the Results section.
"""
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
# import matplotlib

# For clickable plotly events:
from streamlit_plotly_events import plotly_events

from utilities_ml.fixed_params import bench_str, plain_str#, \
    # default_highlighted_team, display_name_of_default_highlighted_team


def plot_combo_waterfalls(df_waterfalls, sorted_results, final_probs, patient_data_waterfall, 
                          default_highlighted_team, display_name_of_default_highlighted_team, use_plotly_events=True):
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
        if team == display_name_of_default_highlighted_team:
            df_team = default_highlighted_team
        else:
            df_team = team
        # ind_h = sorted_results['Index'].loc[sorted_results['Highlighted team'] == team].values[0]
        ind_h = np.where(sorted_results['Highlighted team'].to_numpy() == df_team)[0][0]
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

    
    y_vals = np.arange(0, -(len(set(df_waterfalls['Features']))+1), -1)*0.5 # for features + final prob

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
            if df_team['HB team'].iloc[0] == plain_str:
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

        if team == default_highlighted_team:
            name_list = [display_name_of_default_highlighted_team] * len(df_team['Stroke team'])
            trace_name = display_name_of_default_highlighted_team
        else:
            name_list = df_team['Stroke team']
            trace_name = df_team['HB team'].iloc[0]

        # Draw the waterfall
        fig.add_trace(go.Scatter(
            x=df_team['Probabilities'],
            y=y_vals, #df_team['Features'],
            mode='lines+markers',
            line=dict(width=width, color=colour), opacity=opacity,
            # c=df_waterfalls['Stroke team'],
            customdata=np.stack((
                name_list,
                df_team['Prob shift'],
                df_team['Prob final'],
                df_team['Sorted rank'],
                df_team['Features']
                ), axis=-1),
            name=trace_name,
            showlegend=leggy,
            # legendgroup='2',
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
                showlegend=show_legend_dot,
                # legendgroup='2',
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

    # # Add a box plot of the final probability values.
    # fig.add_trace(go.Box(
    #     x=final_probs,
    #     y0=y_vals[-1],
    #     name='Final probability',
    #     boxpoints=False, # can also be outliers, or suspectedoutliers, or False
    #     # jitter=1.0, # add some jitter for a better separation between points
    #     # pointpos=0, # relative position of points wrt box
    #     # line=dict(color='rgba(0,0,0,0)'),  # Set box and whisker outline to invisible
    #     # fillcolor='rgba(0,0,0,0)',  # Set box and whisker fill to invisible
    #     line_color='black',
    #     # marker=dict(color='black', size=2)    
    #     customdata=np.stack((
    #         sorted_results['Stroke team'],
    #         # ['N/A'],
    #         sorted_results['Probability']*100.0,
    #         sorted_results['Sorted rank']
    #         ), axis=-1),
    #     ))

    # # Add a violin plot
    fig.add_trace(go.Violin(
        x=final_probs,
        line_color='grey',
        # meanline_visible=True,
        # box_visible=True,
        # fillcolor='lightseagreen',
        # opacity=0.6,
        y0=y_vals[-1],
        orientation='h',
        name='Final Probability',
        points=False,  # 'all'
        # legendgroup='2',
        ))

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
        # Add some blank lines below "feature" to help position it.
        yaxis_title=None, #'Feature<br> <br> <br> ',
        legend_title='Highlighted team'  # + '~'*20
        )


    # fig.update_layout(
    #     yaxis=dict(
    #         tickmode='array',
    #         tickvals=np.append(y_vals, 'Final probability'),
    #         ticktext=np.append(df_team['Features'], 'Final probability')
    #     )
    # )

    # Sort out tick values and labels:
    y_tickvals = y_vals # np.append(y_vals, 'Final probability')
    y_ticktext = np.append(df_team['Features'], 'Final probability')
    # Double up the ticks by slightly offsetting one set:
    # y_tickvals = np.append(y_tickvals, y_tickvals + 1e-7)
    # For the second set, add the feature values to the labels.
    # Combine feature names and values for tick labels:
    # (same idea as the original shap red/blue waterfall plot)
    features_with_values_waterfall = ['Base probability']
    for i, value in enumerate(patient_data_waterfall):
        if value != '':
            value = str(value)
            if 'rrival' in y_ticktext[i+1]:
                # Onset to arrival or arrival to scan time:
                value += ' mins'
            elif 'Age' in y_ticktext[i+1]:
                value += ' years'
            # If it's not a dummy feature value, add an equals sign:
            value += ' = '
        # Combine the value and the feature name:
        feature_with_value = value + y_ticktext[i+1]
        features_with_values_waterfall.append(feature_with_value)
    features_with_values_waterfall.append('Final probability')

    
    # # Add these new tick labels to the existing list:
    # y_ticktext = np.append(features_with_values_waterfall, y_ticktext)

    # y_ticktext_colours = []
    # for y, text in enumerate(y_ticktext):
    #     colour = 'gray' if y < len(y_vals) else 'black'
    #     # t = '$\color{' + str(colour) + '}{' + str(text) + '}$'
    #     # t = '<color="' + colour + '">' + text #+ '</color>'
    #     t = 'body{color:"' + colour + '"}' + text #+ '</color>'
    #     y_ticktext_colours.append(t)
    # st.write(y_ticktext_colours)

    # y_range = [y_vals[0]+0.25, y_vals[-1]-0.25]
    y_range = [y_vals[-1] - 0.5, y_vals[0] + 0.1]
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=y_tickvals,
            # ticktext=y_ticktext,  #_colours
            ticktext=features_with_values_waterfall,
            tickfont=dict(color='darkgray'),
            range=y_range
        )
    )

    # Add empty trace to generate second axis:
    fig.add_trace(go.Scatter(yaxis='y2'))
    # Update second axis:
    fig.update_layout(
        yaxis2=dict(
            tickmode='array',
            tickvals=y_tickvals, # + 1e-7,
            ticktext=y_ticktext,  #_colours
            # ticktext=features_with_values_waterfall,
            # tickfont=dict(color='grey'),
            overlaying='y',
            side='left',
            range=y_range
        ),
    )


    # Update the hover text for the lines:
    fig.update_traces(
        hovertemplate=(
            'Stroke team: %{customdata[0]}' +
            '<br>' +
            # 'Effect of %{y}: %{customdata[1]:>+.2f}%' +
            'Effect of %{customdata[4]}: %{customdata[1]:>+.2f}%' +
            '<br>' +
            'Final probability: %{customdata[2]:>.2f}%' +
            '<br>' +
            'Rank: %{customdata[3]} of ' +
            f'{len(stroke_team_list)}' + ' teams' +
            '<extra></extra>'
            )
        )
    # NEed the following for jitter option?
    # # Update the hover text for the final probabilities:
    # fig.update_traces(
    #     # hovertemplate=(
    #     #     # 'Stroke team: %{customdata[0]}' +
    #     #     # '<br>' +
    #     #     # 'Final probability: %{customdata[1]:>.2f}%' +
    #     #     # '<br>' +
    #     #     # 'Rank: %{customdata[2]} of ' +
    #     #     # f'{len(stroke_team_list)}' + ' teams' +
    #     #     '<extra></extra>'
    #     #     ),
    #     # hoveron=False,
    #     hoverinfo='skip',
    #     selector={'name': 'Final Probability'}
    #     )
    # Explicitly set hover mode (else Streamlit sets this to 'x')
    fig.update_layout(hovermode='closest')

    # Add three scatter markers for min/max/median
    # with horizontal line connecting them:
    # (KEEP THIS AFTER HOVER TEMPLATE SET FOR WATERFALLS
    # otherwise it'll overwrite the request to not show any
    # info on hover).
    fig.add_trace(go.Scatter(
        x=[np.min(final_probs), np.max(final_probs), np.median(final_probs)],
        y=[y_vals[-1]]*3,
        line_color='black',
        marker=dict(size=20, symbol='line-ns-open'),
        name='Final Probability',
        showlegend=False,
        hoverinfo='skip',
        # legendgroup='2',
        ))

    # Flip y-axis so bars are read from top to bottom.
    # fig['layout']['yaxis']['autorange'] = 'reversed'
    # fig['layout']['yaxis2']['autorange'] = 'reversed'
    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(
        l=200,
        r=20,  # 150, 
        b=80, t=20), height=600)
    # Make the y axis title stand out from the tick labels:
    # fig.update_yaxes(automargin=True)
    # Move legend to side
    fig.update_layout(legend=dict(
        orientation='v', #'h',
        yanchor='top',
        y=1,
        xanchor='right',
        x=1.04,
        # itemwidth=50
    ))
    # Remove y=0 line:
    fig.update_yaxes(zeroline=False)
    # Remove other vertical grid lines:
    fig.update_xaxes(showgrid=False)


    # Disable zoom and pan:
    fig.update_layout(
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        xaxis2=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True),
        )

    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)
    # Only change the specific item being clicked on, not the whole
    # legend group:
    # # fig.update_layout(legend=dict(groupclick="toggleitem"))

    # plotly_config = {
    #     # Mode bar always visible:
    #     # 'displayModeBar': True,
    #     # Plotly logo in the mode bar:
    #     'displaylogo': False,
    #     # Remove the following from the mode bar:
    #     'modeBarButtonsToRemove': [
    #         'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
    #         'lasso2d'
    #         ],
    #     # Options when the image is saved:
    #     'toImageButtonOptions': {'height': None, 'width': None},
    #     }


    # This doesn't work with plotly_events:
    # fig.update_layout(modebar_remove=[
    #     'zoom', 'pan', 'select', 'zoomIn',
    #     'zoomOut', 'autoScale', 'lasso2d'
    #     ])

    # # Fake a legend with annotations:
    # fig.add_annotation(dict(x=1.0, y=0.7, xref="paper", yref="paper", 
    #                         text='testing', showarrow=False))

    # # Write to streamlit:
    if use_plotly_events is False:
        # Non-interactive version:
        fig.update_layout(width=700)
        plotly_config = {
            # Mode bar always visible:
            # 'displayModeBar': True,
            # Plotly logo in the mode bar:
            'displaylogo': False,
            # Remove the following from the mode bar:
            'modeBarButtonsToRemove': [
                'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
                'lasso2d'
                ],
            # Options when the image is saved:
            'toImageButtonOptions': {'height': None, 'width': None},
            }
        st.plotly_chart(
            fig,
            # use_container_width=True,
            config=plotly_config)
    else:
        # Clickable version:
        # Write the plot to streamlit, and store the details of the last
        # bar that was clicked:
        selected_waterfall = plotly_events(
            fig, click_event=True, key='waterfall_combo',
            override_height=600, override_width='100%')#, config=plotly_config)

        callback_waterfall(
            selected_waterfall, inds_order, stroke_team_list,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            pretty_jitter=pretty_jitter)


def callback_waterfall(
        selected_waterfall,
        inds_order,
        stroke_team_list,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        pretty_jitter=False):
    """
    When the script is re-run, this value of selected bar doesn't
    change. So if the script is re-run for another reason such as
    a streamlit input widget being changed, then the selected bar
    is remembered and it looks indistinguishable from the user
    clicking the bar again. To make sure we only make updates if
    the user actually clicked the bar, compare the current returned
    bar details with the details of the last bar *before* it was
    changed.
    When moved to or from the highlighted list, the bar is drawn
    as part of a different trace and so its details such as
    curveNumber change.
    When the user clicks on the same bar twice in a row, we do want
    the bar to change. For example, a non-highlighted bar might have
    curveNumber=0, then when clicked changes to curveNumber=1.
    When clicked again, the last_changed_waterfall has curveNumber=0,
    but selected_waterfall has curveNumber=1. The bar details have changed
    and so the following loop still happens despite x and y being
    the same.
    """

    try:
        # Pull the details out of the last bar that was changed
        # (moved to or from the "highlighted" list due to being
        # clicked on) out of the session state:
        last_changed_waterfall = st.session_state['last_changed_waterfall']
    except KeyError:
        # Invent some nonsense. It doesn't matter whether it matches
        # the default value of selected_bar before anything is clicked.
        last_changed_waterfall = [0]

    if selected_waterfall != last_changed_waterfall:
        # If the selected bar doesn't match the last changed bar,
        # then we need to update the graph and store the latest
        # clicked bar.
        try:
            # If a bar has been clicked, then the following line
            # will not throw up an IndexError:
            curve_selected = selected_waterfall[0]['curveNumber']
            if pretty_jitter is True:
                # Have to divide this by two because every other curve
                # is a dot in the "final probability" row.
                curve_selected = int(curve_selected*0.5)
            ind = inds_order[curve_selected]
            # Find which team this is:
            team_selected = stroke_team_list[ind]

            # Update the label if this is "St Elsewhere":
            if team_selected == default_highlighted_team:
                team_selected = display_name_of_default_highlighted_team

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
            st.session_state['last_changed_waterfall'] = \
                selected_waterfall.copy()

            # Re-run the script to get immediate feedback in the
            # multiselect input widget and the graph colours:
            st.experimental_rerun()

        except IndexError:
            # Nothing has been clicked yet, so don't change anything.
            pass


def box_plot_of_prob_shifts(
        grid,
        grid_bench,
        grid_non_bench,
        headers,
        sorted_results,
        hb_teams_input,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        starting_probabilities,
        inds=[]
        ):

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
    # inds = [9, 4, 2, 0, 6, 8, 1, 3, 5, 7] #, 10]
    
    # 0 stroke_severity
    # 1 prior_disability
    # 2 age
    # 3 infarction
    # 4 onset_to_arrival_time
    # 5 precise_onset_known
    # 6 onset_during_sleep
    # 7 arrival_to_scan_time
    # 8 afib_anticoagulant
    # 9 Stroke team attended
    inds = [9, 1, 0, 7, 4, 2, 3, 5, 8, 6]

    # Sort feature order:
    if len(inds) > 0:
        grid = grid[inds, :]
        grid_bench = grid_bench[inds, :]
        grid_non_bench = grid_non_bench[inds, :]
        headers = headers[inds]

    # Add final probability to grid:
    final_probs = np.sum(grid, axis=0)
    final_probs_bench = np.sum(grid_bench, axis=0)
    final_probs_non_bench = np.sum(grid_non_bench, axis=0)
    # Add base probabilities to these:
    start_probs = 100.0 * starting_probabilities  # 0.2995270168908044
    final_probs += start_probs
    final_probs_bench += start_probs
    final_probs_non_bench += start_probs

    def add_feature_to_grid(old_grid, new_vals):
        new_grid = np.zeros((old_grid.shape[0]+1, old_grid.shape[1]))
        new_grid[1:, :] = old_grid
        new_grid[0, :] = new_vals
        return new_grid

    grid = add_feature_to_grid(grid, final_probs)
    grid_bench = add_feature_to_grid(grid_bench, final_probs_bench)
    grid_non_bench = add_feature_to_grid(grid_non_bench, final_probs_non_bench)

    headers = np.append('Final probability', headers)

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


    y_vals = [0, 1, 2]
    y_gap = 0.05
    y_max = 0.2
    # Where to scatter the team markers:
    y_offsets_scatter = []#[0.0]
    while len(y_offsets_scatter) < len(highlighted_teams):
        y_extra = np.arange(y_gap, y_max, y_gap)
        y_extra = np.stack((
            y_extra, -y_extra
        )).T.flatten()
        y_offsets_scatter = np.append(y_offsets_scatter, y_extra)
        y_gap = 0.5 * y_gap


    for i, column in enumerate(grid):
        feature = headers[i]
        # Values for this feature:
        feature_values = grid[i]
        # Store effects for highlighted teams in here:
        effect_vals = []

        if i > 0:
            row_headers = [
                'Team',
                f'Rank for this feature (of {n_stroke_teams} teams)',
                'Effect on probability (%)'
            ]
        else:
            # Overall probability:
            row_headers = [
                'Team',
                f'Rank for this feature (of {n_stroke_teams} teams)',
                'Probability (%)'
            ]

        cols = st.columns(2)
        with cols[0]:
            st.markdown('### ' + feature)
            table = []
            for team in highlighted_teams:
                if team == display_name_of_default_highlighted_team:
                    df_team = default_highlighted_team
                    team_for_table = display_name_of_default_highlighted_team
                else:
                    df_team = team
                    team_for_table = sorted_results['HB team'][
                        sorted_results['Stroke team'] == df_team].to_numpy()[0]

                # Find where this team is in the overall list:
                rank_overall = sorted_results['Sorted rank'][
                    sorted_results['Stroke team'] == df_team]
                # Find the effect of this value for this feature:
                effect_val_perc = grid[i][rank_overall-1][0]
                effect_vals.append(effect_val_perc)
                # effect_val_perc = 100.0 * effect_val
                # Find how it compares with the other features
                rank_here = 1 + np.where(
                    np.sort(feature_values)[::-1] == effect_val_perc)[0][0]
                row0 = [team_for_table, rank_here, effect_val_perc]
                table.append(row0)

            def colour_for_table(team):
                if team == display_name_of_default_highlighted_team:
                    team = default_highlighted_team
                if team in hb_teams_input:
                    colour = st.session_state['highlighted_teams_colours'][team]
                else:
                    colour = None

                return 'color: %s' % colour

            df = pd.DataFrame(table, columns=row_headers)
            # st.table(df)
            st.table(df.style.applymap(colour_for_table))

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
            # (black doesn't show up well in dark mode)
            box_colour = 'grey'  # plotly_colours[0]

            # Draw the box plots:
            fig.add_trace(go.Violin(
                x=grid[i],
                y0=y_vals[0],
                name='All',
                line=dict(color=box_colour),
                # boxpoints=False,
                hoveron='points',  # Switch off the hover label
                orientation='h',
                points=False
                ))
            fig.add_trace(go.Violin(
                x=grid_bench[i],
                y0=y_vals[1],
                name='Benchmark',
                line=dict(color=box_colour),
                # boxpoints=False,
                hoveron='points',  # Switch off the hover label
                orientation='h',
                points=False
                ))
            fig.add_trace(go.Violin(
                x=grid_non_bench[i],
                y0=y_vals[2],
                name='Not benchmark',
                line=dict(color=box_colour),
                # boxpoints=False,
                hoveron='points',  # Switch off the hover label
                orientation='h',
                points=False
                ))




            # Setup for box:
            fig.update_layout(boxgap=0.01)#, boxgroupgap=1)
            fig.update_traces(width=0.5)

            # Mark the highlighted teams:
            for t, team in enumerate(highlighted_teams):
                if team == display_name_of_default_highlighted_team:
                    df_team = default_highlighted_team
                else:
                    df_team = team

                # Index in the big array:
                ind = np.where(sorted_results['Stroke team'].values == df_team)[0]
                hb_team = sorted_results['HB team'].values[ind][0]
                colour = st.session_state['highlighted_teams_colours'][hb_team]
                team_effect = effect_vals[t]

                if team == display_name_of_default_highlighted_team:
                    name_for_customdata = team
                else:
                    name_for_customdata = hb_team

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
                        [name_for_customdata]
                    ], axis=-1)

                    fig.add_trace(go.Scatter(
                        x=[team_effect],
                        y=[y_val + y_offsets_scatter[t]],
                        mode='markers',
                        # name=team + extra_strs[y],
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
                    '%{customdata[3]}' +
                    '<br>' +
                    'Effect: %{x:.2f}%' +
                    '<br>' +
                    '%{customdata[0]:+}% from max' +
                    '<br>' +
                    '%{customdata[1]:+}% from min' +
                    '<br>' +
                    '%{customdata[2]:+}% from median' +
                    # Remove everything in the second box:
                    '<extra></extra>'
                    )
                )
            
            # Add three scatter markers for min/max/median
            # with horizontal line connecting them:
            # (KEEP THIS AFTER HOVER TEMPLATE SET FOR WATERFALLS
            # otherwise it'll overwrite the request to not show any
            # info on hover).
            fig.add_trace(go.Scatter(
                x=[np.min(grid[i]), np.max(grid[i]), np.median(grid[i])],
                y=[y_vals[0]]*3,
                line_color='black',
                marker=dict(size=20, symbol='line-ns-open'),
                # name='Final Probability',
                showlegend=False,
                hoverinfo='skip',
                # legendgroup='2',
                ))
        
            fig.add_trace(go.Scatter(
                x=[np.min(grid_bench[i]), np.max(grid_bench[i]), np.median(grid_bench[i])],
                y=[y_vals[1]]*3,
                line_color='black',
                marker=dict(size=20, symbol='line-ns-open'),
                # name='Final Probability',
                showlegend=False,
                hoverinfo='skip',
                # legendgroup='2',
                ))
        
            fig.add_trace(go.Scatter(
                x=[np.min(grid_non_bench[i]), np.max(grid_non_bench[i]), np.median(grid_non_bench[i])],
                y=[y_vals[2]]*3,
                line_color='black',
                marker=dict(size=20, symbol='line-ns-open'),
                # name='Final Probability',
                showlegend=False,
                hoverinfo='skip',
                # legendgroup='2',
                ))


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



            x_title = 'Shift in probability (%)' if i > 0 else 'Probability (%)'
            # Update titles and labels:
            fig.update_layout(
                # title='Effect on probability by feature',
                xaxis_title=x_title,
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

            plotly_config = {
                # Mode bar always visible:
                # 'displayModeBar': True,
                # Plotly logo in the mode bar:
                'displaylogo': False,
                # Remove the following from the mode bar:
                'modeBarButtonsToRemove': [
                    'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
                    'lasso2d'
                    ],
                # Options when the image is saved:
                'toImageButtonOptions': {'height': None, 'width': None},
                }


            # Disable zoom and pan:
            fig.update_layout(
                # Left subplot:
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True),
                # Right subplot:
                xaxis2=dict(fixedrange=True),
                yaxis2=dict(fixedrange=True)
                )

            # Turn off legend click events
            # (default is click on legend item, remove that item from the plot)
            fig.update_layout(legend_itemclick=False)
            # Only change the specific item being clicked on, not the whole
            # legend group:
            # # fig.update_layout(legend=dict(groupclick="toggleitem"))

            # Write to streamlit:
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)


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
