import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# from utilities_ml.fixed_params import \
#     default_highlighted_team, display_name_of_default_highlighted_team


def main(
        outcome_results,
        hb_teams_input,
        proto_name,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        # use_plotly_events,
        ):
    """
    Plot sorted probability bar chart
    """
    bar_labels = ['Independent', 'Dependent', 'Dead']
    n_bars = len(bar_labels)
    # Add the bars to the chart in the same order as the highlighted
    # teams list.
    highlighted_teams_list = hb_teams_input  # st.session_state['hb_teams_input']
    # Remove benchmark entries:
    highlighted_teams_list = [t for t in highlighted_teams_list if
                              (('ench' not in t) | ('average' in t))]
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']
    # Copy the benchmark colour for the new average benchmark entry:
    bench_key = [k for k in highlighted_teams_colours.keys() if 'Bench' in k][0]
    highlighted_teams_colours['Benchmark average'] = (
        highlighted_teams_colours[bench_key])

    x_vals = np.arange(n_bars)
    # Place a gap between the no-treatment and treatment bars:
    # x_vals[int(0.5*len(x_vals)):] += 1
    bar_width = 0.8 / len(highlighted_teams_list)
    x_offset_min = -bar_width * (len(highlighted_teams_list) - 1) * 0.5
    x_offset_max = -x_offset_min
    x_offsets = np.linspace(x_offset_min, x_offset_max, len(highlighted_teams_list))

    # fig = go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=True)
    for i, leg_entry in enumerate(highlighted_teams_list):
        # Choose the colour of this bar:
        try:
            colour = highlighted_teams_colours[leg_entry]
        except KeyError:
            colour = highlighted_teams_colours[f'{leg_entry}']
        # for p, proto_name in enumerate(proto_names):
        # Take the subset of the big dataframe that contains the data
        # for this highlighted team:
        results_here = outcome_results[
            (outcome_results['HB team'] == str(leg_entry)) &
            # (outcome_results['treated'] == t)
            (outcome_results['Patient prototype'] == proto_name)
            ]

        if leg_entry == default_highlighted_team:
            display_name = display_name_of_default_highlighted_team
            name_list = [display_name_of_default_highlighted_team] * len(results_here['Stroke team'])
        else:
            display_name = str(leg_entry)
            name_list = results_here['Stroke team'].copy(deep=True)
            # Update the default highlighted team label when it is
            # not selected as a highlighted team:
            try:
                ind_to_update = np.where(name_list.values == default_highlighted_team)
                name_list.iloc[ind_to_update] = display_name_of_default_highlighted_team
            except IndexError:
                # Nothing to update.
                pass

        dists = [f'{c}_{t}' for t in ['untreated', 'treated']
                 for c in ['independent', 'dependent', 'dead']]
        for t, treated in enumerate(['untreated', 'treated']):
            dists = [f'{c}_{treated}' for c in ['independent', 'dependent', 'dead']]
            # Add bar(s) to the chart for this highlighted team:
            showlegend = True if t == 0 else False

            customdata = np.stack([
                np.array([name_list]*len(dists)),
                ], axis=-1)
            fig.add_trace(go.Bar(
                x=x_vals + x_offsets[i],
                y=100.0*results_here[dists].values.flatten(),
                # Extra data for hover popup:
                customdata=customdata,
                # Name for the legend:
                name=display_name,
                showlegend=showlegend,
                # Set bars colours:
                marker=dict(color=colour),
                ),
                row=1, col=t+1)
            fig.update_traces(width=bar_width)

    # Figure title:
    # Change axis:
    fig.update_yaxes(range=[0.0, 105.0])
    # xmax = len(bar_labels)
    # fig.update_xaxes(range=[-1.0, (xmax)*1.2])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=x_vals,
        ticktext=bar_labels,
        tickangle=90,
        ))
    fig.update_layout(xaxis2=dict(
        tickmode='array',
        tickvals=x_vals,
        ticktext=bar_labels,
        tickangle=90,
        ))

    # Update titles and labels:
    proto_label = f'"{proto_name}" patients'
    fig.update_layout(
        title=f'Discharge disability probability distribution: {proto_label}',
        yaxis_title='Probability of outcome (%)',
        legend_title='Highlighted team',
        xaxis_title='No treatment',
        xaxis2_title='Treated',
        )

    # Hover settings:
    # Make it so cursor can hover over any x value to show the
    # label of the survival line for (x,y), rather than needing to
    # hover directly over the line:
    # fig.update_layout(hovermode='x')
    # Update the information that appears on hover:
    fig.update_traces(
        hovertemplate=(
            # # Patient prototype:
            # '%{customdata[1]}' +
            # '<br>' +
            # Stroke team:
            'Team %{customdata[0]}' +
            '<br>' +
            # Probability to two decimal places:
            '%{y:>.2f}%' +
            '<br>' +
            # # Yes/no whether to thrombolyse:
            # 'Treated: %{customdata[1]}' +
            # '<br>' +
            # # Outcome type:
            # '%{customdata[3]}' +
            # '<br>' +
            # Yes/no whether it's a benchmark team:
            # '%{customdata[2]}'
            '<extra></extra>'
            )
        )

    # # Update y ticks:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 20, 40, 60, 80, 100],
        ))


    # Move legend to bottom
    fig.update_layout(legend=dict(
        orientation='h', #'h',
        yanchor='top',
        y=-0.4,
        # xanchor='right',
        # x=1.03,
        # itemwidth=50
    ))

    # Reduce size of figure by adjusting margins:
    fig_height = 400 + 20 * ((len(highlighted_teams_list) -2)//3)
    fig.update_layout(
        margin=dict(
            # l=50,
            r=5,
            b=0,
            t=20,
            # pad=4
        ),
        height=fig_height,
        width=690
        )
    # fig.update_xaxes(automargin=True)

    # Disable zoom and pan:
    fig.update_layout(
        # Left subplot:
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

    # Write to streamlit:
    # if use_plotly_events is False:
    fig.update_layout(width=700)
    # Non-interactive version:
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
        config=plotly_config
        )

def main2(
        outcome_results,
        leg_entry,
        proto_name,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        # use_plotly_events,
        ):
    """
    Other way round

    Plot sorted probability bar chart
    """
    bar_labels = ['Independent', 'Dependent', 'Dead']
    n_bars = len(bar_labels)
    # Add the bars to the chart in the same order as the highlighted
    # teams list.
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']
    # Copy the benchmark colour for the new average benchmark entry:
    bench_key = [k for k in highlighted_teams_colours.keys() if 'Bench' in k][0]
    highlighted_teams_colours['Benchmark average'] = (
        highlighted_teams_colours[bench_key])

    x_vals = np.arange(n_bars)
    # Place a gap between the no-treatment and treatment bars:
    # x_vals[int(0.5*len(x_vals)):] += 1
    bar_width = 0.4
    x_offsets = [-0.5*bar_width, 0.5*bar_width]

    fig = go.Figure()

    # Choose the colour of this bar:
    try:
        colour = highlighted_teams_colours[leg_entry]
    except KeyError:
        colour = highlighted_teams_colours[f'{leg_entry}']
    # for p, proto_name in enumerate(proto_names):
    # Take the subset of the big dataframe that contains the data
    # for this highlighted team:
    results_here = outcome_results[
        (outcome_results['HB team'] == str(leg_entry)) &
        # (outcome_results['treated'] == t)
        (outcome_results['Patient prototype'] == proto_name)
        ]

    if leg_entry == default_highlighted_team:
        display_name = display_name_of_default_highlighted_team
        name_list = [display_name_of_default_highlighted_team] * len(results_here['Stroke team'])
    else:
        display_name = str(leg_entry)
        name_list = results_here['Stroke team'].copy(deep=True)
        # Update the default highlighted team label when it is
        # not selected as a highlighted team:
        try:
            ind_to_update = np.where(name_list.values == default_highlighted_team)
            name_list.iloc[ind_to_update] = display_name_of_default_highlighted_team
        except IndexError:
            # Nothing to update.
            pass

    marker_pattern_shapes = [None, 'x']
    label_dict = {'untreated': 'No treatment', 'treated': 'Treated'}
    for t, treated in enumerate(['untreated', 'treated']):
        dists = [f'{c}_{treated}' for c in ['independent', 'dependent', 'dead']]
        # Add bar(s) to the chart for this highlighted team:
        # showlegend = True if t == 0 else False

        customdata = np.stack([
            np.array([name_list]*len(dists)),
            ], axis=-1)
        fig.add_trace(go.Bar(
            x=x_vals + x_offsets[t],
            y=100.0*results_here[dists].values.flatten(),
            # Extra data for hover popup:
            customdata=customdata,
            # Name for the legend:
            name=label_dict[treated],
            # showlegend=showlegend,
            # Set bars colours:
            marker=dict(color=colour),
            marker_pattern_shape=marker_pattern_shapes[t]
            ))
        fig.update_traces(width=bar_width)

    # Figure title:
    # Change axis:
    fig.update_yaxes(range=[0.0, 105.0])
    # xmax = len(bar_labels)
    # fig.update_xaxes(range=[-1.0, (xmax)*1.2])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=x_vals,
        ticktext=bar_labels,
        tickangle=90,
        ))
    fig.update_layout(xaxis2=dict(
        tickmode='array',
        tickvals=x_vals,
        ticktext=bar_labels,
        tickangle=90,
        ))

    # Update titles and labels:
    title = (f'Team {display_name}' if display_name != 'Benchmark average'
             else display_name)
    fig.update_layout(
        title=title,
        yaxis_title='Probability of outcome (%)',
        legend_title='Highlighted team',
        xaxis_title='No treatment',
        xaxis2_title='Treated',
        )

    # Hover settings:
    # Make it so cursor can hover over any x value to show the
    # label of the survival line for (x,y), rather than needing to
    # hover directly over the line:
    # fig.update_layout(hovermode='x')
    # Update the information that appears on hover:
    fig.update_traces(
        hovertemplate=(
            # # Patient prototype:
            # '%{customdata[1]}' +
            # '<br>' +
            # Stroke team:
            # 'Team %{customdata[0]}' +
            # '<br>' +
            # Probability to two decimal places:
            '%{y:>.2f}%' +
            '<br>' +
            # # Yes/no whether to thrombolyse:
            # 'Treated: %{customdata[1]}' +
            # '<br>' +
            # # Outcome type:
            # '%{customdata[3]}' +
            # '<br>' +
            # Yes/no whether it's a benchmark team:
            # '%{customdata[2]}'
            '<extra></extra>'
            )
        )

    # # Update y ticks:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 20, 40, 60, 80, 100],
        ))


    # Move legend to bottom
    fig.update_layout(legend=dict(
        orientation='h', #'h',
        yanchor='top',
        y=-0.4,
        # xanchor='right',
        # x=1.03,
        # itemwidth=50
    ))

    # Reduce size of figure by adjusting margins:
    fig_height = 400 + 20 * ((1 -2)//3)
    fig.update_layout(
        margin=dict(
            # l=50,
            r=5,
            b=0,
            t=20,
            # pad=4
        ),
        height=fig_height,
        width=690
        )
    # fig.update_xaxes(automargin=True)

    # Disable zoom and pan:
    fig.update_layout(
        # Left subplot:
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

    # Write to streamlit:
    # if use_plotly_events is False:
    fig.update_layout(width=700)
    # Non-interactive version:
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
        use_container_width=True, 
        config=plotly_config
        )
