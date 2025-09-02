import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# from utilities_ml.fixed_params import \
#     default_highlighted_team, display_name_of_default_highlighted_team


def main(
        proto_results,
        proto_names,
        hb_teams_input,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        # use_plotly_events, 
        allow_maybe=False,
        prob_maybe_min=0.333,
        prob_maybe_max=0.666,
        ):
    """
    Plot sorted probability bar chart
    """
    n_patients = len(proto_names)
    # Add the bars to the chart in the same order as the highlighted
    # teams list.
    highlighted_teams_list = hb_teams_input  # st.session_state['hb_teams_input']
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']
    # Copy the benchmark colour for the new average benchmark entry:
    bench_key = [k for k in highlighted_teams_colours.keys() if 'Bench' in k][0]
    highlighted_teams_colours['Benchmark average'] = (
        highlighted_teams_colours[bench_key])

    x_vals = np.arange(n_patients)
    bar_width = 0.8 / len(highlighted_teams_list)
    x_offset_min = -bar_width * (len(highlighted_teams_list) - 1) * 0.5
    x_offset_max = -x_offset_min
    x_offsets = np.linspace(x_offset_min, x_offset_max, len(highlighted_teams_list))

    fig = go.Figure()
    for i, leg_entry in enumerate(highlighted_teams_list):
        # Choose the colour of this bar:
        try:
            colour = highlighted_teams_colours[leg_entry]
        except KeyError:
            colour = highlighted_teams_colours[f'{leg_entry}']
        # for p, proto_name in enumerate(proto_names):
        # Take the subset of the big dataframe that contains the data
        # for this highlighted team:
        results_here = proto_results[
            (proto_results['HB team'] == str(leg_entry)) #&
            # (proto_results['Patient prototype'] == proto_name)
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

        # Add bar(s) to the chart for this highlighted team:
        fig.add_trace(go.Bar(
            x=x_vals + x_offsets[i], #results_here['Patient prototype'],
            y=results_here['Probability_perc'],
            # Extra data for hover popup:
            customdata=np.stack([
                name_list,
                results_here['Thrombolyse_str'],
                # results_here['Benchmark']
                ], axis=-1),
            # Name for the legend:
            name=display_name,
            # Set bars colours:
            marker=dict(color=colour)
            ))
        fig.update_traces(width=bar_width)

    # Figure title:
    # Change axis:
    fig.update_yaxes(range=[0.0, 120.0])
    xmax = len(proto_names)
    fig.update_xaxes(range=[-1.0, (xmax+1)*1.2])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=x_vals,  #np.arange(0, proto_results.shape[0], 10),
        ticktext=proto_names,
        ))

    # Update titles and labels:
    fig.update_layout(
        # title='Effect on probability by feature',
        yaxis_title='Probability of giving<br>thrombolysis (%)',
        legend_title='Highlighted team'
        )

    # Hover settings:
    # Make it so cursor can hover over any x value to show the
    # label of the survival line for (x,y), rather than needing to
    # hover directly over the line:
    # fig.update_layout(hovermode='x')
    # Update the information that appears on hover:
    fig.update_traces(
        hovertemplate=(
            # Stroke team:
            'Team %{customdata[0]}' +
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

    # Add horizontal line at borders of decision to thrombolyse:
    if allow_maybe:
        fig.add_hline(y=prob_maybe_min*100.0,
                      line=dict(color='grey'), layer='below')
        fig.add_hline(y=prob_maybe_max*100.0,
                      line=dict(color='grey'), layer='below')
    else:
        fig.add_hline(y=50.0, line=dict(color='grey'), layer='below')
    # # Update y ticks:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 20, 40, 60, 80, 100],
        ))

    # How many teams have thrombolysis yes/maybe/no?

    label_yes = '✔️ would<br>thrombolyse'
    label_maybe = '❓ might<br>thrombolyse'
    label_no = '❌ would not<br>thrombolyse'

    if allow_maybe:
        y_label_yes = np.mean([100.0*prob_maybe_max, 100.0])
        y_label_maybe = np.mean([100.0*prob_maybe_min, 100.0*prob_maybe_max])
        y_label_no = np.mean([0.0, 100.0*prob_maybe_min])
        # Squidge label if necessary:
        if prob_maybe_max > 0.7:
            # label_yes = label_yes.replace('<br>', '')#.replace(' teams', '')
            label_yes = label_yes.replace('<br>thr', ' thr').replace('<br>', ' ').replace(' wou', '<br>wou')
        if prob_maybe_min < 0.3:
            # label_no = label_no.replace('<br>', '')#.replace(' teams', '')
            label_no = label_no.replace('<br>thr', ' thr').replace('<br>', ' ').replace(' wou', '<br>wou')
        if (prob_maybe_max < 0.666) & (prob_maybe_min > 0.333):
            # label_maybe = label_maybe.replace('<br>', ' ')#.replace(' teams', '')
            label_maybe = label_maybe.replace('<br>thr', ' thr').replace('<br>', ' ').replace(' mig', '<br>mig')
    else:
        y_label_yes = np.mean([50.0, 100.0])
        y_label_no = np.mean([0.0, 50.0])
    # Right-hand-side label:
    fig.add_annotation(
        x=n_patients*1.1, y=y_label_yes,
        text=label_yes,
        showarrow=False,
        yshift=0,
        align='center',
        xref='x', yref='y'
        )

    if allow_maybe:
        # Right-hand-side label:
        fig.add_annotation(
            x=n_patients*1.1, y=y_label_maybe,
            text=label_maybe,
            showarrow=False,
            yshift=0,
            align='center',
            xref='x', yref='y'
            )

    # Right-hand-side label:
    fig.add_annotation(
        x=n_patients*1.1, y=y_label_no,
        text=label_no,
        showarrow=False,
        yshift=0,
        align='center',
        xref='x', yref='y'
        )


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
    fig_height = 600 + 20 * ((len(highlighted_teams_list) -2)//3)
    fig.update_layout(
        margin=dict(
            # l=50,
            r=5,
            b=80,
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
        yaxis=dict(fixedrange=True)
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
