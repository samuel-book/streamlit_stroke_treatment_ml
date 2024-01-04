import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# For clickable plotly events:
from streamlit_plotly_events import plotly_events

# from utilities_ml.fixed_params import \
#     default_highlighted_team, display_name_of_default_highlighted_team


def main(sorted_results, hb_teams_input, use_plotly_events, 
    default_highlighted_team, display_name_of_default_highlighted_team):
    """
    Plot sorted probability bar chart
    """

    # Add the bars to the chart in the same order as the highlighted
    # teams list.
    highlighted_teams_list = hb_teams_input  # st.session_state['hb_teams_input']
    highlighted_teams_colours = st.session_state['highlighted_teams_colours']

    fig = go.Figure()
    for i, leg_entry in enumerate(highlighted_teams_list):
        # Take the subset of the big dataframe that contains the data
        # for this highlighted team:
        results_here = sorted_results[
            sorted_results['HB team'] == leg_entry]
        # Choose the colour of this bar:
        colour = highlighted_teams_colours[leg_entry]

        if leg_entry == default_highlighted_team:
            display_name = display_name_of_default_highlighted_team
            name_list = [display_name_of_default_highlighted_team] * len(results_here['Stroke team'])
        else:
            display_name = leg_entry
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
            x=results_here['Sorted rank'],
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

    # Figure title:
    # Change axis:
    fig.update_yaxes(range=[0.0, 120.0])
    xmax = sorted_results.shape[0]
    fig.update_xaxes(range=[0.0, (xmax+1)*1.2])
    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, sorted_results.shape[0], 10),
        ))

    # Update titles and labels:
    fig.update_layout(
        # title='Effect on probability by feature',
        xaxis_title=f'Rank out of {sorted_results.shape[0]} stroke teams',
        yaxis_title='Probability of giving<br>thrombolysis (%)',
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

    # Add horizontal line at prob=0.5, the decision to thrombolyse:
    fig.add_hline(y=33.3, line=dict(color='black'), layer='below')
    fig.add_hline(y=66.6, line=dict(color='black'), layer='below')
    # # Update y ticks:
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 20, 40, 60, 80, 100],
        ))

    # How many teams have thrombolysis yes/maybe/no?
    n_teams = len(sorted_results)
    n_yes = len(sorted_results[sorted_results['Thrombolyse'] == 2])
    n_maybe = len(sorted_results[sorted_results['Thrombolyse'] == 1])
    n_no = len(sorted_results[sorted_results['Thrombolyse'] == 0])
    # Add vertical lines to split yes/maybe/no teams:
    y_arrows = 110

    # for x in [0.5, n_yes+0.5, n_yes+n_maybe+0.5, n_teams+0.5]:
    #     fig.add_annotation(
    #         x=x, y=y_arrows-10, ax=x, ay=y_arrows+10, text='',
    #         showarrow=True, axref='x', ayref='y', arrowside='none', arrowcolor='black'
    #     )

    # Annotate the yes/maybe/no numbers:
    x_yes = (0.5 * n_yes) + 0.5
    x_maybe = (0.5 * n_maybe) + n_yes + 0.5
    x_no = (0.5 * n_no) + n_yes + n_maybe + 0.5
    y_labels = 140


    # Right-hand-side label:
    fig.add_annotation(
        x=n_teams*1.1, y=np.mean([66.6, 100.0]),
        text=f'{n_yes} team' + ('s' if n_yes != 1 else '') + '<br>✔️ would<br>thrombolyse',
        showarrow=False,
        yshift=0,
        align='center',
        xref='x', yref='y'
        )
    if n_yes > 0:
        # Arrow:
        fig.add_annotation(
            x=0.5, y=y_arrows, ax=n_yes+0.5, ay=y_arrows, text='',
            showarrow=True, axref='x', ayref='y', arrowside='end+start'
        )
        # Arrow label:
        fig.add_annotation(
            x=x_yes, y=y_arrows+10,
            text='✔️',
            showarrow=False,
            yshift=0,
            align='center',
            xref='x', yref='y'
            )
    
    # Right-hand-side label:
    fig.add_annotation(
        x=n_teams*1.1, y=np.mean([33.3, 66.6]),
        text=f'{n_maybe} team' + ('s' if n_maybe != 1 else '') + '<br>❓ might<br>thrombolyse',
        showarrow=False,
        yshift=0,
        align='center',
        xref='x', yref='y'
        )
    if n_maybe > 0:
        # Arrow:
        fig.add_annotation(
            x=n_yes+0.5, y=y_arrows, ax=n_yes+n_maybe+0.5, ay=y_arrows, text='',
            showarrow=True, axref='x', ayref='y', arrowside='end+start'
        )
        # Arrow label:
        fig.add_annotation(
            x=x_maybe, y=y_arrows+10,
            text='❓',
            showarrow=False,
            yshift=0,
            align='center',
            xref='x', yref='y'
            )
    
    # Right-hand-side label:
    fig.add_annotation(
        x=n_teams*1.1, y=np.mean([0.0, 33.3]),
        text=f'{n_no} team' + ('s' if n_no != 1 else '') + '<br>❌ would not<br>thrombolyse',
        showarrow=False,
        yshift=0,
        align='center',
        xref='x', yref='y'
        )
    if n_no > 0:
        fig.add_annotation(
            x=n_yes+n_maybe+0.5, y=y_arrows, ax=n_yes+n_maybe+n_no+0.5, ay=y_arrows, text='',
            showarrow=True, axref='x', ayref='y', arrowside='end+start'
        )
        # Arrow label:
        fig.add_annotation(
            x=x_no, y=y_arrows+10,
            text='❌',
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
    fig_height = 300 + 20 * ((len(highlighted_teams_list) -2)//3)
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
    if use_plotly_events is False:
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
            config=plotly_config)
    else:
        # Clickable version:
        # Write the plot to streamlit, and store the details of the last
        # bar that was clicked:
        selected_bar = plotly_events(
            fig, click_event=True, key='bars', override_height=fig_height, override_width='900%')

        callback_bar(selected_bar, sorted_results,
            default_highlighted_team, display_name_of_default_highlighted_team)
        # return selected_bar


def callback_bar(selected_bar, sorted_results, 
        default_highlighted_team, display_name_of_default_highlighted_team):
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
    try:
        # Pull the details out of the last bar that was changed
        # (moved to or from the "highlighted" list due to being
        # clicked on) out of the session state:
        last_changed_bar = st.session_state['last_changed_bar']
    except KeyError:
        # Invent some nonsense. It doesn't matter whether it matches
        # the default value of selected_bar before anything is clicked.
        last_changed_bar = [0]

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
            st.session_state['last_changed_bar'] = selected_bar.copy()

            # Re-run the script to get immediate feedback in the
            # multiselect input widget and the graph colours:
            st.experimental_rerun()

        except IndexError:
            # Nothing has been clicked yet, so don't change anything.
            pass
