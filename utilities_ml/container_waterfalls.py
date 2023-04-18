"""
All of the content for plotting the red/blue SHAP waterfalls.
"""
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import importlib
# import matplotlib

import utilities_ml.main_calculations

# # For matplotlib plots:
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock

# # Import local package
# from utilities import waterfall
# # Force package to be reloaded
# importlib.reload(waterfall)


# from utilities_ml.fixed_params import \
#     default_highlighted_team, display_name_of_default_highlighted_team
# #     starting_probabilities


def show_waterfalls_max_med_min(
        shap_values_probability_extended_high_mid_low,
        indices_high_mid_low,
        sorted_results,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        model_version='SAMueL-1'
        ):
    # titles = [
    #     'Maximum probability',
    #     'Median probability',
    #     'Minimum probability'
    #     ]
    for i_here, i in enumerate(indices_high_mid_low):
        # Find the data:
        sv_to_display = shap_values_probability_extended_high_mid_low[i_here]
        # Change integer 0/1 to str no/yes for display:
        sv_to_display = utilities_ml.main_calculations.\
            convert_explainer_01_to_noyes(sv_to_display, model_version)

        # Write to streamlit:
        sorted_rank = sorted_results['Sorted rank'].loc[i]
        # Final probability:
        final_prob = sorted_results['Probability'].loc[i]

        # title = '__' + titles[i_here] + ' of thrombolysis__'
        team_info = (
            'Team ' +
            sorted_results['Stroke team'].loc[i] +
            f' (Rank {sorted_rank} of {sorted_results.shape[0]})'
        )
        # st.markdown(title)
        # st.markdown(team_info)
        # Plot:
        plot_shap_waterfall(
            sv_to_display, final_prob,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            team_info, model_version=model_version
        )


def show_waterfalls_highlighted(
        shap_values_probability_extended_highlighted,
        indices_highlighted,
        sorted_results,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        model_version='SAMueL-1'
        ):
    for i_here, i in enumerate(indices_highlighted):
        # Find the data:
        sv_to_display = shap_values_probability_extended_highlighted[i_here]
        # # Change integer 0/1 to str no/yes for display:
        sv_to_display = utilities_ml.main_calculations.\
            convert_explainer_01_to_noyes(sv_to_display, model_version)
        # Final probability:
        final_prob = sorted_results['Probability'].loc[i]

        # Write to streamlit:
        team_name = sorted_results['Stroke team'].loc[i]
        if team_name == default_highlighted_team:
            team_name = display_name_of_default_highlighted_team
        # title = 'Team ' + sorted_results['Stroke team'].loc[i]
        sorted_rank = sorted_results['Sorted rank'].loc[i]
        team_info = (
            'Team ' +
            team_name +
            f' (Rank {sorted_rank} of {sorted_results.shape[0]})'
        )
        # st.markdown(team_info)
        # st.markdown(
        #     f'Rank: {sorted_rank} of {sorted_results.shape[0]}')
        # Plot:
        plot_shap_waterfall(
            sv_to_display,
            final_prob,
            default_highlighted_team,
            display_name_of_default_highlighted_team,
            team_info,
            model_version=model_version
            )


def plot_shap_waterfall(
        shap_values, final_prob,
        default_highlighted_team,
        display_name_of_default_highlighted_team,
        title='', n_to_show=9, model_version='SAMueL-1'):

    # Make lists of all of the features:
    shap_probs = shap_values.values
    feature_names = np.array(shap_values.feature_names)

    # Collect the input patient data:
    patient_data = shap_values.data
    # Adjust some values to add units for showing on the plot:
    if 'SAMueL-1' in model_version:
        extra_bits = {
            'Onset-to-arrival time': ' minutes',
            'Arrival-to-scan time': ' minutes',
            # 'Stroke severity': ' (out of 42)',
            # 'Prior disability level': ' (mRS)',
            'Age': ' years',
        }
    else:
        extra_bits = {
            'onset-to-arrival time': ' minutes',
            'arrival-to-scan time': ' minutes',
            # 'Stroke severity': ' (out of 42)',
            # 'Prior disability level': ' (mRS)',
            'age': ' years',
        }
    for feature in extra_bits.keys():
        i = np.where(feature_names == feature)[0][0]
        patient_data[i] = f'{patient_data[i]}' + extra_bits[feature]

    # Rename the default highlighted team
    i = np.where(feature_names == 'team_' + default_highlighted_team)[0][0]
    feature_names[i] = 'team_' + display_name_of_default_highlighted_team

    # Start probability:
    # base_values = starting_probabilities
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

    features_with_values_waterfall = []
    for i, value in enumerate(patient_data_to_show):
        # Combine the value and the feature name:
        feature_with_value = str(value) + ' = ' + feature_names_to_show[i]
        features_with_values_waterfall.append(feature_with_value)

    y_vals = np.arange(0, -len(feature_names_to_show), -1)
    y_range = [0.5, y_vals[-1] - 0.5]


    fig = go.Figure()
    # Main plot:
    fig.add_trace(go.Waterfall(
        orientation='h',  # horizontal
        measure=measures,
        y=y_vals[::-1],
        x=shap_probs_perc,
        base=base_values_perc,
        decreasing={'marker': {'color': '#008bfa'}},  # Blue bars
        increasing={'marker': {'color': '#ff0050'}},  # Red bars
        hoverinfo='skip'  # Nothing happens when you hover.
    ))


    # For some reason, custom_data needs to be columns rather than rows:
    custom_data = np.stack((shap_probs_perc, patient_data_to_show), axis=-1)
    # Add the custom data:
    fig.update_traces(customdata=custom_data, selector=dict(type='waterfall'))

    # Hover options:
    # # When hovering, show bar at this y value:
    # fig.update_layout(hovermode='y')

    # # Update the hover message with the stroke team:
    # fig.update_traces(hovertemplate=(
    #     # 'Team %{customdata[0]}' +
    #     # '<br>' +
    #     # 'Rank: %{x}' +
    #     # '<br>' +
    #     # 'Feature value: %{customdata[1]}' +
    #     # '<br>' +
    #     'Effect on probability: %{customdata[0]:.2f}%' +
    #     # Need the following line to remove default "trace" bit
    #     # in second "extra" box:
    #     '<extra></extra>'
    #     ))

    # Change axis:
    # fig.update_xaxes(range=[0, 100])

    # Write the size of each bar within the bar:
    fig.update_traces(text=np.round(shap_probs_perc, 2))
    # Set text position to "auto" so it sits inside the bar if there's
    # room, and outside the bar if there's not enough room.
    # For everything inside and auto-size the text, use "inside".
    fig.update_traces(textposition='auto')
    # Explicitly ask for + and - signs:
    fig.update_traces(texttemplate='%{text:+}%')

    # Set axis labels:
    fig.update_xaxes(title_text=' <br>Probability of thrombolysis (%)')
    fig.update_yaxes(title_text='Feature')
    fig.update_layout(title=title)

    # Add start and end prob annotations:
    fig.add_annotation(
        x=base_values_perc,
        y=y_vals[-1]-0.3,
        text=f'Start probability: {base_values_perc:.2f}%',
        showarrow=True,
        yshift=1,
        ax=0  # Make arrow vertical - a = arrow, x = x-shift.
        )
    fig.add_annotation(
        x=final_prob_perc,
        y=y_vals[0]+0.3,
        text=' <br>'+f'End probability: {final_prob_perc:.2f}%',
        showarrow=True,
        # yshift=-100,
        ax=0,  # Make arrow vertical - a = arrow, x = x-shift.
        ay=35,  # Make the arrow sit below the final bar
        )

    # Update x range manually so that bar labels don't get cut off.
    # Find current plotted extent:
    x_values_plotted = np.append(
        base_values_perc, base_values_perc + np.cumsum(shap_probs_perc))
    x_min = np.min(x_values_plotted)
    x_max = np.max(x_values_plotted)
    # Choose a padding value that's a fraction of the current extent:
    x_pad = (x_max - x_min) * 0.15
    fig.update_layout(xaxis=dict(range=[
        x_min - x_pad,
        x_max + x_pad
    ]))

    # Explicitly set some axis information
    # so that the second axis can match it exactly:
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=y_vals,
            ticktext=features_with_values_waterfall[::-1],
            tickfont=dict(color='darkgray'),
            range=y_range
        ),
    )
    # Add empty trace to generate second axis:
    fig.add_trace(go.Scatter(yaxis='y2'))
    # Update second axis:
    fig.update_layout(
        yaxis2=dict(
            tickmode='array',
            tickvals=y_vals,
            ticktext=feature_names_to_show[::-1],
            overlaying='y',
            side='left',
            range=y_range
        ),
    )

    # Add vertical line at starting probability.
    fig.add_vline(
        x=base_values_perc,
        line=dict(color='silver', width=1.0),
        layer='below'  # Puts it below the waterfall
        )
    # Remove y=0 and x=0 lines:
    fig.update_yaxes(zeroline=False)
    fig.update_xaxes(zeroline=False)
    # Remove other vertical grid lines:
    fig.update_xaxes(showgrid=False)

    # Flip y-axis so bars are read from top to bottom.
    # fig['layout']['yaxis']['autorange'] = 'reversed'
    # fig['layout']['yaxis2']['autorange'] = 'reversed'


    # Update margins to prevent annotations being cut off:
    fig.update_layout(margin=dict(r=50))

    # Disable zoom and pan:
    fig.update_layout(
        # Main subplot:
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        # Second subplot for y-axis labels
        xaxis2=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True),
        )

    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)
    # Only change the specific item being clicked on, not the whole
    # legend group:
    # # fig.update_layout(legend=dict(groupclick="toggleitem"))

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

    # Set a specific width, not container width,
    # for the sake of viewing on thin mobile screens where
    # half the plot width is used for the y-axis labels.
    fig.update_layout(width=700)

    # Write to streamlit:
    st.plotly_chart(
        fig,
        # use_container_width=True,
        config=plotly_config
        )
