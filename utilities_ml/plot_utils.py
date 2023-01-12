"""
Helper functions for plotting.
"""
import plotly.express as px
import matplotlib
import numpy as np
import streamlit as st

from utilities_ml.fixed_params import plain_str, bench_str

# Functions:

def remove_old_colours_for_highlights(highlighted_teams_input):
    # Remove highlighted colours that are no longer needed:
    try:
        highlighted_teams_colours_before = \
            st.session_state['highlighted_teams_colours']
        highlighted_teams_colours = {}
        for team in highlighted_teams_input:
            try:
                highlighted_teams_colours[team] = \
                    highlighted_teams_colours_before[team]
            except KeyError:
                pass
        st.session_state['highlighted_teams_colours'] = highlighted_teams_colours
    except KeyError:
        st.session_state['highlighted_teams_colours'] = {}


def choose_colours_for_highlights(highlighted_teams_list):

    highlighted_teams_colours = \
        st.session_state['highlighted_teams_colours']
    # Specify the indices to get a mix of colours:
    plotly_colours = px.colors.qualitative.Plotly
    # Pick out some colours we prefer (not too close to existing colours):
    inds_preferred = [1, 5, 4, 7, 8, 9, 6, 2, 3, 0]
    preferred_colours = np.array(plotly_colours)[inds_preferred]

    for i, leg_entry in enumerate(highlighted_teams_list):
        try:
            # Check if there's already a designated colour:
            colour = highlighted_teams_colours[leg_entry]
        except KeyError:
            if leg_entry == plain_str:
                colour = 'grey'
            # elif leg_entry == '-':
            #     colour = 'grey'
            elif leg_entry == bench_str:
                colour = 'Navy'
            else:
                # Pick a colour that hasn't already been used.
                unused_colours = list(np.setdiff1d(
                    preferred_colours,
                    list(highlighted_teams_colours.values())
                    ))
                if len(unused_colours) < 1:
                    # Select a colour from this list:
                    mpl_colours = list(matplotlib.colors.cnames.values())
                    colour = list(highlighted_teams_colours.values())[0]
                    while colour in list(highlighted_teams_colours.values()):
                        colour = mpl_colours[
                            np.random.randint(0, len(mpl_colours))]
                else:
                    # np.random.shuffle(unused_colours)
                    # colour = unused_colours[0]
                    # # Take either the first or the last from this list
                    # # (mixes up the reds and blues a bit.)
                    # c_ind = 0 if i % 2 == 0 else -1
                    # colour = unused_colours[c_ind]
                    # preferred_colours = [plotly_colours[1], ]
                    success = 0
                    j = 0
                    while success < 1:
                        colour = preferred_colours[j]
                        if colour in highlighted_teams_colours.values():
                            j += 1
                        else:
                            success = 1

            # Add this to the dictionary:
            highlighted_teams_colours[leg_entry] = colour
        # # Store the colour:
        # # highlighted_teams_colours.append(colour)
        # highlighted_teams_colours[leg_entry] = colour
    # Save the new colour dictionary to the session state:
    st.session_state['highlighted_teams_colours'] = highlighted_teams_colours
