"""
Helper functions for plotting.
"""
import plotly.express as px
import matplotlib
import numpy as np
import streamlit as st


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
    
    for i, leg_entry in enumerate(highlighted_teams_list):
        try:
            # Check if there's already a designated colour:
            colour = highlighted_teams_colours[leg_entry]
        except KeyError:
            if leg_entry == '-':
                colour = 'grey'
            elif 'Benchmark' in leg_entry:
                colour = 'Navy'
            else:
                # Pick a colour that hasn't already been used.
                unused_colours = list(np.setdiff1d(
                    plotly_colours,
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
                    # Take either the first or the last from this list
                    # (mixes up the reds and blues a bit.)
                    c_ind = 0 if i % 2 == 0 else -1
                    colour = unused_colours[c_ind]

            # Add this to the dictionary:
            highlighted_teams_colours[leg_entry] = colour
        # Store the colour:
        # highlighted_teams_colours.append(colour)
        highlighted_teams_colours[leg_entry] = colour
    # Save the new colour dictionary to the session state:
    st.session_state['highlighted_teams_colours'] = highlighted_teams_colours