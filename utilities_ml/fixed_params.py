import streamlit as st


def page_setup():
    # ----- Page setup -----
    # The following options set up the display in the tab in your browser.
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='Demo: thrombolysis prediction',
        page_icon=':hospital:',
        # layout='wide'
        )
    # n.b. this can be set separately for each separate page if you like.

# Starting probability in the model:
starting_probabilities = 0.2995270168908044

# How to label non-highlighted teams:
plain_str = 'Non-benchmark team'
bench_str = 'Benchmark team: \U00002605'

# Default highlighted team:
default_highlighted_team = 'LECHF1024T'
display_name_of_default_highlighted_team = '"St Elsewhere"'
