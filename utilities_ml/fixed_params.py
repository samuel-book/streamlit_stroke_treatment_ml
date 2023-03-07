import streamlit as st

# For drawing a sneaky bar:
import base64

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


def draw_sneaky_bar():
    # Add an invisible bar that's wider than the column:
    file_ = open('./utilities_ml/sneaky_bar.png', "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'''<center><img src="data:image/png;base64,{data_url}" width="500"
            height="1" alt="It's a secret to everybody">''',
        unsafe_allow_html=True,
    )


# Starting probability in the model:
starting_probabilities = 0.2995270168908044

# How to label non-highlighted teams:
plain_str = 'Non-benchmark team'
bench_str = 'Benchmark team: \U00002605'

# Default highlighted team:
default_highlighted_team = 'LECHF1024T'
display_name_of_default_highlighted_team = '"St Elsewhere"'
