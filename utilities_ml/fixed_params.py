import streamlit as st

# For drawing a sneaky bar:
import base64




def page_setup():
    # ----- Page setup -----
    # The following options set up the display in the tab in your browser.
    # Set page to widescreen must be first call to st.
    st.set_page_config(
        page_title='Thrombolysis decisions',
        page_icon='🔮',
        # layout='wide'
        )
    # n.b. this can be set separately for each separate page if you like.


def draw_sneaky_bar():
    # Add an extra bit to the path if we need to.
    # Try importing something as though we're running this from the same
    # directory as the landing page.
    try:
        file_ = open('./utilities_ml/sneaky_bar.png', "rb")
    except FileNotFoundError:
        # If the import fails, add the landing page directory to path.
        # Assume that the script is being run from the directory above
        # the landing page directory, which is called
        # stroke_outcome_app.
        import sys
        sys.path.append('./streamlit_stroke_treatment_ml/')
        file_ = open('./streamlit_stroke_treatment_ml/utilities_ml/sneaky_bar.png', "rb")
    # Add an invisible bar that's wider than the column:
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'''<center><img src="data:image/png;base64,{data_url}" width="500"
            height="1" alt="It's a secret to everybody">''',
        unsafe_allow_html=True,
    )


def write_markdown_in_colour(string, colour):
    write_str = (
        '<p style="color:' + colour +
        '">' + string + '</p>'
    )
    st.markdown(write_str, unsafe_allow_html=True)



model_version = 'SAMueL-2: August 2023'

stroke_teams_file = 'stroke_teams.csv'
ml_model_file = 'model.p'
explainer_file = 'shap_explainer_probability.p'

# Stroke team column heading for the model
stroke_team_col = 'stroke_team_id'

# Benchmark teams:
benchmark_filename = 'benchmark_codes.csv'
benchmark_team_column = 'stroke_team_id'
n_benchmark_teams = 25

# Default highlighted team:
default_highlighted_team = '42'
display_name_of_default_highlighted_team = (
    str(default_highlighted_team))

# SHAP:
starting_probabilities = 0.3481594278820853
    
# How to label non-highlighted teams:
plain_str = 'Non-benchmark team'
bench_str = 'Benchmark team: \U00002605'