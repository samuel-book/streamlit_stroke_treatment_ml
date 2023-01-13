import streamlit as st

from utilities_ml.fixed_params import page_setup
from utilities_ml.inputs import write_text_from_file

page_setup()

write_text_from_file('pages/text_for_pages/3_Advanced.txt',
                     head_lines_to_skip=2)
