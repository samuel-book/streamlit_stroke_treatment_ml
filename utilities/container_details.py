"""
All of the content for the Details section.
"""
import streamlit as st


def main(animal, feature, row_value):
    st.write('We plotted this equation:')
    st.latex(
        r'''y = \frac{x}{'''+f'{row_value}'+r'''}'''
        )
    st.write(f'where {row_value} comes from {animal} and {feature}.')
