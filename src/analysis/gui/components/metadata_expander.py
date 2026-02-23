import streamlit as st
import pandas as pd


def build(metadata: dict, susi_params: dict) -> None:
    with st.expander("see metadata", expanded=False):
        st.write("metadata")
        st.json(metadata)
        st.write("Susi params")
        st.json(susi_params)
