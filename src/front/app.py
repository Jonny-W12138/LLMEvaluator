import streamlit as st

pages = {
    "Evaluate": [
        st.Page("page/eval/summarization_eval.py", title="Summarization"),
    ],
    "Report": [
        st.Page("page/report/summarization_report.py", title="Summarization"),
    ],
}

st.set_page_config(layout="wide")
pg = st.navigation(pages)
pg.run()