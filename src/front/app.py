import streamlit as st

pages = {
    "Evaluate": [
        st.Page("page/eval/summarization_eval.py", title="Summarization"),
        st.Page("page/eval/planning_eval.py", title="Planning"),
    ],
    "Report": [
        st.Page("page/report/summarization_report.py", title="Summarization"),
    ],
}

st.set_page_config(layout="wide")
pg = st.navigation(pages)
pg.run()