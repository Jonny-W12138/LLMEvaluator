import streamlit as st

pages = {
    "Evaluate": [
        st.Page("page/eval/summarization_eval.py", title="Summarization"),
        st.Page("page/eval/planning_eval.py", title="Planning"),
        st.Page("page/eval/computation_eval.py", title="Computation"),
    ],
    "Report": [
        st.Page("page/report/summarization_report.py", title="Summarization"),
        st.Page("page/report/planning_report.py", title="Planning"),
        st.Page("page/report/computation_report.py", title="Computation"),
    ],
}

st.set_page_config(layout="wide")
pg = st.navigation(pages)
pg.run()