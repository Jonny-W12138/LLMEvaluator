import streamlit as st

pages = {
    "Home": [st.Page("page/Homepage.py", title="FineEval")],
    "Evaluate": [
        st.Page("page/eval/summarization_eval.py", title="Summarization"),
        st.Page("page/eval/planning_eval.py", title="Planning"),
        st.Page("page/eval/computation_eval.py", title="Calculation"),
        st.Page("page/eval/custom_eval.py", title="Custom"),
        st.Page("page/eval/reasoning_eval.py", title="Reasoning"),
        st.Page("page/eval/retrieval_eval.py", title="Retrieval") 
    ],
    "Report": [
        st.Page("page/report/summarization_report.py", title="Summarization"),
        st.Page("page/report/planning_report.py", title="Planning"),
        st.Page("page/report/computation_report.py", title="Calculation"),
        st.Page("page/report/custom_report.py", title="Custom"),
        st.Page("page/report/reasoning_report.py", title="Reasoning"),
        st.Page("page/report/retrieval_report.py", title="Retrieval")
    ],
}

st.set_page_config(layout="wide")
pg = st.navigation(pages)
pg.run()