import streamlit as st

pages = {
    "Evaluate": [
        st.Page("page/eval/summarization_eval.py", title="Summarization"),
    ],
    "Models & Dataset": [
        st.Page("page/util/upload_dataset.py", title="Upload Dataset"),
    ],
}

st.set_page_config(layout="wide")
pg = st.navigation(pages)
pg.run()