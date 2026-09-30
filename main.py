import json
import os

import pandas as pd
import streamlit as st

from helper import (
    build_chart_frame,
    clean_data,
    load_data,
    safe_filename,
    suggest_questions,
)
from orchestrator import get_api_key, get_models, run_agent

UPLOAD_DIR = os.path.join("data", "uploads")

st.set_page_config(page_title="Autonomous Data Analysis Agent", page_icon="📊", layout="wide")


# --------------------------------------------------
# CACHED HELPERS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_preview(path, mtime):
    """Load + clean once per file version (mtime is part of the cache key)."""
    cleaned, _ = clean_data(load_data(path))
    return cleaned


def set_question(text):
    st.session_state["question"] = text


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    if get_api_key():
        st.success("Gemini API key detected")
    else:
        st.error("No API key found. Add `GOOGLE_API_KEY` to your `.env` file and restart.")
    st.caption("Models (in fallback order)")
    st.code("\n".join(get_models()), language=None)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload a CSV / Excel / JSON file\n"
        "2. Ask a question in plain English\n"
        "3. The agent plans the analysis, runs it with pandas and explains the result"
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Autonomous Data Analysis Agent")
st.markdown(
    "Upload a dataset and ask analytical questions in plain English. "
    "The LLM only *plans* the analysis; all numbers are computed locally with pandas."
)

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is None:
    st.info("👆 Upload a file to get started.")
    st.stop()

# --------------------------------------------------
# SAVE + PREVIEW FILE
# --------------------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)
file_path = os.path.join(UPLOAD_DIR, safe_filename(uploaded_file.name))
with open(file_path, "wb") as fh:
    fh.write(uploaded_file.getbuffer())

file_id = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.get("file_id") != file_id:
    # New file: drop the previous result and question
    st.session_state["file_id"] = file_id
    st.session_state.pop("result", None)
    st.session_state["question"] = ""

try:
    preview_df = load_preview(file_path, os.path.getmtime(file_path))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read `{uploaded_file.name}`: {exc}")
    st.stop()

st.success(f"Loaded `{uploaded_file.name}` — {len(preview_df):,} rows × {preview_df.shape[1]} columns")

# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

st.markdown("### 💡 Example questions")
suggestions = suggest_questions(preview_df)
for col, text in zip(st.columns(len(suggestions)), suggestions):
    col.button(text, key=f"sample_{text}", on_click=set_question, args=(text,))

st.text_area(
    "Ask a question about your dataset",
    key="question",
    placeholder="Example: Which category has the highest average rating?",
)

if st.button("🚀 Analyze Data", type="primary"):
    question = st.session_state.get("question", "").strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your dataset..."):
            try:
                st.session_state["result"] = run_agent(file_path, question)
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("result", None)
                st.error(f"Error during analysis: {exc}")

# --------------------------------------------------
# RESULTS (kept in session_state so reruns don't erase them)
# --------------------------------------------------

result = st.session_state.get("result")

if result:
    analysis = result.get("analysis_result", {})
    cleaning = result["data_cleaning"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{result['dataset']['rows']:,}")
    m2.metric("Columns", result["dataset"]["columns"])
    m3.metric("Duplicates removed", cleaning["duplicates_removed"])
    m4.metric("Missing values filled", cleaning["missing_values_before"] - cleaning["missing_values_after"])

    tab_insight, tab_chart, tab_quality, tab_data, tab_json = st.tabs(
        ["💡 Insight & Results", "📈 Chart", "🧹 Data quality", "🗂️ Data preview", "🔍 Raw JSON"]
    )

    with tab_insight:
        st.info(result.get("insight") or "No insight generated.")
        if result.get("insight_source") == "fallback":
            st.caption("The AI explanation was unavailable, so a basic summary is shown instead.")

        st.caption(f"Operation: `{analysis.get('operation', 'n/a')}`")
        res = analysis.get("result", analysis.get("result_sample"))

        if analysis.get("matching_count") is not None:
            st.metric("Matching records", f"{analysis['matching_count']:,}")

        if isinstance(res, list):
            st.dataframe(pd.DataFrame(res))
        elif isinstance(res, dict):
            st.dataframe(pd.DataFrame(res).T)
        elif res is not None:
            st.metric(str(analysis.get("column") or analysis.get("operation", "result")).title(), res)

    with tab_chart:
        frame = build_chart_frame(analysis.get("chart_data"))
        if frame is None:
            st.caption("No chart is available for this type of analysis.")
        elif analysis["chart_data"].get("type") == "line":
            st.line_chart(frame)
        else:
            st.bar_chart(frame)

    with tab_quality:
        st.json(cleaning)
        st.dataframe(pd.DataFrame(result["column_information"]).T)

    with tab_data:
        st.dataframe(preview_df.head(100))

    with tab_json:
        st.json(result)

    st.download_button(
        "📥 Download JSON report",
        data=json.dumps(result, indent=4, default=str),
        file_name="analysis_result.json",
        mime="application/json",
    )
