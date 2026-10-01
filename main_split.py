"""Split-screen UI for the Autonomous Data Analysis Agent.

Run with:   streamlit run main_split.py

Left partition  : upload, dataset summary, question box, suggestions, history.
Right partition : answer, chart, table, plan, data quality, data preview.

This file is an alternative front end. It only *imports* helper.py and
orchestrator.py; main.py, orchestrator.py, helper.py, test.py and
.streamlit/config.toml are untouched, and the original UI still works with
`streamlit run main.py`.
"""

import html
import json
import os

import pandas as pd
import streamlit as st

from helper import (
    build_chart_frame,
    clean_data,
    get_column_information,
    load_data,
    safe_filename,
    suggest_questions,
)
from orchestrator import get_api_key, get_models, run_agent

UPLOAD_DIR = os.path.join("data", "uploads")
MAX_HISTORY = 12  # answers kept per dataset
ACCENT = "#D2562B"
PANE_HEIGHT = 770  # px height of each partition; lower it on small screens

st.set_page_config(
    page_title="Autonomous Data Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# STYLE  (colours and fonts come from .streamlit/config.toml)
# --------------------------------------------------

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,800&display=swap');
:root {
  --paper: #F7F3EA; --card: #FFFDF8; --line: #E4DAC5; --ink: #1B2A27;
  --muted: #6B7570; --accent: #D2562B; --deep: #17302B;
  --serif: 'Fraunces', Georgia, serif;
}
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1560px; padding: 1.1rem 1.6rem 1rem; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; margin: 0 0 .75rem; }
.topbar .brand { font-family: var(--serif); font-size: 3.2rem; font-weight: 800;
  color: #FF6A3D; line-height: 1; letter-spacing: -.015em; }
.topbar .brand small { font-family: 'Inter', sans-serif; font-size: .95rem; font-weight: 400;
  letter-spacing: 0; color: #8B9690; margin-left: 1rem; }
.status { display: inline-flex; align-items: center; gap: .5rem; padding: .25rem .85rem;
  border-radius: 999px; font-size: .8rem; color: var(--ink);
  border: 1px solid var(--line); background: var(--card); white-space: nowrap; }
.status i { width: 8px; height: 8px; border-radius: 50%; background: #3E7A5A; display: inline-block; }
.status.off i { background: #B3261E; }

/* the two partitions */
.st-key-left_pane { background: var(--card); }
.st-key-right_pane { background: #FBF8F1; }
.pane-title { font-family: var(--serif); font-size: 1.2rem; color: var(--ink); margin: .1rem 0 .45rem; }
.pane-title.gap { margin-top: 1.3rem; }
.hint { font-size: .8rem; color: var(--muted); line-height: 1.45; }

/* dataset card (left) */
.ds { border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem;
  background: #F7F3EA; margin: .6rem 0 0; }
.ds .fn { font-size: .8rem; color: var(--muted); word-break: break-all; }
.ds .dims { font-family: var(--serif); font-size: 1.2rem; color: var(--ink); margin: .1rem 0; }
.ds .kinds { font-size: .78rem; color: var(--muted); margin-bottom: .5rem; }
.colchip { display: inline-block; margin: 0 .3rem .3rem 0; padding: .08rem .55rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--card);
  font-size: .72rem; color: var(--ink); }

/* question box */
.st-key-sx_question textarea { font-family: var(--serif); font-size: 1.08rem; line-height: 1.5; }
[class*="st-key-sample_"] button, [class*="st-key-hist_"] button {
  width: 100%; justify-content: flex-start; text-align: left; min-height: 0;
  padding: .35rem .9rem; font-size: .85rem; background: transparent;
  border: 1px solid var(--line); color: var(--ink); }
[class*="st-key-sample_"] button:hover, [class*="st-key-hist_"] button:hover {
  border-color: var(--accent); color: var(--accent); }
[class*="st-key-hist_sel_"] button { border-color: var(--accent); background: rgba(210,86,43,.08); }
.st-key-analyze_btn button { width: 100%; }

/* answer (right) */
.qline { font-family: var(--serif); font-size: 1.02rem; color: var(--muted); margin: .3rem 0 .6rem; }
[class*="st-key-insight_"] { background: var(--card); border: 1px solid var(--line);
  border-left: 5px solid var(--accent); border-radius: 16px; padding: .9rem 1.2rem; }
[class*="st-key-insight_"] p { font-family: var(--serif); font-size: 1.25rem; line-height: 1.5; margin: 0; }
.bignum { background: var(--card); border: 1px solid var(--line); border-radius: 18px;
  padding: 1.1rem 1.5rem; margin: .8rem 0; }
.bignum .l { font-size: .85rem; color: var(--muted); }
.bignum .v { font-family: var(--serif); font-size: 3.4rem; line-height: 1.05; color: var(--ink); }
.bignum .v.sm { font-size: 1.8rem; }
.mini { font-size: .85rem; font-weight: 600; color: var(--muted); margin: 1rem 0 .4rem; }
.chip { display: inline-block; padding: .18rem .7rem; border-radius: 999px;
  background: #EFE8D6; color: var(--ink); font-size: .78rem; margin: 0 .4rem .3rem 0; }
.chip.acc { background: rgba(210,86,43,.12); color: #A63F1C; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; padding: .7rem 1rem; }
[data-testid="stMetricValue"] { font-family: var(--serif); }

/* empty state */
.empty { text-align: left; padding: 2.2rem 1rem 1rem; max-width: 34rem; }
.empty .big { font-family: var(--serif); font-size: 1.9rem; color: var(--ink); line-height: 1.2; }
.empty .sub { color: var(--muted); margin: .5rem 0 1.4rem; line-height: 1.55; }
.empty ol { padding-left: 1.2rem; color: var(--ink); line-height: 1.9; }
/* keep text readable inside the cream panes even when Streamlit is in dark mode */
.st-key-left_pane [data-testid="stMarkdownContainer"] p,
.st-key-right_pane [data-testid="stMarkdownContainer"] p,
.st-key-left_pane label, .st-key-right_pane label { color: var(--ink); }
.st-key-left_pane .hint, .st-key-right_pane .hint { color: var(--muted); }
.st-key-left_pane [data-testid="stCaptionContainer"],
.st-key-right_pane [data-testid="stCaptionContainer"] { color: var(--muted); }
.st-key-left_pane [data-testid="stCaptionContainer"] p,
.st-key-right_pane [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stMetric"] * { color: var(--ink); }
.st-key-right_pane [data-baseweb="tab-list"] button { color: var(--muted); }
.st-key-right_pane [data-baseweb="tab-list"] button[aria-selected="true"] { color: var(--accent); }
.st-key-left_pane button [data-testid="stMarkdownContainer"] p,
.st-key-right_pane button [data-testid="stMarkdownContainer"] p { color: inherit; }
.st-key-right_pane [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: inherit; }
.st-key-analyze_btn button [data-testid="stMarkdownContainer"] p { color: #fff; }
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_preview(path, mtime):
    """Load + clean once per file version (mtime is part of the cache key)."""
    cleaned, _ = clean_data(load_data(path))
    return cleaned


def esc(value):
    return html.escape(str(value))


def pane_title(text, gap=False):
    st.markdown(f'<div class="pane-title{" gap" if gap else ""}">{esc(text)}</div>', unsafe_allow_html=True)


def mini_label(text):
    st.markdown(f'<div class="mini">{esc(text)}</div>', unsafe_allow_html=True)


def pretty(name):
    return str(name).replace("_", " ").strip().title()


def shorten(text, limit=64):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt_value(value):
    """Readable number: thousands separators, no trailing zeros."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        text = f"{value:,.2f}" if abs(value) >= 1000 else f"{value:,.4f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value)


def big_number(label, value):
    text = fmt_value(value)
    size = " sm" if len(text) > 14 else ""
    st.markdown(
        f'<div class="bignum"><div class="l">{esc(label)}</div>'
        f'<div class="v{size}">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def to_table(res):
    """Turn an analysis result into a DataFrame, or None if it is a single value."""
    if isinstance(res, list):
        return pd.DataFrame(res)
    if isinstance(res, dict):
        try:
            return pd.DataFrame(res).T
        except ValueError:
            return pd.DataFrame({"value": res})
    return None


def dataset_card(name, df):
    numeric = df.select_dtypes("number").shape[1]
    columns = [str(c) for c in df.columns]
    chips = "".join(f'<span class="colchip">{esc(c)}</span>' for c in columns[:12])
    if len(columns) > 12:
        chips += f'<span class="colchip">+{len(columns) - 12} more</span>'
    st.markdown(
        f'<div class="ds"><div class="fn">{esc(name)}</div>'
        f'<div class="dims">{len(df):,} rows × {df.shape[1]} columns</div>'
        f'<div class="kinds">{numeric} numeric, {df.shape[1] - numeric} other</div>'
        f"{chips}</div>",
        unsafe_allow_html=True,
    )


def select_entry(entry_id):
    st.session_state["sx_selected"] = entry_id


def set_question(text):
    st.session_state["sx_question"] = text


# --------------------------------------------------
# RIGHT PARTITION PIECES
# --------------------------------------------------

def render_welcome():
    st.markdown(
        '<div class="empty"><div class="big">Your answers will appear here.</div>'
        '<div class="sub">Upload a CSV, Excel or JSON file on the left, then ask a '
        "question the way you would ask a colleague. The model only plans the "
        "analysis; every number is computed locally with pandas.</div>"
        "<ol><li>Upload a dataset</li><li>Ask a question</li>"
        "<li>Read the answer, chart and plan</li></ol></div>",
        unsafe_allow_html=True,
    )


def render_ready(df):
    numeric = df.select_dtypes("number").shape[1]
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", df.shape[1])
    c.metric("Numeric columns", numeric)
    st.markdown(
        '<div class="empty"><div class="big">Ask your first question.</div>'
        '<div class="sub">Pick a suggestion or type your own in the question box '
        "on the left, then press Analyze data.</div></div>",
        unsafe_allow_html=True,
    )


def render_result(entry):
    result = entry["result"]
    uid = entry["id"]
    analysis = result.get("analysis_result", {}) or {}
    cleaning = result["data_cleaning"]
    operation = analysis.get("operation", "n/a")

    st.markdown(f'<div class="qline">“{esc(result.get("question", ""))}”</div>', unsafe_allow_html=True)
    with st.container(key=f"insight_{uid}"):
        st.markdown(result.get("insight") or "No insight generated.")
    if result.get("insight_source") == "fallback":
        st.caption("The AI explanation was unavailable, so a basic summary is shown instead.")

    chips_col, download_col = st.columns([3, 2], vertical_alignment="center")
    with chips_col:
        st.markdown(
            f'<span class="chip acc">{esc(pretty(operation))}</span>'
            f'<span class="chip">{result["dataset"]["rows"]:,} rows analysed</span>',
            unsafe_allow_html=True,
        )
    with download_col:
        st.download_button(
            "Download report",
            data=json.dumps(result, indent=4, default=str),
            file_name="analysis_result.json",
            mime="application/json",
            icon=":material/download:",
            key=f"download_{uid}",
        )

    res = analysis.get("result", analysis.get("result_sample"))
    table = to_table(res)
    frame = build_chart_frame(analysis.get("chart_data"))
    chart_meta = analysis.get("chart_data") or {}

    if analysis.get("message"):
        st.caption(analysis["message"])
    if analysis.get("matching_count") is not None:
        big_number("Matching records", analysis["matching_count"])
    if res is not None and table is None:
        big_number(pretty(analysis.get("column") or operation), res)

    if frame is not None:
        mini_label(f"{pretty(chart_meta.get('y_label', 'value'))} by {pretty(chart_meta.get('x_label', 'x'))}")
        if chart_meta.get("type") == "line":
            st.line_chart(frame, color=ACCENT, height=300)
        else:
            st.bar_chart(frame, color=ACCENT, height=300)

    if table is not None:
        mini_label("Results" if analysis.get("matching_count") is None else "Sample of matching rows")
        st.dataframe(table, hide_index=isinstance(res, list), width="stretch")

    tab_plan, tab_quality, tab_json = st.tabs(["How it was computed", "Data quality", "Raw JSON"])

    with tab_plan:
        st.caption("The model only chose what to compute. pandas ran the plan below on your data.")
        st.json(result.get("analysis_plan", {}))

    with tab_quality:
        m1, m2 = st.columns(2)
        m1.metric("Rows", f"{result['dataset']['rows']:,}")
        m2.metric("Columns", result["dataset"]["columns"])
        m3, m4 = st.columns(2)
        m3.metric("Duplicates removed", cleaning["duplicates_removed"])
        m4.metric("Missing values filled", cleaning["missing_values_before"] - cleaning["missing_values_after"])
        mini_label("Columns")
        st.dataframe(pd.DataFrame(result["column_information"]).T, width="stretch")

    with tab_json:
        st.json(result)


def run_analysis(file_path):
    """Run the agent for the current question and store the answer in history."""
    question = st.session_state.get("sx_question", "").strip()
    if not question:
        st.warning("Type a question in the box on the left first.")
        return
    with st.spinner("Reading your data and planning the analysis..."):
        try:
            result = run_agent(file_path, question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error during analysis: {exc}")
            return
    run_id = st.session_state.get("sx_run_id", 0) + 1
    st.session_state["sx_run_id"] = run_id
    history = st.session_state.get("sx_history", [])
    st.session_state["sx_history"] = ([{"id": run_id, "result": result}] + history)[:MAX_HISTORY]
    st.session_state["sx_selected"] = run_id


def current_entry():
    history = st.session_state.get("sx_history", [])
    if not history:
        return None
    wanted = st.session_state.get("sx_selected")
    return next((e for e in history if e["id"] == wanted), history[0])


def render_history():
    history = st.session_state.get("sx_history", [])
    if not history:
        return
    pane_title("Earlier questions", gap=True)
    chosen = current_entry()["id"]
    for entry in history:
        is_chosen = entry["id"] == chosen
        st.button(
            shorten(entry["result"].get("question", "Question")),
            key=f"hist_{'sel_' if is_chosen else ''}{entry['id']}",
            on_click=select_entry,
            args=(entry["id"],),
            icon=":material/chevron_right:" if is_chosen else ":material/history:",
        )


# --------------------------------------------------
# TOP BAR
# --------------------------------------------------

has_key = bool(get_api_key())
st.markdown(
    '<div class="topbar"><div class="brand">Analyst<small>Plain-English questions, pandas answers</small></div>'
    f'<div class="status{"" if has_key else " off"}"><i></i>'
    f'{"Gemini API key detected" if has_key else "No API key: add GOOGLE_API_KEY to .env and restart"}</div></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SPLIT SCREEN
# --------------------------------------------------

left, right = st.columns([5, 8], gap="medium")

file_path = None
preview_df = None
analyze = False
history_slot = None

# ---------- LEFT: upload + question ----------
with left:
    with st.container(key="left_pane", height=PANE_HEIGHT, border=True):
        pane_title("Dataset")
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx", "xls", "json"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(UPLOAD_DIR, safe_filename(uploaded_file.name))
            file_id = f"{uploaded_file.name}:{uploaded_file.size}"
            new_file = st.session_state.get("sx_file_id") != file_id
            if new_file or not os.path.isfile(file_path):
                with open(file_path, "wb") as fh:
                    fh.write(uploaded_file.getbuffer())
            if new_file:
                # New file: drop earlier answers and the question
                st.session_state["sx_file_id"] = file_id
                st.session_state["sx_history"] = []
                st.session_state["sx_selected"] = None
                st.session_state["sx_question"] = ""
            try:
                preview_df = load_preview(file_path, os.path.getmtime(file_path))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read `{uploaded_file.name}`: {exc}")

        if preview_df is None:
            if uploaded_file is None:
                st.markdown(
                    '<p class="hint">Supported files: CSV, Excel and JSON. '
                    "The data is cleaned automatically after upload.</p>",
                    unsafe_allow_html=True,
                )
        else:
            dataset_card(uploaded_file.name, preview_df)

            pane_title("Question", gap=True)
            for i, text in enumerate(suggest_questions(preview_df)):
                st.button(text, key=f"sample_{i}", on_click=set_question, args=(text,))

            st.text_area(
                "Ask a question about your dataset",
                key="sx_question",
                placeholder="Example: Which category has the highest average rating?",
                label_visibility="collapsed",
                height=110,
            )
            analyze = st.button("Analyze data", type="primary", icon=":material/arrow_forward:", key="analyze_btn")
            st.markdown(
                '<p class="hint">Calculations run locally. Gemini sees your question, the column '
                "names, 3 sample rows and the computed result, never the full file.</p>",
                unsafe_allow_html=True,
            )

            history_slot = st.container()  # filled after the analysis has run

        with st.expander("Model fallback order"):
            st.code("\n".join(get_models()), language=None)

# ---------- RIGHT: answers ----------
with right:
    with st.container(key="right_pane", height=PANE_HEIGHT, border=True):
        if preview_df is None:
            render_welcome()
        else:
            tab_answer, tab_data = st.tabs(["Answer", "Data preview"])

            with tab_answer:
                if analyze:
                    run_analysis(file_path)
                entry = current_entry()
                if entry is None:
                    render_ready(preview_df)
                else:
                    render_result(entry)

            with tab_data:
                mini_label("First 100 rows")
                st.dataframe(preview_df.head(100), width="stretch")
                mini_label("Columns")
                st.dataframe(pd.DataFrame(get_column_information(preview_df)).T, width="stretch")

# The history list is drawn last so a brand-new answer shows up immediately.
if history_slot is not None:
    with history_slot:
        render_history()