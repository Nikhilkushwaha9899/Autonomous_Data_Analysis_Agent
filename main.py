import html
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
MAX_HISTORY = 8  # answers kept on screen per dataset
ACCENT = "#D2562B"

st.set_page_config(
    page_title="Autonomous Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# STYLE
# Colours, fonts and radii live in .streamlit/config.toml. This CSS only adds
# the custom layout pieces (hero, section headers, cards, chips).
# --------------------------------------------------

STYLE = """
<style>
:root {
  --paper: #F7F3EA; --card: #FFFDF8; --line: #E4DAC5; --ink: #1B2A27;
  --muted: #6B7570; --accent: #D2562B; --serif: 'Fraunces', Georgia, serif;
}
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem; }

/* hero */
.eyebrow { font-size: .74rem; font-weight: 600; letter-spacing: .18em;
  text-transform: uppercase; color: var(--accent); margin-bottom: .2rem; }
.hero-sub { font-size: 1.12rem; line-height: 1.55; color: var(--muted);
  max-width: 44rem; margin: .2rem 0 .4rem; }
.hero-sub em { color: var(--ink); font-family: var(--serif); }

/* numbered section headers */
.sec { display: flex; align-items: baseline; gap: .75rem; margin: 2.4rem 0 .9rem; }
.sec .n { font-family: var(--serif); color: var(--accent); font-size: .95rem; }
.sec .t { font-family: var(--serif); font-size: 1.55rem; color: var(--ink); }
.sec .rule { flex: 1; height: 1px; background: var(--line); }

/* question composer */
.st-key-composer { background: var(--card); border: 1px solid var(--line);
  border-radius: 20px; padding: 1.3rem 1.4rem 1.1rem;
  box-shadow: 0 14px 34px -22px rgba(27,42,39,.35); }
.st-key-question textarea { font-family: var(--serif); font-size: 1.15rem; line-height: 1.5; }
[class*="st-key-sample_"] button { min-height: 0; padding: .3rem .95rem; font-size: .84rem;
  background: transparent; border: 1px solid var(--line); color: var(--ink); }
[class*="st-key-sample_"] button:hover { border-color: var(--accent); color: var(--accent); }
.try { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 .35rem; }

/* answer */
.qline { font-family: var(--serif); font-size: 1.05rem; color: var(--muted);
  margin: 0 0 .6rem; }
[class*="st-key-insight_"] [data-testid="stAlert"] { border: 1px solid var(--line);
  border-left: 5px solid var(--accent); border-radius: 16px; padding: .5rem .6rem; }
[class*="st-key-insight_"] [data-testid="stAlert"] p { font-family: var(--serif);
  font-size: 1.28rem; line-height: 1.5; }
.bignum { background: var(--card); border: 1px solid var(--line); border-radius: 20px;
  padding: 1.4rem 1.7rem; margin: .9rem 0; }
.bignum .l { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
.bignum .v { font-family: var(--serif); font-size: 3.8rem; line-height: 1.05; color: var(--ink); }
.bignum .v.sm { font-size: 1.9rem; }
.mini { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); margin: 1rem 0 .4rem; }
.chip { display: inline-block; padding: .18rem .7rem; border-radius: 999px;
  background: #EFE8D6; color: var(--ink); font-size: .78rem; margin: 0 .4rem .3rem 0; }
.chip.acc { background: rgba(210,86,43,.12); color: #A63F1C; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; padding: .8rem 1rem; }
[data-testid="stMetricValue"] { font-family: var(--serif); }

/* empty state */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 1rem; margin-top: 1.4rem; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.1rem 1.2rem; }
.step .n { font-family: var(--serif); color: var(--accent); font-size: 1.6rem; }
.step .h { font-family: var(--serif); font-size: 1.1rem; margin: .1rem 0 .3rem; }
.step .p { color: var(--muted); font-size: .92rem; line-height: 1.5; }

/* sidebar */
.brand { font-family: var(--serif); font-size: 1.4rem; color: #F1EBDD; margin: .2rem 0 0; }
.brand-sub { font-size: .8rem; color: #9DB3AC; margin-bottom: .6rem; }
.side-label { font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
  color: #9DB3AC; margin: 1.3rem 0 .4rem; }
.ds { border: 1px solid rgba(241,235,221,.18); border-radius: 14px;
  padding: .9rem 1rem; background: rgba(255,255,255,.04); margin-top: .8rem; }
.ds .fn { font-size: .82rem; color: #9DB3AC; word-break: break-all; }
.ds .dims { font-family: var(--serif); font-size: 1.15rem; margin: .15rem 0 .1rem; color: #F1EBDD; }
.ds .kinds { font-size: .78rem; color: #9DB3AC; margin-bottom: .55rem; }
.colchip { display: inline-block; margin: 0 .3rem .3rem 0; padding: .1rem .55rem;
  border-radius: 999px; border: 1px solid rgba(241,235,221,.2); font-size: .72rem; color: #D8E2DE; }
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


def set_question(text):
    st.session_state["question"] = text


def esc(value):
    return html.escape(str(value))


def section(number, title):
    st.markdown(
        f'<div class="sec"><span class="n">{number}</span>'
        f'<span class="t">{esc(title)}</span><span class="rule"></span></div>',
        unsafe_allow_html=True,
    )


def mini_label(text):
    st.markdown(f'<div class="mini">{esc(text)}</div>', unsafe_allow_html=True)


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


def pretty(name):
    return str(name).replace("_", " ").strip().title()


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
    chips = "".join(f'<span class="colchip">{esc(c)}</span>' for c in columns[:14])
    if len(columns) > 14:
        chips += f'<span class="colchip">+{len(columns) - 14} more</span>'
    st.markdown(
        f'<div class="ds"><div class="fn">{esc(name)}</div>'
        f'<div class="dims">{len(df):,} rows × {df.shape[1]} columns</div>'
        f'<div class="kinds">{numeric} numeric · {df.shape[1] - numeric} other</div>'
        f'{chips}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# RESULT RENDERING
# --------------------------------------------------

def render_result(entry):
    result = entry["result"]
    uid = entry["id"]
    analysis = result.get("analysis_result", {}) or {}
    cleaning = result["data_cleaning"]
    operation = analysis.get("operation", "n/a")

    st.markdown(f'<div class="qline">“{esc(result.get("question", ""))}”</div>', unsafe_allow_html=True)
    with st.container(key=f"insight_{uid}"):
        st.info(result.get("insight") or "No insight generated.", icon=":material/auto_awesome:")
    if result.get("insight_source") == "fallback":
        st.caption("The AI explanation was unavailable, so a basic summary is shown instead.")

    chips_col, download_col = st.columns([4, 1], vertical_alignment="center")
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

    def draw_table():
        mini_label("Results" if analysis.get("matching_count") is None else "Sample of matching rows")
        st.dataframe(table, hide_index=isinstance(res, list), width="stretch")

    def draw_chart():
        mini_label(f"{pretty(chart_meta.get('y_label', 'value'))} by {pretty(chart_meta.get('x_label', 'x'))}")
        if chart_meta.get("type") == "line":
            st.line_chart(frame, color=ACCENT, height=320)
        else:
            st.bar_chart(frame, color=ACCENT, height=320)

    if table is not None and frame is not None:
        left, right = st.columns(2, gap="large")
        with left:
            draw_table()
        with right:
            draw_chart()
    elif table is not None:
        draw_table()
    elif frame is not None:
        draw_chart()

    tab_plan, tab_quality, tab_json = st.tabs(["How it was computed", "Data quality", "Raw JSON"])

    with tab_plan:
        st.caption(
            "The model only chose *what* to compute. pandas ran the plan below on your data."
        )
        st.json(result.get("analysis_plan", {}))

    with tab_quality:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{result['dataset']['rows']:,}")
        m2.metric("Columns", result["dataset"]["columns"])
        m3.metric("Duplicates removed", cleaning["duplicates_removed"])
        m4.metric("Missing values filled", cleaning["missing_values_before"] - cleaning["missing_values_after"])
        mini_label("Columns")
        st.dataframe(pd.DataFrame(result["column_information"]).T, width="stretch")

    with tab_json:
        st.json(result)


# --------------------------------------------------
# SIDEBAR  (upload, dataset card, engine status)
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">Analyst</div>'
        '<div class="brand-sub">Plain-English questions, pandas answers</div>'
        '<div class="side-label">Dataset</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )
    dataset_slot = st.container()  # filled in once the file has been read

    st.markdown('<div class="side-label">Engine</div>', unsafe_allow_html=True)
    if get_api_key():
        st.success("Gemini API key detected")
    else:
        st.error("No API key found. Add `GOOGLE_API_KEY` to your `.env` file and restart.")
    with st.expander("Model fallback order"):
        st.code("\n".join(get_models()), language=None)

# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown('<div class="eyebrow">Autonomous analyst</div>', unsafe_allow_html=True)
st.title("Autonomous Data Analysis Agent")
st.markdown(
    '<p class="hero-sub">Ask a question about your data in plain English. '
    'The model only <em>plans</em> the analysis; every number is computed locally with pandas.</p>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("Upload a file from the sidebar to get started.", icon=":material/upload_file:")
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="n">1</div><div class="h">Upload</div>'
        '<div class="p">Drop in a CSV, Excel or JSON file. It is cleaned automatically.</div></div>'
        '<div class="step"><div class="n">2</div><div class="h">Ask</div>'
        '<div class="p">Type a question the way you would ask a colleague.</div></div>'
        '<div class="step"><div class="n">3</div><div class="h">Read</div>'
        '<div class="p">Get the answer, the table, a chart and the exact plan that produced it.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
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
    # New file: drop earlier answers and the question
    st.session_state["file_id"] = file_id
    st.session_state["history"] = []
    st.session_state["question"] = ""

try:
    preview_df = load_preview(file_path, os.path.getmtime(file_path))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read `{uploaded_file.name}`: {exc}")
    st.stop()

with dataset_slot:
    dataset_card(uploaded_file.name, preview_df)

st.success(f"Loaded `{uploaded_file.name}` — {len(preview_df):,} rows × {preview_df.shape[1]} columns")

with st.expander("Preview the data (first 100 rows)"):
    st.dataframe(preview_df.head(100), width="stretch")

# --------------------------------------------------
# 01 · ASK
# --------------------------------------------------

section("01", "Ask")

with st.container(key="composer"):
    suggestions = suggest_questions(preview_df)
    st.markdown('<div class="try">Try one of these</div>', unsafe_allow_html=True)
    for col, text in zip(st.columns(len(suggestions)), suggestions):
        col.button(text, key=f"sample_{text}", on_click=set_question, args=(text,))

    st.text_area(
        "Ask a question about your dataset",
        key="question",
        placeholder="Example: Which category has the highest average rating?",
        label_visibility="collapsed",
        height=110,
    )

    btn_col, note_col = st.columns([1, 3], vertical_alignment="center")
    analyze = btn_col.button("Analyze data", type="primary", icon=":material/arrow_forward:")
    note_col.caption(
        "Calculations run locally. Gemini sees your question, the column names, "
        "3 sample rows and the computed result, never the full file."
    )

if analyze:
    question = st.session_state.get("question", "").strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Reading your data and planning the analysis..."):
            try:
                result = run_agent(file_path, question)
                run_id = st.session_state.get("run_id", 0) + 1
                st.session_state["run_id"] = run_id
                history = st.session_state.get("history", [])
                st.session_state["history"] = ([{"id": run_id, "result": result}] + history)[:MAX_HISTORY]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error during analysis: {exc}")

# --------------------------------------------------
# 02 · ANSWER  (kept in session_state so reruns don't erase them)
# --------------------------------------------------

history = st.session_state.get("history", [])

if history:
    section("02", "Answer")
    render_result(history[0])

    if len(history) > 1:
        section("03", "Earlier questions")
        for entry in history[1:]:
            label = str(entry["result"].get("question", "Question")).strip()
            with st.expander(label if len(label) <= 90 else label[:87] + "..."):
                render_result(entry)