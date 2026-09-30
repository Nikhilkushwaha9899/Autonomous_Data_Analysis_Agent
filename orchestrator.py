"""LLM planning + pandas execution for the Autonomous Data Analysis Agent."""

import json
import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from helper import (
    clean_data,
    get_column_information,
    get_statistics,
    load_data,
    profile_data,
)

load_dotenv()


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Override with e.g.  GEMINI_MODELS=gemini-3.5-flash,gemini-2.5-flash  in .env
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-latest"]

RETRYABLE_MARKERS = ("503", "500", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "DEADLINE", "INTERNAL")
AUTH_MARKERS = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403")


class MissingAPIKeyError(RuntimeError):
    """Raised when no Gemini API key is configured."""


class AnalysisError(ValueError):
    """Raised when a plan cannot be executed on the dataset."""


def get_models():
    env = os.getenv("GEMINI_MODELS", "")
    models = [m.strip() for m in env.split(",") if m.strip()]
    return models or list(DEFAULT_MODELS)


def get_api_key():
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


_client = None


def get_client():
    """Create the Gemini client lazily so importing this module never needs a key."""
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key:
            raise MissingAPIKeyError(
                "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file."
            )
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


# --------------------------------------------------
# GEMINI RETRY WRAPPER WITH MODEL FALLBACK
# --------------------------------------------------

def call_gemini_with_retry(prompt, config=None, models=None, max_retries=3):
    """
    Call Gemini, retrying transient errors (429/5xx) with backoff and falling
    back to the next model on failure. Auth errors are raised immediately.
    """
    models = list(models) if models else get_models()
    client = get_client()
    last_exception = None

    for model_name in models:
        for attempt in range(max_retries):
            try:
                kwargs = {"model": model_name, "contents": prompt}
                if config:
                    kwargs["config"] = config
                response = client.models.generate_content(**kwargs)
                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise RuntimeError(f"Model {model_name} returned an empty response.")
                return text.strip()
            except Exception as e:  # noqa: BLE001 - we classify by message below
                last_exception = e
                message = str(e)
                if any(m in message for m in AUTH_MARKERS):
                    raise
                if any(m in message for m in RETRYABLE_MARKERS) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                    continue
                break  # not retryable (or retries used up) -> next model

    raise last_exception if last_exception else RuntimeError("No Gemini models configured.")


def parse_llm_json(text):
    """Extract and parse a JSON object from LLM output (handles ``` fences and chatter)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"The model did not return valid JSON: {cleaned[:200]!r}")
        parsed = json.loads(match.group(0))

    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("The analysis plan must be a JSON object.")
    return parsed


# --------------------------------------------------
# ANALYSIS PLANNING
# --------------------------------------------------

def analyze_dataset(df, question):
    columns = {str(c): str(t) for c, t in df.dtypes.items()}
    sample = df.head(3).astype(str).to_dict(orient="records")

    prompt = f"""
You are an autonomous data analysis planning agent.

Dataset columns and types:
{json.dumps(columns)}

First rows (for context only):
{json.dumps(sample, default=str)[:1500]}

User question:
{question}

Determine what type of analysis is required.

Allowed operations and JSON schema formats:

1. "average": {{"operation": "average", "column": "column_name"}}
2. "sum": {{"operation": "sum", "column": "column_name"}}
3. "minimum": {{"operation": "minimum", "column": "column_name"}}
4. "maximum": {{"operation": "maximum", "column": "column_name"}}
5. "count": {{"operation": "count"}}
6. "describe": {{"operation": "describe"}}
7. "top_n": {{"operation": "top_n", "column": "column_name", "n": 10, "ascending": false}}
8. "group_by": {{"operation": "group_by", "group_column": "group_column_name", "target_column": "target_column_name", "agg": "average|sum|count|min|max|median", "top_n": 10}}
9. "filter": {{"operation": "filter", "column": "column_name", "condition": ">|<|>=|<=|==|!=|contains", "value": 50}}
10. "correlation": {{"operation": "correlation", "column1": "col1", "column2": "col2"}}
11. "time_trend": monthly trend of a numeric column over a date column.
    {{"operation": "time_trend", "date_column": "date_col", "target_column": "numeric_col", "agg": "sum|average|count"}}

Return ONLY valid JSON using exact dataset column names.
Do NOT use markdown. Do NOT use ```json. Do NOT include explanations.
"""

    response_text = call_gemini_with_retry(
        prompt=prompt,
        config={"temperature": 0, "response_mime_type": "application/json"},
    )
    return parse_llm_json(response_text)


# --------------------------------------------------
# EXECUTE ANALYSIS
# --------------------------------------------------

def _native(value):
    """Convert numpy/pandas scalars to plain Python types so results are JSON-safe."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _records(frame):
    return [{str(k): _native(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _normalise(name):
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def _make_resolver(df):
    exact = {c.lower(): c for c in df.columns}
    fuzzy = {_normalise(c): c for c in df.columns}

    def resolve(name):
        if name is None or name == "":
            return None
        if name in df.columns:
            return name
        key = str(name).lower()
        return exact.get(key) or fuzzy.get(_normalise(name)) or name

    return resolve


def _require(df, column, numeric=False, label="column"):
    if column is None or column not in df.columns:
        raise AnalysisError(
            f"The {label} {column!r} does not exist in the dataset. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    if numeric and not pd.api.types.is_numeric_dtype(df[column]):
        raise AnalysisError(f"The {label} {column!r} is not numeric, so this analysis cannot be applied to it.")
    return column


def _int_param(plan, key, default, minimum=1, maximum=1000):
    try:
        value = int(plan.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _label_column(df, exclude=None):
    for preferred in ("product_name", "name", "title", "Sales_ID"):
        if preferred in df.columns and preferred != exclude:
            return preferred
    for col in df.columns:
        if col != exclude and not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _display_columns(df, column, extra=()):
    cols = [c for c in (*extra, column) if c in df.columns]
    return list(dict.fromkeys(cols))


AGG_FUNCS = {"average": "mean", "mean": "mean", "sum": "sum", "min": "min", "minimum": "min",
             "max": "max", "maximum": "max", "median": "median", "count": "count"}


def execute_analysis(df, plan):
    if not isinstance(plan, dict):
        raise AnalysisError("The analysis plan must be a dictionary.")

    operation = str(plan.get("operation", "")).lower()
    resolve = _make_resolver(df)
    column = resolve(plan.get("column"))

    if operation in ("average", "sum"):
        _require(df, column, numeric=True)
        value = df[column].mean() if operation == "average" else df[column].sum()
        return {"operation": operation, "column": column,
                "result": round(float(value), 4) if pd.notnull(value) else 0.0}

    if operation in ("minimum", "maximum"):
        _require(df, column)
        value = df[column].min() if operation == "minimum" else df[column].max()
        return {"operation": operation, "column": column, "result": _native(value) if isinstance(value, (int, float, np.number)) else str(value)}

    if operation == "count":
        return {"operation": "count", "result": int(len(df))}

    if operation == "describe":
        return {"operation": "describe", "result": get_statistics(df)}

    if operation == "top_n":
        _require(df, column)
        n = _int_param(plan, "n", 10)
        ascending = bool(plan.get("ascending", False))
        label = _label_column(df, exclude=column)
        sorted_df = df.sort_values(by=column, ascending=ascending).head(n)
        shown = _display_columns(
            df, column,
            extra=[c for c in (label, "category", "discounted_price", "actual_price", "rating", "rating_count") if c],
        )
        records = _records(sorted_df[shown])

        is_num = pd.api.types.is_numeric_dtype(df[column])
        chart_data = None
        if is_num:
            chart_data = {
                "type": "bar",
                "x_label": label or "row",
                "y_label": column,
                "data": [
                    {"x": str(r.get(label, i))[:30], "y": r[column] if r[column] is not None else 0}
                    for i, r in enumerate(records)
                ],
            }
        return {"operation": "top_n", "column": column, "n": n, "ascending": ascending,
                "result": records, "chart_data": chart_data}

    if operation == "group_by":
        group_col = _require(df, resolve(plan.get("group_column")), label="group column")
        agg_name = str(plan.get("agg", "average")).lower()
        agg = AGG_FUNCS.get(agg_name, "mean")
        top_n = _int_param(plan, "top_n", 10)
        ascending = bool(plan.get("ascending", False))

        if agg == "count":
            grouped = df.groupby(group_col).size().reset_index(name="count")
            value_col = "count"
        else:
            target_col = _require(df, resolve(plan.get("target_column")), numeric=True, label="target column")
            if target_col == group_col:
                raise AnalysisError("The group column and target column must be different.")
            grouped = df.groupby(group_col)[target_col].agg(agg).reset_index()
            value_col = target_col

        grouped = grouped.sort_values(by=value_col, ascending=ascending).head(top_n)
        grouped[value_col] = grouped[value_col].round(2)
        records = _records(grouped)
        return {
            "operation": "group_by", "group_column": group_col, "target_column": value_col,
            "agg": agg_name, "result": records,
            "chart_data": {"type": "bar", "x_label": group_col, "y_label": value_col,
                           "data": [{"x": str(r[group_col])[:30], "y": r[value_col] or 0} for r in records]},
        }

    if operation == "filter":
        _require(df, column)
        cond = str(plan.get("condition", ">")).lower()
        raw = plan.get("value", 0)
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise AnalysisError(f"Filter value {raw!r} is not a number for numeric column {column!r}.")
            mask = {">": series > val, "<": series < val, ">=": series >= val, "<=": series <= val,
                    "!=": series != val}.get(cond, series == val)
        elif pd.api.types.is_datetime64_any_dtype(series):
            val = pd.to_datetime(raw, errors="coerce")
            if pd.isnull(val):
                raise AnalysisError(f"Filter value {raw!r} is not a valid date for column {column!r}.")
            mask = {">": series > val, "<": series < val, ">=": series >= val, "<=": series <= val,
                    "!=": series != val}.get(cond, series == val)
        else:
            val = str(raw)
            text = series.astype(str).str.lower()
            if cond == "contains":
                mask = text.str.contains(re.escape(val.lower()), na=False)
            elif cond == "!=":
                mask = text != val.lower()
            else:
                mask = text == val.lower()

        filtered = df[mask]
        label = _label_column(df, exclude=column)
        shown = _display_columns(df, column, extra=[c for c in (label, "category", "discounted_price", "actual_price", "rating") if c])
        return {"operation": "filter", "column": column, "condition": cond, "value": _native(val) if not isinstance(val, pd.Timestamp) else str(val.date()),
                "matching_count": int(len(filtered)), "result_sample": _records(filtered.head(20)[shown])}

    if operation == "correlation":
        col1 = _require(df, resolve(plan.get("column1")), numeric=True, label="first column")
        col2 = _require(df, resolve(plan.get("column2")), numeric=True, label="second column")
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = df[col1].corr(df[col2])
        return {"operation": "correlation", "column1": col1, "column2": col2,
                "result": round(float(corr), 4) if pd.notnull(corr) else 0.0}

    if operation == "time_trend":
        date_col = _require(df, resolve(plan.get("date_column")), label="date column")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
            if parsed.notnull().mean() < 0.8:
                raise AnalysisError(f"The column {date_col!r} does not contain valid dates.")
            dates = parsed
        else:
            dates = df[date_col]

        agg_name = str(plan.get("agg", "sum")).lower()
        agg = AGG_FUNCS.get(agg_name, "sum")
        work = pd.DataFrame({"period": dates.dt.to_period("M").astype(str)})
        if agg == "count":
            trend = work.dropna().groupby("period").size().reset_index(name="count")
            value_col = "count"
        else:
            value_col = _require(df, resolve(plan.get("target_column")), numeric=True, label="target column")
            work[value_col] = df[value_col].values
            trend = work[work["period"] != "NaT"].groupby("period")[value_col].agg(agg).reset_index()
        trend = trend[trend["period"] != "NaT"].sort_values("period")
        trend[value_col] = trend[value_col].round(2)
        records = _records(trend)
        return {"operation": "time_trend", "date_column": date_col, "target_column": value_col,
                "agg": agg_name, "result": records,
                "chart_data": {"type": "line", "x_label": "month", "y_label": value_col,
                               "data": [{"x": r["period"], "y": r[value_col]} for r in records]}}

    return {"operation": operation or "unknown",
            "message": "Unrecognised operation; showing summary statistics instead.",
            "result": get_statistics(df)}


# --------------------------------------------------
# GENERATE INSIGHT
# --------------------------------------------------

def summarize_result(analysis_result):
    """Deterministic fallback insight used when the LLM is unavailable."""
    op = analysis_result.get("operation", "analysis")
    res = analysis_result.get("result", analysis_result.get("result_sample"))
    if analysis_result.get("matching_count") is not None:
        return f"{analysis_result['matching_count']} records match the filter on '{analysis_result.get('column')}'."
    if isinstance(res, (int, float)):
        col = analysis_result.get("column")
        return f"The {op}{' of ' + repr(col) if col else ''} is {res:,}."
    if isinstance(res, list):
        return f"The {op} analysis returned {len(res)} rows; see the table for details."
    return f"The {op} analysis completed; see the detailed results."


def generate_insight(question, analysis_result):
    payload = json.dumps(
        {k: v for k, v in analysis_result.items() if k != "chart_data"}, default=str
    )[:4000]

    prompt = f"""
User question:
{question}

Analysis result (JSON):
{payload}

Explain the result clearly and concisely in two or three sentences.
Highlight key numerical values or key takeaways.
Do not invent any information.
"""
    return call_gemini_with_retry(prompt=prompt)


# --------------------------------------------------
# MAIN AGENT
# --------------------------------------------------

def run_agent(file_path, question):
    if not question or not str(question).strip():
        raise ValueError("Please provide a question.")

    df = load_data(file_path)
    profile = profile_data(df)
    cleaned_df, cleaning_report = clean_data(df)
    if cleaned_df.empty:
        raise ValueError("The dataset has no usable rows after cleaning.")

    column_information = get_column_information(cleaned_df)

    plan = analyze_dataset(cleaned_df, question)
    analysis_result = execute_analysis(cleaned_df, plan)

    # A failed insight must not throw away a successful analysis
    try:
        insight = generate_insight(question, analysis_result)
        insight_source = "llm"
    except MissingAPIKeyError:
        raise
    except Exception:  # noqa: BLE001
        insight = summarize_result(analysis_result)
        insight_source = "fallback"

    return {
        "status": "success",
        "question": question,
        "dataset": {"rows": profile["rows"], "columns": profile["columns"]},
        "data_cleaning": cleaning_report,
        "column_information": column_information,
        "analysis_plan": plan,
        "analysis_result": analysis_result,
        "insight": insight,
        "insight_source": insight_source,
    }