import os
import re

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json")
MISSING_TOKENS = {"nan", "none", "", "null", "n/a", "na"}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data(file_path):
    """Load a CSV, Excel or JSON file into a DataFrame."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")
    elif extension in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif extension == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV, Excel or JSON.")

    if df.empty or df.shape[1] == 0:
        raise ValueError("The uploaded file contains no data.")

    return df


def safe_filename(name):
    """Return a filesystem-safe base name (prevents path traversal on upload)."""
    base = os.path.basename(str(name).replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return base or "uploaded_file"


# --------------------------------------------------
# NUMERIC / DATE CLEANING HELPERS
# --------------------------------------------------

def _clean_numeric_series(series, threshold=0.8):
    """
    Convert text such as '₹1,099', '$5.50' or '64%' to numbers.
    The column is only converted if at least `threshold` of its non-empty
    values parse as numbers; otherwise it is returned unchanged.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    # Plain-Python lists avoid pandas' string dtype re-introducing NaN in .map()
    texts = [None if pd.isna(v) else str(v).strip() for v in series.tolist()]
    non_empty = [t is not None and t.lower() not in MISSING_TOKENS for t in texts]

    if not any(non_empty):
        return series

    stripped = [re.sub(r"[₹$€£,%\s]", "", t) if t is not None else None for t in texts]
    numeric = pd.Series(pd.to_numeric(pd.Series(stripped, dtype="object"), errors="coerce"), index=series.index)

    ratio = numeric[np.array(non_empty)].notnull().mean()
    return numeric if ratio >= threshold else series


def _looks_like_date_column(name):
    return bool(re.search(r"(date|time|timestamp|_at$|^dt_)", str(name), re.IGNORECASE))


def _try_parse_dates(series, threshold=0.8):
    """Parse a text column as datetimes when most of its values look like dates."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series
    non_null = series.dropna()
    if non_null.empty:
        return series
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    ratio = parsed[non_null.index].notnull().mean()
    return parsed if ratio >= threshold else series


# --------------------------------------------------
# PROFILE DATA
# --------------------------------------------------

def profile_data(df):
    """Generate basic information about the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_values": {
            str(c): int(v) for c, v in df.isnull().sum().items() if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

def clean_data(df):
    """
    Clean the dataset and return (cleaned_df, report).

    Steps: normalise column names, drop pandas' 'Unnamed: N' index columns and
    duplicate rows, convert numeric-looking / date-looking text, impute
    missing numeric values with the median and categorical values with the mode.
    Missing dates are left as NaT rather than invented.
    """
    df = df.copy()
    original_rows = len(df)

    # Column names: strip whitespace, drop export-index columns, de-duplicate names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.columns:
        df[col] = _clean_numeric_series(df[col])
        if _looks_like_date_column(col):
            df[col] = _try_parse_dates(df[col])

    missing_before = int(df.isnull().sum().sum())

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            median = series.median()
            df[col] = series.fillna(median if pd.notnull(median) else 0)
        else:
            mode = series.mode()
            few_values = series.nunique() <= 100
            fill = mode.iloc[0] if (few_values and not mode.empty) else "Unknown"
            df[col] = series.fillna(fill)

    report = {
        "rows_before": original_rows,
        "rows_after": int(len(df)),
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_before,
        "missing_values_after": int(df.isnull().sum().sum()),
    }
    return df, report


# --------------------------------------------------
# STATISTICS / COLUMN INFO
# --------------------------------------------------

def get_statistics(df):
    """Descriptive statistics of numeric columns, as JSON-safe floats."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return {}
    desc = numeric_df.describe().to_dict()
    return {
        str(col): {k: (float(v) if pd.notnull(v) else None) for k, v in metrics.items()}
        for col, metrics in desc.items()
    }


def get_column_information(df):
    """Type, unique count and missing count for every column."""
    return {
        str(col): {
            "data_type": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "missing_values": int(df[col].isnull().sum()),
        }
        for col in df.columns
    }


def get_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def get_categorical_columns(df, max_unique=50):
    """Low-cardinality, non-identifier text columns, suitable for group-by questions."""
    limit = min(max_unique, max(2, len(df) // 2))  # near-unique columns are IDs/names, not categories
    return [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and 1 < df[c].nunique() <= limit
    ]


# --------------------------------------------------
# UI HELPERS (pure functions so they can be unit-tested)
# --------------------------------------------------

def suggest_questions(df, limit=3):
    """Example questions built from the dataset's actual columns."""
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)
    suggestions = []

    if numeric:
        suggestions.append(f"What is the average {numeric[0]}?")
    if numeric and categorical:
        suggestions.append(
            f"Which {categorical[0]} has the highest average {numeric[0]}?"
        )
    if numeric:
        suggestions.append(f"Show the top 10 rows by {numeric[-1]}")
    if not suggestions:
        suggestions.append("How many records are in the dataset?")

    return suggestions[:limit]


def build_chart_frame(chart_data):
    """
    Turn the agent's chart_data into a DataFrame indexed by unique labels
    with a single numeric 'value' column, or None if nothing can be drawn.
    """
    if not chart_data or not chart_data.get("data"):
        return None
    frame = pd.DataFrame(chart_data["data"])
    if not {"x", "y"}.issubset(frame.columns):
        return None

    frame["y"] = pd.to_numeric(frame["y"], errors="coerce").fillna(0)
    labels = frame["x"].astype(str)
    # Truncated labels can collide; make them unique so bars are not merged
    dup_counts = labels.groupby(labels).cumcount()
    labels = labels.where(dup_counts == 0, labels + " (" + (dup_counts + 1).astype(str) + ")")
    out = pd.DataFrame({chart_data.get("y_label") or "value": frame["y"].values}, index=labels.values)
    out.index.name = chart_data.get("x_label") or "x"
    return out
