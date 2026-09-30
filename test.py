import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import helper  # noqa: E402
import orchestrator  # noqa: E402
from orchestrator import AnalysisError, MissingAPIKeyError  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "product_name": ["A", "B", "C", "D", "E", "F"],
        "category": ["x", "x", "y", "y", "y", "z"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 1.0],
        "sold_on": pd.to_datetime(
            ["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-11", "2024-03-01", "2024-03-02"]
        ),
    })


@pytest.fixture
def messy_csv(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text(
        " id ,price,discount,rating,Order Date,category\n"
        "1,\"₹1,099\",64%,4.2,2024-01-03,a\n"
        "2,₹399,50%,|,2024-02-03,b\n"
        "2,₹399,50%,|,2024-02-03,b\n"
        "3,,10%,3.5,,\n"
        "4,₹250,20%,4.0,2024-03-03,a\n"
        "5,₹300,30%,4.5,2024-04-03,b\n"
        "6,₹350,35%,3.9,2024-05-03,a\n",
        encoding="utf-8",
    )
    return str(path)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Scriptable stand-in for client.models. `script` items are texts or Exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a fake Gemini client; returns a function to set the scripted replies."""
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: None)

    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(orchestrator, "_client", client)
        return client

    return install


# --------------------------------------------------
# helper.load_data / safe_filename
# --------------------------------------------------

class TestLoadData:
    def test_csv(self, tmp_path):
        p = tmp_path / "a.csv"
        p.write_text("a,b\n1,2\n3,4\n")
        assert helper.load_data(str(p)).shape == (2, 2)

    def test_excel(self, tmp_path):
        p = tmp_path / "a.xlsx"
        pd.DataFrame({"a": [1, 2]}).to_excel(p, index=False)
        assert helper.load_data(str(p)).shape == (2, 1)

    def test_json(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        assert len(helper.load_data(str(p))) == 2

    def test_latin1_csv_fallback(self, tmp_path):
        p = tmp_path / "latin.csv"
        p.write_bytes("name,city\nJosé,Zürich\n".encode("latin-1"))
        assert helper.load_data(str(p)).iloc[0]["name"] == "José"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            helper.load_data("does_not_exist.csv")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            helper.load_data(str(p))

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("a,b\n")
        with pytest.raises(ValueError, match="no data"):
            helper.load_data(str(p))


class TestSafeFilename:
    @pytest.mark.parametrize("raw,expected", [
        ("data.csv", "data.csv"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.csv", "evil.csv"),
        ("my file (1).csv", "my_file__1_.csv"),
        ("", "uploaded_file"),
        (".hidden", "hidden"),
    ])
    def test_cases(self, raw, expected):
        assert helper.safe_filename(raw) == expected


# --------------------------------------------------
# helper cleaning / profiling
# --------------------------------------------------

class TestCleaning:
    def test_currency_and_percent_to_numbers(self):
        s = pd.Series(["₹1,099", "₹399", "$5.50", "64%"])
        out = helper._clean_numeric_series(s)
        assert out.tolist() == [1099.0, 399.0, 5.5, 64.0]

    def test_text_column_is_untouched(self):
        s = pd.Series(["hello", "world", "foo"])
        assert helper._clean_numeric_series(s).tolist() == ["hello", "world", "foo"]

    def test_numeric_series_passthrough(self):
        s = pd.Series([1, 2, 3])
        assert helper._clean_numeric_series(s) is s

    def test_all_empty_series(self):
        s = pd.Series([None, None], dtype="object")
        assert helper._clean_numeric_series(s).isnull().all()

    def test_clean_data_on_messy_file(self, messy_csv):
        df, report = helper.clean_data(helper.load_data(messy_csv))
        assert "id" in df.columns  # header whitespace stripped
        assert report["duplicates_removed"] == 1
        assert report["rows_after"] == report["rows_before"] - 1
        assert pd.api.types.is_numeric_dtype(df["price"])
        assert pd.api.types.is_numeric_dtype(df["discount"])
        assert pd.api.types.is_numeric_dtype(df["rating"])  # '|' coerced then imputed
        assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
        assert df["price"].isnull().sum() == 0
        assert df["category"].isnull().sum() == 0  # mode-filled
        assert report["missing_values_after"] == int(df["Order Date"].isnull().sum())

    def test_unnamed_index_column_dropped(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1], "a": [1, 2]})
        cleaned, _ = helper.clean_data(df)
        assert list(cleaned.columns) == ["a"]

    def test_duplicate_column_names_collapsed(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a ", "b"])
        cleaned, _ = helper.clean_data(df)
        assert list(cleaned.columns) == ["a", "b"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1, None, 1]})
        helper.clean_data(df)
        assert df["a"].isnull().sum() == 1

    def test_high_cardinality_text_filled_with_unknown(self):
        df = pd.DataFrame({"review": [f"text {i}" for i in range(150)] + [None]})
        cleaned, _ = helper.clean_data(df)
        assert cleaned["review"].iloc[-1] == "Unknown"

    def test_all_nan_numeric_column_filled_with_zero(self):
        df = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan]})
        cleaned, _ = helper.clean_data(df)
        assert (cleaned["b"] == 0).all()


class TestProfileAndStats:
    def test_profile(self):
        df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
        p = helper.profile_data(df)
        assert p["rows"] == 3 and p["columns"] == 2
        assert p["missing_values"] == {"a": 1}
        assert p["duplicate_rows"] == 1

    def test_statistics_json_serialisable(self, sales_df):
        stats = helper.get_statistics(sales_df)
        assert set(stats) == {"price", "rating"}
        json.dumps(stats)
        assert stats["price"]["mean"] == 35.0

    def test_statistics_no_numeric_columns(self):
        assert helper.get_statistics(pd.DataFrame({"a": ["x", "y"]})) == {}

    def test_column_information(self, sales_df):
        info = helper.get_column_information(sales_df)
        assert info["category"]["unique_values"] == 3
        assert info["price"]["missing_values"] == 0

    def test_numeric_and_categorical_detection(self, sales_df):
        assert helper.get_numeric_columns(sales_df) == ["price", "rating"]
        assert helper.get_categorical_columns(sales_df) == ["category"]


class TestUIHelpers:
    def test_suggest_questions_use_real_columns(self, sales_df):
        qs = helper.suggest_questions(sales_df)
        assert len(qs) == 3
        assert any("price" in q for q in qs)
        assert any("category" in q for q in qs)

    def test_suggest_questions_without_numeric_columns(self):
        qs = helper.suggest_questions(pd.DataFrame({"a": ["x", "y"]}))
        assert qs == ["How many records are in the dataset?"]

    def test_chart_frame_basic(self):
        f = helper.build_chart_frame({"x_label": "n", "y_label": "v", "data": [{"x": "a", "y": 1}, {"x": "b", "y": 2}]})
        assert f["v"].tolist() == [1, 2]

    def test_chart_frame_duplicate_labels_made_unique(self):
        f = helper.build_chart_frame({"data": [{"x": "same", "y": 1}, {"x": "same", "y": 2}]})
        assert f.index.is_unique and len(f) == 2

    @pytest.mark.parametrize("bad", [None, {}, {"data": []}, {"data": [{"a": 1}]}])
    def test_chart_frame_returns_none_when_not_drawable(self, bad):
        assert helper.build_chart_frame(bad) is None

    def test_chart_frame_non_numeric_y_becomes_zero(self):
        f = helper.build_chart_frame({"data": [{"x": "a", "y": "oops"}]})
        assert f.iloc[0, 0] == 0


# --------------------------------------------------
# orchestrator: config + client
# --------------------------------------------------

class TestConfig:
    def test_default_models(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODELS", raising=False)
        assert orchestrator.get_models() == orchestrator.DEFAULT_MODELS

    def test_models_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODELS", " a-model , b-model ,")
        assert orchestrator.get_models() == ["a-model", "b-model"]

    def test_api_key_prefers_google_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g")
        monkeypatch.setenv("GEMINI_API_KEY", "m")
        assert orchestrator.get_api_key() == "g"

    def test_api_key_falls_back_to_gemini_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "m")
        assert orchestrator.get_api_key() == "m"

    def test_missing_key_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "_client", None)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            orchestrator.get_client()

    def test_importing_orchestrator_needs_no_key(self):
        # Regression: the old module built genai.Client at import time and crashed.
        import subprocess
        env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        env["PYTHONPATH"] = ROOT
        proc = subprocess.run(
            [sys.executable, "-c", "import orchestrator"],
            cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr


# --------------------------------------------------
# orchestrator: parse_llm_json
# --------------------------------------------------

class TestParseJson:
    def test_plain(self):
        assert orchestrator.parse_llm_json('{"operation": "count"}') == {"operation": "count"}

    def test_fenced(self):
        assert orchestrator.parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_without_language(self):
        assert orchestrator.parse_llm_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_text(self):
        assert orchestrator.parse_llm_json('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_single_item_list_unwrapped(self):
        assert orchestrator.parse_llm_json('[{"a": 1}]') == {"a": 1}

    @pytest.mark.parametrize("bad", ["not json at all", "", None, "[1, 2, 3]", '"string"'])
    def test_invalid_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            orchestrator.parse_llm_json(bad)


# --------------------------------------------------
# orchestrator: retry wrapper
# --------------------------------------------------

class TestRetryWrapper:
    def test_success_first_try(self, fake_gemini):
        client = fake_gemini(["  hello  "])
        assert orchestrator.call_gemini_with_retry("hi", models=["m1"]) == "hello"
        assert client.models.calls == ["m1"]

    def test_retries_transient_error_then_succeeds(self, fake_gemini):
        client = fake_gemini([Exception("503 UNAVAILABLE"), "ok"])
        assert orchestrator.call_gemini_with_retry("hi", models=["m1"]) == "ok"
        assert client.models.calls == ["m1", "m1"]

    def test_falls_back_to_next_model_after_retries(self, fake_gemini):
        client = fake_gemini([Exception("429 RESOURCE_EXHAUSTED")] * 3 + ["from m2"])
        assert orchestrator.call_gemini_with_retry("hi", models=["m1", "m2"], max_retries=3) == "from m2"
        assert client.models.calls == ["m1", "m1", "m1", "m2"]

    def test_model_not_found_moves_on_without_retrying(self, fake_gemini):
        client = fake_gemini([Exception("404 NOT_FOUND model"), "ok"])
        assert orchestrator.call_gemini_with_retry("hi", models=["bad", "good"]) == "ok"
        assert client.models.calls == ["bad", "good"]

    def test_auth_error_raised_immediately(self, fake_gemini):
        client = fake_gemini([Exception("400 API_KEY_INVALID")])
        with pytest.raises(Exception, match="API_KEY_INVALID"):
            orchestrator.call_gemini_with_retry("hi", models=["m1", "m2"])
        assert client.models.calls == ["m1"]

    def test_empty_response_treated_as_failure(self, fake_gemini):
        fake_gemini(["", "second model text"])
        assert orchestrator.call_gemini_with_retry("hi", models=["m1", "m2"]) == "second model text"

    def test_none_text_does_not_crash_with_attribute_error(self, fake_gemini, monkeypatch):
        class NoText:
            text = None
        client = fake_gemini(["x"])
        monkeypatch.setattr(client.models, "generate_content", lambda **kw: NoText())
        with pytest.raises(RuntimeError, match="empty response"):
            orchestrator.call_gemini_with_retry("hi", models=["m1"])

    def test_all_models_fail_raises_last_error(self, fake_gemini):
        fake_gemini([Exception("503 UNAVAILABLE")])
        with pytest.raises(Exception, match="503"):
            orchestrator.call_gemini_with_retry("hi", models=["m1", "m2"], max_retries=2)

    def test_config_is_forwarded(self, fake_gemini, monkeypatch):
        seen = {}
        client = fake_gemini(["ok"])

        def gen(model, contents, config=None):
            seen["config"] = config
            return FakeResponse("ok")
        monkeypatch.setattr(client.models, "generate_content", gen)
        orchestrator.call_gemini_with_retry("hi", config={"temperature": 0}, models=["m1"])
        assert seen["config"] == {"temperature": 0}

    def test_default_models_argument_not_shared_state(self):
        import inspect
        assert inspect.signature(orchestrator.call_gemini_with_retry).parameters["models"].default is None


# --------------------------------------------------
# orchestrator: execute_analysis
# --------------------------------------------------

class TestExecuteAnalysis:
    def test_average_and_sum(self, sales_df):
        assert orchestrator.execute_analysis(sales_df, {"operation": "average", "column": "price"})["result"] == 35.0
        assert orchestrator.execute_analysis(sales_df, {"operation": "sum", "column": "price"})["result"] == 210.0

    def test_min_max_numeric_and_text(self, sales_df):
        assert orchestrator.execute_analysis(sales_df, {"operation": "minimum", "column": "price"})["result"] == 10.0
        assert orchestrator.execute_analysis(sales_df, {"operation": "maximum", "column": "category"})["result"] == "z"

    def test_count(self, sales_df):
        assert orchestrator.execute_analysis(sales_df, {"operation": "count"})["result"] == 6

    def test_describe(self, sales_df):
        res = orchestrator.execute_analysis(sales_df, {"operation": "describe"})["result"]
        assert "price" in res and "rating" in res

    def test_column_name_resolved_case_and_underscore_insensitive(self, sales_df):
        for name in ("PRICE", "Price", " price"):
            r = orchestrator.execute_analysis(sales_df, {"operation": "average", "column": name.strip()})
            assert r["column"] == "price"
        df = sales_df.rename(columns={"price": "Sales Amount"})
        r = orchestrator.execute_analysis(df, {"operation": "sum", "column": "sales_amount"})
        assert r["column"] == "Sales Amount"

    def test_unknown_column_gives_helpful_error(self, sales_df):
        with pytest.raises(AnalysisError, match="Available columns"):
            orchestrator.execute_analysis(sales_df, {"operation": "average", "column": "nope"})

    def test_non_numeric_average_rejected(self, sales_df):
        with pytest.raises(AnalysisError, match="not numeric"):
            orchestrator.execute_analysis(sales_df, {"operation": "average", "column": "category"})

    def test_missing_column_in_plan(self, sales_df):
        with pytest.raises(AnalysisError):
            orchestrator.execute_analysis(sales_df, {"operation": "sum"})

    def test_plan_must_be_dict(self, sales_df):
        with pytest.raises(AnalysisError):
            orchestrator.execute_analysis(sales_df, ["average"])

    def test_top_n_descending(self, sales_df):
        r = orchestrator.execute_analysis(sales_df, {"operation": "top_n", "column": "price", "n": 2})
        assert [row["product_name"] for row in r["result"]] == ["F", "E"]
        assert r["chart_data"]["data"][0] == {"x": "F", "y": 60.0}

    def test_top_n_ascending(self, sales_df):
        r = orchestrator.execute_analysis(sales_df, {"operation": "top_n", "column": "rating", "n": 1, "ascending": True})
        assert r["result"][0]["product_name"] == "F"

    @pytest.mark.parametrize("n", ["abc", None, -5, 0, 10**9])
    def test_top_n_bad_n_is_clamped(self, sales_df, n):
        r = orchestrator.execute_analysis(sales_df, {"operation": "top_n", "column": "price", "n": n})
        assert 1 <= len(r["result"]) <= len(sales_df)

    def test_top_n_without_label_columns(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        r = orchestrator.execute_analysis(df, {"operation": "top_n", "column": "a", "n": 2})
        assert [row["a"] for row in r["result"]] == [3, 2]

    def test_top_n_results_are_json_safe(self, sales_df):
        json.dumps(orchestrator.execute_analysis(sales_df, {"operation": "top_n", "column": "price"}))

    def test_group_by_average(self, sales_df):
        r = orchestrator.execute_analysis(
            sales_df, {"operation": "group_by", "group_column": "category", "target_column": "price", "agg": "average"})
        assert r["result"][0] == {"category": "z", "price": 60.0}
        assert len(r["chart_data"]["data"]) == 3

    @pytest.mark.parametrize("agg,expected", [("sum", 120.0), ("min", 30.0), ("max", 50.0), ("median", 40.0)])
    def test_group_by_other_aggs(self, sales_df, agg, expected):
        r = orchestrator.execute_analysis(
            sales_df, {"operation": "group_by", "group_column": "category", "target_column": "price", "agg": agg, "top_n": 10})
        y = next(row for row in r["result"] if row["category"] == "y")
        assert y["price"] == expected

    def test_group_by_count_same_column_regression(self, sales_df):
        # Old code crashed when counting the group column itself.
        r = orchestrator.execute_analysis(
            sales_df, {"operation": "group_by", "group_column": "category", "target_column": "category", "agg": "count"})
        assert r["result"][0] == {"category": "y", "count": 3}

    def test_group_by_respects_top_n(self, sales_df):
        r = orchestrator.execute_analysis(
            sales_df, {"operation": "group_by", "group_column": "category", "target_column": "price", "top_n": 2})
        assert len(r["result"]) == 2

    def test_group_by_non_numeric_target_rejected(self, sales_df):
        with pytest.raises(AnalysisError):
            orchestrator.execute_analysis(
                sales_df, {"operation": "group_by", "group_column": "category", "target_column": "product_name", "agg": "sum"})

    @pytest.mark.parametrize("cond,expected", [(">", 3), ("<", 2), (">=", 4), ("<=", 3), ("==", 1), ("!=", 5)])
    def test_filter_numeric_operators(self, sales_df, cond, expected):
        r = orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "price", "condition": cond, "value": 30})
        assert r["matching_count"] == expected

    def test_filter_sample_capped_at_20(self):
        df = pd.DataFrame({"a": range(100)})
        r = orchestrator.execute_analysis(df, {"operation": "filter", "column": "a", "condition": ">=", "value": 0})
        assert r["matching_count"] == 100 and len(r["result_sample"]) == 20

    def test_filter_text_equals_and_contains(self, sales_df):
        r = orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "category", "condition": "==", "value": "Y"})
        assert r["matching_count"] == 3
        r = orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "product_name", "condition": "contains", "value": "a"})
        assert r["matching_count"] == 1

    def test_filter_date_column(self, sales_df):
        r = orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "sold_on", "condition": ">=", "value": "2024-02-01"})
        assert r["matching_count"] == 4
        json.dumps(r)

    def test_filter_bad_numeric_value(self, sales_df):
        with pytest.raises(AnalysisError, match="not a number"):
            orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "price", "condition": ">", "value": "abc"})

    def test_filter_bad_date_value(self, sales_df):
        with pytest.raises(AnalysisError, match="valid date"):
            orchestrator.execute_analysis(sales_df, {"operation": "filter", "column": "sold_on", "condition": ">", "value": "not a date"})

    def test_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
        assert orchestrator.execute_analysis(df, {"operation": "correlation", "column1": "a", "column2": "b"})["result"] == 1.0
        assert orchestrator.execute_analysis(df, {"operation": "correlation", "column1": "a", "column2": "c"})["result"] == -1.0

    def test_correlation_constant_column_returns_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        assert orchestrator.execute_analysis(df, {"operation": "correlation", "column1": "a", "column2": "b"})["result"] == 0.0

    def test_correlation_non_numeric_rejected(self, sales_df):
        with pytest.raises(AnalysisError):
            orchestrator.execute_analysis(sales_df, {"operation": "correlation", "column1": "price", "column2": "category"})

    def test_time_trend_sum(self, sales_df):
        r = orchestrator.execute_analysis(
            sales_df, {"operation": "time_trend", "date_column": "sold_on", "target_column": "price", "agg": "sum"})
        assert r["result"] == [
            {"period": "2024-01", "price": 30.0},
            {"period": "2024-02", "price": 70.0},
            {"period": "2024-03", "price": 110.0},
        ]
        assert r["chart_data"]["type"] == "line"

    def test_time_trend_count_and_text_dates(self):
        df = pd.DataFrame({"d": ["1/5/2024", "1/9/2024", "2/1/2024"]})
        r = orchestrator.execute_analysis(df, {"operation": "time_trend", "date_column": "d", "agg": "count"})
        assert [row["count"] for row in r["result"]] == [2, 1]

    def test_time_trend_invalid_dates(self, sales_df):
        with pytest.raises(AnalysisError, match="valid dates"):
            orchestrator.execute_analysis(sales_df, {"operation": "time_trend", "date_column": "category", "target_column": "price"})

    def test_unknown_operation_falls_back_to_statistics(self, sales_df):
        r = orchestrator.execute_analysis(sales_df, {"operation": "teleport"})
        assert "message" in r and "price" in r["result"]

    def test_operation_name_case_insensitive(self, sales_df):
        assert orchestrator.execute_analysis(sales_df, {"operation": "COUNT"})["result"] == 6


# --------------------------------------------------
# orchestrator: insight + agent pipeline
# --------------------------------------------------

class TestInsight:
    def test_fallback_summaries(self):
        assert "35" in orchestrator.summarize_result({"operation": "average", "column": "price", "result": 35.0})
        assert "3 rows" in orchestrator.summarize_result({"operation": "top_n", "result": [1, 2, 3]})
        assert "4 records" in orchestrator.summarize_result({"operation": "filter", "matching_count": 4, "column": "x"})
        assert orchestrator.summarize_result({"operation": "describe", "result": {}})

    def test_generate_insight_uses_llm(self, fake_gemini):
        fake_gemini(["The average is 35."])
        assert orchestrator.generate_insight("avg?", {"operation": "average", "result": 35.0}) == "The average is 35."

    def test_insight_payload_excludes_chart_and_is_truncated(self, fake_gemini, monkeypatch):
        seen = {}
        client = fake_gemini(["ok"])

        def gen(model, contents, config=None):
            seen["prompt"] = contents
            return FakeResponse("ok")
        monkeypatch.setattr(client.models, "generate_content", gen)
        big = {"operation": "top_n", "result": ["x" * 100] * 500, "chart_data": {"SECRET_CHART": 1}}
        orchestrator.generate_insight("q", big)
        assert "SECRET_CHART" not in seen["prompt"]
        assert len(seen["prompt"]) < 5000


class TestRunAgent:
    PLAN = json.dumps({"operation": "average", "column": "price"})

    def test_end_to_end(self, fake_gemini, messy_csv):
        fake_gemini([self.PLAN, "Average price is about 483."])
        out = orchestrator.run_agent(messy_csv, "What is the average price?")
        assert out["status"] == "success"
        assert out["analysis_plan"]["operation"] == "average"
        assert out["analysis_result"]["result"] > 0
        assert out["insight"] == "Average price is about 483."
        assert out["insight_source"] == "llm"
        assert out["data_cleaning"]["duplicates_removed"] == 1
        json.dumps(out, default=str)

    def test_insight_failure_keeps_analysis(self, fake_gemini, messy_csv):
        fake_gemini([self.PLAN, Exception("500 INTERNAL")])
        out = orchestrator.run_agent(messy_csv, "average price?")
        assert out["insight_source"] == "fallback"
        assert out["analysis_result"]["operation"] == "average"

    def test_plan_with_markdown_fences(self, fake_gemini, messy_csv):
        fake_gemini(["```json\n" + self.PLAN + "\n```", "fine"])
        assert orchestrator.run_agent(messy_csv, "q")["analysis_plan"]["column"] == "price"

    def test_bad_plan_column_surfaces_clear_error(self, fake_gemini, messy_csv):
        fake_gemini([json.dumps({"operation": "average", "column": "ghost"})])
        with pytest.raises(AnalysisError, match="ghost"):
            orchestrator.run_agent(messy_csv, "q")

    def test_garbage_plan_raises_value_error(self, fake_gemini, messy_csv):
        fake_gemini(["I cannot help with that."])
        with pytest.raises(ValueError):
            orchestrator.run_agent(messy_csv, "q")

    @pytest.mark.parametrize("q", ["", "   ", None])
    def test_empty_question_rejected_before_any_api_call(self, fake_gemini, messy_csv, q):
        client = fake_gemini(["never used"])
        with pytest.raises(ValueError, match="question"):
            orchestrator.run_agent(messy_csv, q)
        assert client.models.calls == []

    def test_missing_file(self, fake_gemini):
        fake_gemini(["x"])
        with pytest.raises(FileNotFoundError):
            orchestrator.run_agent("nope.csv", "q")

    def test_missing_api_key_propagates(self, monkeypatch, messy_csv):
        monkeypatch.setattr(orchestrator, "_client", None)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            orchestrator.run_agent(messy_csv, "q")


# --------------------------------------------------
# Real sample datasets (skipped when files are not present)
# --------------------------------------------------

AMAZON = os.path.join(ROOT, "data", "uploads", "amazon.csv")
SALES = os.path.join(ROOT, "data", "uploads", "sales_100k.csv")


@pytest.mark.skipif(not os.path.exists(AMAZON), reason="amazon.csv not present")
class TestAmazonDataset:
    def test_cleaning(self):
        df, report = helper.clean_data(helper.load_data(AMAZON))
        for col in ("discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"):
            assert pd.api.types.is_numeric_dtype(df[col]), col
        assert report["missing_values_after"] == 0

    def test_analyses(self):
        df, _ = helper.clean_data(helper.load_data(AMAZON))
        avg = orchestrator.execute_analysis(df, {"operation": "average", "column": "discounted_price"})
        assert avg["result"] > 0
        top = orchestrator.execute_analysis(df, {"operation": "top_n", "column": "rating_count", "n": 10})
        assert len(top["result"]) == 10 and top["chart_data"]["data"]
        grp = orchestrator.execute_analysis(
            df, {"operation": "group_by", "group_column": "category", "target_column": "rating", "agg": "average"})
        assert grp["result"]


@pytest.mark.skipif(not os.path.exists(SALES), reason="sales_100k.csv not present")
class TestSalesDataset:
    def test_cleaning_and_trend(self):
        df, _ = helper.clean_data(helper.load_data(SALES))
        assert "Unnamed: 0" not in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Date_of_Sale"])
        r = orchestrator.execute_analysis(
            df, {"operation": "time_trend", "date_column": "Date_of_Sale", "target_column": "Sales_Amount", "agg": "sum"})
        assert len(r["result"]) > 1


# --------------------------------------------------
# UI smoke test (Streamlit AppTest, no network)
# --------------------------------------------------

class TestStreamlitUI:
    def test_app_renders_without_upload(self, monkeypatch):
        from streamlit.testing.v1 import AppTest
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=30).run()
        assert not at.exception
        assert "Autonomous Data Analysis Agent" in at.title[0].value
        assert any("Upload a file" in i.value for i in at.info)
        assert any("No API key" in e.value for e in at.sidebar.error)  # missing key is visible, not a crash

    # ---- full upload -> ask -> analyze flow with a mocked agent ----

    CSV = b"price,rating,category\n10,4.5,a\n20,3.5,b\n30,4.0,a\n40,2.0,b\n"

    @staticmethod
    def _fake_result(question):
        return {
            "status": "success", "question": question,
            "dataset": {"rows": 4, "columns": 3},
            "data_cleaning": {"rows_before": 4, "rows_after": 4, "duplicates_removed": 0,
                              "missing_values_before": 2, "missing_values_after": 0},
            "column_information": {"price": {"data_type": "int64", "unique_values": 4, "missing_values": 0}},
            "analysis_plan": {"operation": "top_n"},
            "analysis_result": {"operation": "top_n", "result": [{"price": 40}],
                                "chart_data": {"type": "bar", "x_label": "n", "y_label": "price",
                                               "data": [{"x": "a", "y": 40}]}},
            "insight": "INSIGHT TEXT", "insight_source": "llm",
        }

    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest
        monkeypatch.chdir(tmp_path)  # uploads land in tmp_path/data/uploads
        calls = []

        def fake_run_agent(path, question):
            calls.append((path, question))
            if "boom" in question:
                raise RuntimeError("kaboom")
            return self._fake_result(question)

        monkeypatch.setattr(orchestrator, "run_agent", fake_run_agent)
        at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=30).run()
        at.file_uploader[0].upload("../my data (1).csv", self.CSV, "text/csv").run()
        assert not at.exception
        return at, calls, tmp_path

    @staticmethod
    def _analyze_button(at):
        return next(b for b in at.button if "Analyze" in b.label)

    def test_upload_is_saved_under_sanitised_name(self, app):
        at, calls, tmp = app
        assert os.listdir(tmp / "data" / "uploads") == ["my_data__1_.csv"]
        assert any("Loaded" in s.value for s in at.success)

    def test_examples_use_uploaded_datasets_columns(self, app):
        at, _, _ = app
        labels = [b.label for b in at.button if b.key and b.key.startswith("sample_")]
        assert labels and all("discounted_price" not in l for l in labels)
        assert any("price" in l for l in labels)

    def test_example_button_fills_question_box(self, app):
        at, _, _ = app
        first = next(b for b in at.button if b.key and b.key.startswith("sample_"))
        label = first.label
        first.click().run()
        assert at.text_area[0].value == label

    def test_analyze_shows_results_and_keeps_question(self, app):
        at, calls, tmp = app
        at.text_area[0].set_value("What is the top price?").run()
        self._analyze_button(at).click().run()
        assert not at.exception
        assert calls and calls[0][1] == "What is the top price?"
        assert calls[0][0].startswith(os.path.join("data", "uploads"))
        assert any("INSIGHT TEXT" in i.value for i in at.info)
        # Regression: the old UI reset the question box after clicking Analyze
        assert at.text_area[0].value == "What is the top price?"

    def test_results_survive_a_rerun(self, app):
        # Regression: results used to vanish on any rerun (e.g. clicking download).
        at, _, _ = app
        at.text_area[0].set_value("question").run()
        self._analyze_button(at).click().run()
        at.run()
        assert any("INSIGHT TEXT" in i.value for i in at.info)

    def test_empty_question_warns_and_does_not_call_agent(self, app):
        at, calls, _ = app
        self._analyze_button(at).click().run()
        assert any("enter a question" in w.value for w in at.warning)
        assert calls == []

    def test_agent_error_is_shown_not_raised(self, app):
        at, _, _ = app
        at.text_area[0].set_value("boom").run()
        self._analyze_button(at).click().run()
        assert not at.exception
        assert any("kaboom" in e.value for e in at.error)

    def test_unreadable_file_shows_error(self, tmp_path, monkeypatch):
        from streamlit.testing.v1 import AppTest
        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=30).run()
        at.file_uploader[0].upload("empty.csv", b"a,b\n", "text/csv").run()
        assert not at.exception
        assert any("Could not read" in e.value for e in at.error)

    def test_app_source_has_no_deprecated_api(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
            assert "use_container_width" not in fh.read()


# --------------------------------------------------
# Optional live Gemini smoke test (replaces test.gemini.py)
# --------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI") != "1" or not orchestrator.get_api_key(),
    reason="set RUN_LIVE_GEMINI=1 and a real API key to run",
)
class TestLiveGemini:
    def test_say_hello(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "_client", None)
        text = orchestrator.call_gemini_with_retry("Say hello")
        assert isinstance(text, str) and text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
