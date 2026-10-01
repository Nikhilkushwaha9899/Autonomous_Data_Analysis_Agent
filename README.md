# Autonomous Data Analysis Agent

Upload a CSV, Excel or JSON file, ask a question in plain English, and get the answer, a table, a chart and a short explanation.

**Key idea:** the LLM only *plans* the analysis. Every number is computed locally with pandas, so results come from your data, not from the model.

## Use Cases (E-Commerce)

Built and tested on an Amazon products dataset (`amazon.csv`) and a 100k-row sales dataset (`sales_100k.csv`).

| Area | Example question |
| ---- | ---------------- |
| Pricing | What is the average discounted_price? |
| Category performance | Which categories have the highest average rating? |
| Popular products | Find the top 10 products by rating_count |
| Discount analysis | Show products with discount_percentage above 60 |
| Price vs. rating | What is the correlation between actual_price and rating? |
| Revenue by category | Which Product_Category has the highest total Sales_Amount? |
| Regional sales | Which Sales_Region has the most sales? |
| Sales trends | Show monthly Sales_Amount trend |
| Sales reps | Who are the top 10 Sales_Representative by sales? |
| Data quality | How many records are in the dataset? (plus an automatic duplicate / missing-value report) |

## System Architecture

```text
                 USER
                   │  upload file + question
                   ▼
        ┌─────────────────────────┐
        │  main.py & main_split.py│  UI, example questions, results, JSON download
        └──────────┬──────────────┘
                   ▼
        ┌─────────────────────┐        ┌───────────────┐
        │   orchestrator.py   │◄──────►│  Gemini API   │
        │  plan → execute →   │        │ (plan+explain)│
        │  explain            │        └───────────────┘
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │      helper.py      │  load, profile, clean, statistics
        │   pandas / NumPy    │
        └──────────┬──────────┘
                   ▼
           data/uploads/*.csv
```

## Agent Workflow

```text
1. Load       read CSV / Excel / JSON
2. Profile    rows, columns, types, missing values, duplicates
3. Clean      drop duplicates, ₹1,099 / 64% → numbers, parse dates, fill missing values
4. Plan       Gemini converts the question into a JSON plan
              e.g. {"operation": "group_by", "group_column": "category", ...}
5. Execute    plan is validated (column exists? numeric?) and run with pandas
6. Explain    Gemini writes a 2–3 sentence insight (built-in summary if the call fails)
7. Display    insight + table + chart + data-quality report + JSON download
```

Supported operations: `average`, `sum`, `minimum`, `maximum`, `count`, `describe`, `top_n`, `group_by`, `filter`, `correlation`, `time_trend`.

Reliability: retry with backoff on rate limits, automatic fallback to the next Gemini model, and clear messages for a missing API key or a wrong column name.

## Structure

```text
├── main.py & main_split.py         Streamlit UI
├── orchestrator.py   Planning + execution + insight
├── helper.py         Loading, cleaning, UI helpers
├── test.py           Test suite
├── data/uploads/     Datasets
├── .env              API key (not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```env
GOOGLE_API_KEY=your_api_key_here
# optional: GEMINI_MODELS=gemini-3.5-flash,gemini-2.5-flash
```

## Run

```bash
 streamlit run main_split.py  
```

## Test

```bash
python test.py
```

Gemini is mocked, so tests run offline without a key. Set `RUN_LIVE_GEMINI=1` to also run a live API check.