# Autonomous Retail Sales Analyst Agent

An **Agentic AI application** for analyzing retail sales data using natural-language queries. The system uses an LLM with LangGraph, LangChain, and Python data-analysis tools to select and perform the required analysis.

## Domain

**Retail Sales Analytics**

The system uses a predefined retail sales dataset with these 9 attributes:

```text
Sales_ID
Product_Category
Sales_Amount
Discount
Sales_Region
Date_of_Sale
Customer_Age
Customer_Gender
Sales_Representative
```

## Main Use Cases

* Overall sales analysis
* Product category performance
* Regional sales analysis
* Discount vs. sales analysis
* Customer age and gender analysis
* Sales representative performance
* Monthly sales trends
* Missing-value and duplicate checking
* Automatic generation of retail insights

## System Architecture

```text
                    USER
                      │
                      ▼
                 ┌─────────┐
                 │ main.py │
                 │Streamlit│
                 └────┬────┘
                      │
                      ▼
            ┌──────────────────┐
            │ orchestrator.py  │
            │                  │
            │ LangGraph        │
            │ LangChain        │
            │ LLM              │
            │ Tool Calling     │
            └────────┬─────────┘
                     │
                     ▼
              ┌─────────────┐
              │  helper.py  │
              │             │
              │ Pandas      │
              │ NumPy       │
              │ Analysis    │
              │ Charts      │
              └──────┬──────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Retail Dataset  │
            │                 │
            │ retail_sales   │
            │     .csv        │
            └────────┬────────┘
                     │
                     ▼
              Analysis Result
                     │
                     ▼
              LLM Interpretation
                     │
                     ▼
               Final Response
```

### Workflow

```text
User Query
    ↓
main.py
    ↓
orchestrator.py
    ↓
Understand Query
    ↓
Select Appropriate Tool
    ↓
helper.py
    ↓
Pandas / NumPy / Visualization
    ↓
LLM Interpretation
    ↓
Final Answer
```

## Project Structure

```text
autonomous-retail-sales-agent/
│
├── venv/
├── data/
│   └── retail_sales.csv
│
├── main.py
├── helper.py
├── orchestrator.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### File Responsibilities

| File               | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `main.py`          | Streamlit UI and application entry point            |
| `helper.py`        | Common retail-analysis functions                    |
| `orchestrator.py`  | LangGraph/LangChain agent workflow and tool calling |
| `data/`            | Retail sales dataset                                |
| `.env`             | API keys and environment variables                  |
| `requirements.txt` | Python dependencies                                 |

## Technology Stack

* **Python**
* **LangGraph** — agent workflow
* **LangChain** — LLM/tool integration
* **OpenAI API** — language model
* **Pandas & NumPy** — data analysis
* **Matplotlib & Seaborn** — visualization
* **Streamlit** — user interface
* **python-dotenv** — environment variables

## Prerequisites

* Python 3.10+
* Git
* OpenAI API key
* Retail Sales CSV dataset

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate

**Windows CMD:**

```bash
venv\Scripts\activate
```

**PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Do not upload `.env` to GitHub.

## Run

```bash
streamlit run main.py
```

## Objective

To develop a **domain-specific Agentic AI system** that autonomously analyzes retail sales data, answers natural-language questions, generates relevant visualizations, and provides meaningful retail sales insights.
