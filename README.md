# InsightFlow — AI-Powered Natural Language Data Analytics

Upload a CSV or Excel file, ask questions in plain English, and get an answer, an automatically chosen
interactive chart, and the SQL that produced it. Built with FastAPI, PostgreSQL, LangGraph, Gemini,
Plotly and Streamlit.

## 1. Quick start (Windows PowerShell; on Mac/Linux use `cp` and `source venv/bin/activate`)

You need: Python 3.10+, Docker Desktop (running), and a free Gemini API key
(https://aistudio.google.com/apikey).

```powershell
cd insightflow
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env          # then open .env and paste your key after GEMINI_API_KEY=
docker compose up -d            # starts PostgreSQL (on port 5433)

python scripts/check_setup.py   # checks database + Gemini key and tells you how to fix anything wrong
```

When the check shows no `[FAIL]`, start the two programs in two terminals:

```powershell
uvicorn app.main:app --reload                 # terminal 1: backend  (http://127.0.0.1:8000/docs)
streamlit run frontend/streamlit_app.py       # terminal 2: dashboard (http://localhost:8501)
```
(`start_backend.bat` and `start_frontend.bat` do the same when double-clicked.)

Upload `sample_data/zomato_sample.csv` or `sample_data/sales_sample.xlsx` and ask a question.
Re-create the samples any time with `python scripts/make_sample_data.py`.

### Troubleshooting

| Message | Meaning and fix |
|---|---|
| `password authentication failed for user "insightflow_admin"` | You are reaching a different PostgreSQL, or an old Docker volume has other credentials. Make sure `DB_PORT=5433` in `.env`, then `docker compose down -v` and `docker compose up -d`. |
| `Connection refused` | PostgreSQL is not running. Start Docker Desktop, then `docker compose up -d`. |
| `The AI service is not configured` | `GEMINI_API_KEY` is missing in `.env`. Restart the backend after editing `.env`. |
| `models/... is not found` / 404 from Gemini | The model name is not available for your key or API version. The app now tries to pick a working Flash model by itself (the log says which). To choose one yourself run `python scripts/list_models.py` and put a listed name in `GEMINI_MODEL` in `.env`. |
| `503 UNAVAILABLE ... high demand` | Google's model is overloaded for a moment. The app retries and switches to another Flash model by itself. If you still see "Gemini is busy", wait a minute and ask again. |
| `The AI service is not responding` | Network problem, quota/rate limit, or a wrong `GEMINI_MODEL`. Run `python scripts/check_setup.py` to see Google's exact error. |
| Dashboard says it cannot reach the backend | Start the backend (terminal 1) first. |

`python scripts/check_setup.py` is the first thing to run whenever something does not work.

## 2. What happens when you ask a question

```
Question
  -> load_schema        (deterministic: real columns, types, top values from the database)
  -> resolve_question   (LLM: "What about Chennai?" -> "How many restaurants are in Chennai?")
  -> understand_intent  (LLM + code guards: answerable? refuse data-changing requests? which analysis type?)
        |-- cannot answer / clarify / rejected  --> honest message, no SQL is run
  -> generate_sql       (LLM: one SELECT, only the real table and columns)
  -> validate_sql       (deterministic: sqlglot parse, SELECT only, one table, no dangerous functions, LIMIT)
  -> execute_sql        (deterministic: READ-ONLY database user, read-only transaction, timeout)
        |-- validation/execution error --> back to generate_sql once, with the error message
  -> analyze            (deterministic: min/max/avg/share/trend/correlation computed in Python)
  -> select_chart       (deterministic rules -> ChartSpec; honours an explicit chart request)
  -> render_chart       (Plotly from the real result; falls back to a table if drawing fails)
  -> explain            (LLM: explains the VERIFIED facts; if the LLM fails, the facts are the answer)
```

The LLM only does language work (understand, write SQL, explain). Reading files, profiling, validation,
execution, arithmetic, chart choice and rendering are ordinary tested code.

## 3. Automatic chart selection (`app/visualization/chart_selector.py`)

| Shape of the result | Chart |
|---|---|
| one number | headline value |
| date/time column + number | line |
| category + number | bar (horizontal when many/long labels) |
| category + number, "share of total" question, ≤ 8 categories | pie |
| two numeric columns | scatter |
| one numeric column, many rows | histogram |
| several rows per group | box plot |
| 3+ numeric columns | correlation heatmap |

If the question names a chart ("as a pie chart") it is used when the data allows it. If not (for example a
pie chart of 20 categories) a suitable chart is used and the reason is shown under the chart.
The selector returns a `ChartSpec` (type, columns, title, reason); the renderer never runs LLM-written code.

## 4. Security design

| Threat | Protection |
|---|---|
| "Delete everything" in the question | refused by a code-level guard before any SQL exists |
| The model writes DELETE / DROP / UPDATE / INSERT / ALTER / TRUNCATE / CREATE | SQL is parsed; only a single SELECT is allowed (`sql_validation.py`) |
| `SELECT 1; DROP TABLE x`, comment tricks | exactly one statement, checked on the parsed tree |
| Reading other datasets or system tables | only the active dataset's table is allowed |
| `pg_sleep`, `pg_read_file`, `dblink` ... | blocked function list (`pg_*` prefix is blocked entirely) |
| Hallucinated columns | rejected before execution with the list of real columns, then one self-correction retry |
| Validation bug / bypass | queries run as `insightflow_readonly` (SELECT on `user_data` only), in a read-only transaction |
| Runaway queries | `statement_timeout` (default 10 s) and enforced `LIMIT` (default 1000 rows) |
| Bad uploads | extension, size, row and column limits; safe column names; table names are generated, never user-supplied |
| Secrets | only in `.env` (git-ignored) |
| Model text rendered as HTML | escaped in the dashboard |

## 5. Project layout

```
app/
  main.py                FastAPI app, error handlers, /health
  config.py              settings from .env
  api/                   routes_upload.py, routes_dataset.py, routes_chat.py, schemas.py
  data/                  loader.py (read files), cleaner.py, profiler.py
  database/              connection.py (admin + read-only engines), models.py, repository.py
  llm/                   base.py (interface), gemini_client.py (Gemini via google-genai)
  agent/                 graph.py, state.py, runner.py, nodes/*, prompts/templates.py
  visualization/         chart_selector.py, plotly_renderer.py
  utils/                 errors.py, safety.py, formatting.py, logging_config.py
frontend/streamlit_app.py
db/init.sql              creates schemas + read-only role (the backend also repairs these at startup)
scripts/check_setup.py   diagnoses database + Gemini problems
scripts/list_models.py   lists the Gemini models your key can use
tests/                   101 tests
sample_data/             zomato_sample.csv, sales_sample.xlsx
```

Database: schema `app` holds `datasets`, `dataset_columns`, `conversations`, `messages`, `query_logs`.
Each upload becomes its own table `user_data.ds_<random>`, so datasets never mix.

## 6. Tests

```bash
pytest -v
```
Pure-code tests (cleaning, profiling, SQL validation, chart selection, analysis) always run. Tests marked
`db` need PostgreSQL (`docker compose up -d`) and are skipped automatically otherwise.
The end-to-end tests use a scripted fake LLM (`tests/fake_llm.py`) so they are fast, free and repeatable;
everything after the model is the real code. They cover the eight scenarios from the brief
(most restaurants by city, average rating by city, monthly sales, compare cities, votes vs rating,
missing "profit" column, explicit pie chart, destructive requests) plus follow-ups, retry and isolation.

**Important:** the fake LLM proves the pipeline, not Gemini's quality. Test the real model by asking your own
questions in the dashboard; if answers are poor, improve `app/agent/prompts/templates.py`.

## 7. Extending

* **Another LLM**: write a class with `generate(...)` like `GeminiClient` (see `llm/base.py`) and return it from `get_llm_client()`.
* **Another file format**: add a reader function to `READERS` in `app/data/loader.py`.
* **Model name**: change `GEMINI_MODEL` in `.env`. The default `gemini-flash-latest` follows Google's newest Flash model;
  pin an exact name from https://ai.google.dev/gemini-api/docs/models if you want fixed behaviour.

## 8. Known limits (version 1)

* Excel: first sheet only. Dates are detected from text like `2024-03-15`.
* Results are capped at 1000 rows; distribution/relationship questions use a random sample of up to 1000 rows.
* The dashboard keeps conversation history for the browser session (the backend stores every message).
* No user accounts: all datasets are visible to everyone using the app. Add authentication before sharing it.
