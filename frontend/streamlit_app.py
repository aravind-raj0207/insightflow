"""InsightFlow dashboard (Streamlit). It only talks to the FastAPI backend over HTTP.

Run:  streamlit run frontend/streamlit_app.py
"""
import html
import json
import os

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="InsightFlow", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1.insightflow-title {margin-bottom: 0; letter-spacing: -0.5px;}
  .insightflow-sub {color: #5b6b68; margin-bottom: 1.2rem;}
  .answer-box {border-left: 4px solid #0F766E; background: #F1F5F4; padding: 0.9rem 1.1rem;
               border-radius: 0 8px 8px 0; font-size: 1.05rem;}
  .warn-box {border-left: 4px solid #B45309; background: #FEF6E7; padding: 0.9rem 1.1rem;
             border-radius: 0 8px 8px 0;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- API calls
def api(method: str, path: str, **kwargs):
    """Call the backend; returns (data, error_message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.ConnectionError:
        return None, f"Cannot reach the InsightFlow backend at {API_URL}. Start it with: uvicorn app.main:app --reload"
    except requests.Timeout:
        return None, "The request took too long. Try a simpler question or a smaller dataset."
    if response.status_code >= 400:
        try:
            return None, response.json().get("detail", "Something went wrong.")
        except ValueError:
            return None, "Something went wrong."
    return response.json(), None


# --------------------------------------------------------------------- state
st.session_state.setdefault("dataset_id", None)
st.session_state.setdefault("turns", {})        # dataset_id -> list of {question, response}
st.session_state.setdefault("conversations", {})  # dataset_id -> conversation_id


def render_answer(turn: dict, expanded_sql: bool = False) -> None:
    response = turn["response"]
    status = response["status"]
    css = "answer-box" if status == "ok" else "warn-box"
    safe_answer = html.escape(response["answer"]).replace("$", "&#36;")   # never render model/data text as HTML
    st.markdown(f'<div class="{css}">{safe_answer}</div>', unsafe_allow_html=True)

    if response.get("figure"):
        figure = pio.from_json(json.dumps(response["figure"]))
        st.plotly_chart(figure, width="stretch", key=f"chart_{turn['id']}")
    if response.get("chart_reason"):
        st.caption(f"📈 Visualization decision: {response['chart_reason']}")

    if response.get("sql"):
        with st.expander("Show SQL", expanded=expanded_sql):
            st.code(response["sql"], language="sql")
            if response.get("resolved_question") and response["resolved_question"] != turn["question"]:
                st.caption(f"Understood as: “{response['resolved_question']}”")
    if response.get("rows"):
        with st.expander(f"Show result data ({response['row_count']:,} rows)"):
            st.dataframe(pd.DataFrame(response["rows"]), width="stretch")


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xlsm"])
    if uploaded is not None and st.session_state.get("last_upload") != (uploaded.name, uploaded.size):
        with st.spinner("Reading, cleaning and storing your data..."):
            data, error = api("POST", "/api/datasets/upload",
                              files={"file": (uploaded.name, uploaded.getvalue())})
        if error:
            st.error(error)
        else:
            st.session_state["last_upload"] = (uploaded.name, uploaded.size)
            st.session_state["dataset_id"] = data["id"]
            st.success(f"Loaded {data['name']}")

    datasets, list_error = api("GET", "/api/datasets")
    if list_error:
        st.error(list_error)
        datasets = []
    if datasets:
        ids = [d["id"] for d in datasets]
        names = {d["id"]: f"{d['name']} (#{d['id']})" for d in datasets}
        if st.session_state["dataset_id"] not in ids:
            st.session_state["dataset_id"] = ids[0]
        st.session_state["dataset_id"] = st.selectbox(
            "Active dataset", ids, index=ids.index(st.session_state["dataset_id"]),
            format_func=lambda i: names[i])
        if st.button("Delete this dataset"):
            _, error = api("DELETE", f"/api/datasets/{st.session_state['dataset_id']}")
            if error:
                st.error(error)
            else:
                st.session_state["dataset_id"] = None
                st.rerun()
    st.divider()
    if st.button("Start a new conversation"):
        st.session_state["conversations"].pop(st.session_state["dataset_id"], None)
        st.session_state["turns"].pop(st.session_state["dataset_id"], None)
        st.rerun()
    st.caption("InsightFlow only reads your data. It can never change or delete it.")

# ---------------------------------------------------------------------- main
st.markdown('<h1 class="insightflow-title">InsightFlow</h1>', unsafe_allow_html=True)
st.markdown('<div class="insightflow-sub">AI-Powered Natural Language Data Analytics</div>', unsafe_allow_html=True)

dataset_id = st.session_state["dataset_id"]
if dataset_id is None:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

dataset, error = api("GET", f"/api/datasets/{dataset_id}")
if error:
    st.error(error)
    st.stop()
profile = dataset["profile"]

st.subheader(f"Dataset: {dataset['name']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['row_count']:,}")
c2.metric("Columns", profile["column_count"])
c3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
c4.metric("Missing values", f"{profile['total_missing']:,}")

cleaning = profile.get("cleaning", {})
for note in cleaning.get("warnings", []):
    st.warning(note)

with st.expander("Data preview and profile", expanded=False):
    tab_preview, tab_columns, tab_stats = st.tabs(["Preview", "Columns", "Numeric statistics"])
    with tab_preview:
        st.dataframe(pd.DataFrame(profile["sample_rows"]), width="stretch")
    with tab_columns:
        st.dataframe(pd.DataFrame([{
            "column": c["name"], "type": c["kind"], "missing": c["missing_count"],
            "missing %": c["missing_pct"], "unique values": c["unique_count"],
        } for c in profile["columns"]]), width="stretch", hide_index=True)
    with tab_stats:
        numeric_rows = [{"column": c["name"], **c["stats"]} for c in profile["columns"] if c["kind"] == "numeric"]
        if numeric_rows:
            st.dataframe(pd.DataFrame(numeric_rows).round(2), width="stretch", hide_index=True)
        else:
            st.write("This dataset has no numeric columns.")
    if cleaning.get("renamed_columns"):
        st.caption("Column names were made database-safe: " + ", ".join(
            f"{r['original']} → {r['new']}" for r in cleaning["renamed_columns"][:12]))

st.divider()
st.subheader("Ask your question")
with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Question", placeholder="e.g. Which city has the highest average rating?",
                             label_visibility="collapsed")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted and question.strip():
    with st.spinner("Analysing your data..."):
        data, error = api("POST", "/api/chat", json={
            "dataset_id": dataset_id, "question": question.strip(),
            "conversation_id": st.session_state["conversations"].get(dataset_id)})
    if error:
        st.error(error)
    else:
        st.session_state["conversations"][dataset_id] = data["conversation_id"]
        turns = st.session_state["turns"].setdefault(dataset_id, [])
        turns.append({"id": len(turns), "question": question.strip(), "response": data})

turns = st.session_state["turns"].get(dataset_id, [])
if turns:
    st.subheader("Answer")
    st.markdown(f"**{turns[-1]['question']}**")
    render_answer(turns[-1], expanded_sql=False)
    if len(turns) > 1:
        st.subheader("Conversation history")
        for turn in reversed(turns[:-1]):
            with st.expander(turn["question"]):
                render_answer(turn)
