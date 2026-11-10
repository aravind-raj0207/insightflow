"""A scripted stand-in for Gemini, so tests run offline, quickly and deterministically.

It plays the role of the language model: for each known question it returns the intent and SQL
that a good model would produce. Everything AFTER the model (validation, execution, analysis,
chart selection, rendering, API) is the real production code.
"""
import json
import re

from app.llm.base import LLMClient

CITIES = ["Bangalore", "Chennai", "Mumbai", "Delhi", "Hyderabad", "Kolkata"]


def _intent(intent_type="ranking", columns=None, answerable=True, reason=None, modification=False):
    return {"answerable": answerable, "reason": reason, "needs_clarification": False,
            "clarifying_question": None, "is_modification_request": modification,
            "intent_type": intent_type, "requested_chart": None, "required_columns": columns or []}


# question fragment -> (intent, sql, chart title)
SCENARIOS: dict[str, tuple[dict, str | None, str]] = {
    "which city has the most restaurants": (
        _intent("ranking", ["city"]),
        "SELECT city, COUNT(*) AS restaurant_count FROM {table} GROUP BY city ORDER BY restaurant_count DESC LIMIT 10",
        "Restaurants by City"),
    "average rating by city": (
        _intent("comparison", ["city", "rating"]),
        "SELECT city, ROUND(AVG(rating)::numeric, 2) AS avg_rating FROM {table} WHERE rating IS NOT NULL "
        "GROUP BY city ORDER BY avg_rating DESC",
        "Average Rating by City"),
    "which city has the highest average rating": (
        _intent("ranking", ["city", "rating"]),
        "SELECT city, ROUND(AVG(rating)::numeric, 2) AS avg_rating FROM {table} WHERE rating IS NOT NULL "
        "GROUP BY city ORDER BY avg_rating DESC LIMIT 5",
        "Top Cities by Average Rating"),
    "how many restaurants are in": (
        _intent("aggregate", ["city"]),
        "SELECT COUNT(*) AS restaurant_count FROM {table} WHERE LOWER(city) = LOWER('{city}')",
        "Restaurant Count"),
    "show monthly sales using a bar chart": (
        _intent("trend", ["date", "sales"]),
        "SELECT DATE_TRUNC('month', date) AS month, SUM(sales) AS total_sales FROM {table} GROUP BY 1 ORDER BY 1",
        "Monthly Sales"),
    "show monthly sales": (
        _intent("trend", ["date", "sales"]),
        "SELECT DATE_TRUNC('month', date) AS month, SUM(sales) AS total_sales FROM {table} GROUP BY 1 ORDER BY 1",
        "Monthly Sales"),
    "compare chennai and bangalore": (
        _intent("comparison", ["city", "rating", "votes"]),
        "SELECT city, COUNT(*) AS restaurant_count, ROUND(AVG(rating)::numeric, 2) AS avg_rating "
        "FROM {table} WHERE LOWER(city) IN ('chennai', 'bangalore') GROUP BY city ORDER BY city",
        "Chennai vs Bangalore"),
    "relationship between votes and rating": (
        _intent("relationship", ["votes", "rating"]),
        "SELECT votes, rating FROM {table} WHERE votes IS NOT NULL AND rating IS NOT NULL ORDER BY RANDOM() LIMIT 1000",
        "Votes vs Rating"),
    "which restaurant has the highest profit": (
        _intent("ranking", ["restaurant_name", "profit"], answerable=False,
                reason="the uploaded dataset does not contain profit or revenue information"),
        None, ""),
    "sales by region": (
        _intent("composition", ["region", "sales"]),
        "SELECT region, SUM(sales) AS total_sales FROM {table} GROUP BY region ORDER BY total_sales DESC",
        "Sales by Region"),
    "empty result": (
        _intent("lookup", ["city"]),
        "SELECT city, rating FROM {table} WHERE city = 'Atlantis'",
        "Nothing"),
    # A malicious / buggy model: harmless-looking question, destructive SQL
    "tricky question": (
        _intent("lookup", ["city"]),
        "SELECT city FROM {table}; DROP TABLE {table}",
        "Tricky"),
    # The model claims it can answer but keeps using a column that does not exist
    "invented column": (
        _intent("ranking", ["city", "profit"]),
        "SELECT city, SUM(profit) AS total_profit FROM {table} GROUP BY city",
        "Invented"),
    "name pattern": (
        _intent("aggregate", ["restaurant_name"]),
        "SELECT COUNT(*) AS n FROM {table} WHERE restaurant_name LIKE '%Restaurant 00%'",
        "Pattern"),
    "wrong column first": (
        _intent("ranking", ["city"]),
        "SELECT city, COUNT(*) AS n FROM {table} GROUP BY city ORDER BY n DESC LIMIT 3",
        "Retry"),
}


class FakeLLM(LLMClient):
    def __init__(self):
        self.calls: list[str] = []
        self._retry_state: dict[str, int] = {}

    # ---- helper
    @staticmethod
    def _find(pattern: str, prompt: str) -> str:
        match = re.search(pattern, prompt)
        return match.group(1) if match else ""

    def _scenario(self, question: str):
        lowered = question.lower()
        for fragment, scenario in SCENARIOS.items():
            if fragment in lowered:
                return scenario
        return None

    def generate(self, prompt, system=None, json_output=False, task="generic") -> str:
        self.calls.append(task)

        if task == "resolve_question":
            question = self._find(r'New question: "(.*?)"\n', prompt)
            lowered = question.lower()
            last_answer = prompt.split("Assistant answered:")[-1]
            last_question = re.findall(r'User asked: "(.*?)"', prompt)[-1]
            if "restaurants are there" in lowered:
                city = next((c for c in CITIES if c in last_answer), CITIES[0])
                question = f"How many restaurants are in {city}?"
            elif lowered.startswith("what about"):
                new_city = question.split()[-1].strip("?")
                old_city = next((c for c in CITIES if c in last_question), None)
                question = last_question.replace(old_city, new_city) if old_city else question
            return json.dumps({"standalone_question": question})

        if task == "intent":
            question = self._find(r'User question: "(.*?)"\n', prompt)
            if "delete" in question.lower():  # never reached: the code-level guard refuses first
                return json.dumps(_intent(modification=True))
            scenario = self._scenario(question)
            return json.dumps(scenario[0] if scenario else _intent("other", [], answerable=False, reason="no scenario"))

        if task == "generate_sql":
            question = self._find(r'Question: "(.*?)"\n', prompt)
            table = "user_data." + self._find(r"Table: user_data\.(ds_\w+)", prompt)
            scenario = self._scenario(question)
            if scenario is None or scenario[1] is None:
                return json.dumps({"sql": None, "cannot_answer_reason": "not possible"})
            sql = scenario[1].format(table=table, city=(re.findall(r"in (\w+)\?", question) or ["Bangalore"])[0])
            if "wrong column first" in question.lower() and "Your previous query failed" not in prompt:
                sql = f"SELECT city, COUNT(profit) AS n FROM {table} GROUP BY city"   # hallucinated column
            return json.dumps({"sql": sql, "chart_title": scenario[2]})

        if task == "explain":
            facts = prompt.split("Verified facts (calculated by code):")[1].split("Result table")[0].strip().splitlines()
            return "Based on the data: " + (facts[0].lstrip("- ") if facts else "done.")

        return "{}"
