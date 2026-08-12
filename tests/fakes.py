"""A deterministic fake LLM client so the graph/backend can be tested without
network access or real Azure credentials.
"""

from __future__ import annotations

from app.llm.client import LLMClient, LLMError
from app.llm.schemas import IntentResult, InsightResult, SQLGenerationResult, VisualizationPlan


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        sql: str,
        relevant_tables: list[str] | None = None,
        fail_first_n_sql: int = 0,
        bad_sql: str = "SELECT * FROM NotATable",
        always_fail: bool = False,
    ):
        self.sql = sql
        self.relevant_tables = relevant_tables or []
        self.fail_first_n_sql = fail_first_n_sql
        self.bad_sql = bad_sql
        self.always_fail = always_fail
        self.sql_call_count = 0
        self.calls: list[str] = []

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return "ok"

    def complete_json(self, *, system_prompt, user_prompt, schema, max_repair_attempts=1):
        self.calls.append(schema.__name__)

        if schema is IntentResult:
            return IntentResult(
                intent_summary="Test intent",
                analysis_type="aggregation",
                relevant_tables=self.relevant_tables,
            )

        if schema is SQLGenerationResult:
            self.sql_call_count += 1
            if self.always_fail or self.sql_call_count <= self.fail_first_n_sql:
                return SQLGenerationResult(
                    sql=self.bad_sql, explanation="bad", tables_used=["NotATable"]
                )
            return SQLGenerationResult(sql=self.sql, explanation="Test query", tables_used=[])

        if schema is InsightResult:
            return InsightResult(summary="Test insight summary.", key_findings=["finding one"])

        if schema is VisualizationPlan:
            return VisualizationPlan(chart_type="bar", title="Test chart", x=None, y=None)

        raise LLMError(f"FakeLLMClient has no canned response for schema {schema}")
