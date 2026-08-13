"""LLM-backed SQL generation.

The model only ever sees the relevance-trimmed metadata text produced by
:mod:`app.metadata.retrieval` -- never the raw database -- and is
instructed to stay strictly within it. Its output is a plan
(:class:`~app.llm.schemas.SQLGenerationResult`), not something executed
directly: :mod:`app.sql.validator` and :mod:`app.sql.executor` are the
only things that ever touch the database.
"""

from __future__ import annotations

from app.llm.client import LLMClient
from app.llm.schemas import SQLGenerationResult

_SYSTEM_PROMPT = """You are the SQL generation engine of a read-only analytics platform.

Rules (all are hard requirements):
- Generate exactly one SQLite SELECT statement (CTEs / joins / aggregates allowed).
- Use ONLY the tables and columns given in the metadata context below. Never invent
  table or column names, and never reference any table not listed there.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH or any
  other statement that is not a SELECT.
- Prefer explicit column lists over SELECT * when the question asks for specific
  fields; use SELECT * only for open-ended "show me the data" style questions.
- Use the aggregation rules / default measures given in the metadata when the
  question is ambiguous about how to aggregate a measure.
- Treat declared types and observed data families as facts. SUM/AVG and arithmetic
  require numeric data; date functions require a temporal or date-like field. Never
  cast arbitrary labels merely to make a query run.
- Respect table grain and relationship cardinality. Never join two many-side/event
  tables row-for-row when that can multiply measures. Aggregate each source to the
  requested grain first, then combine compatible aggregates.
- COUNT(*) counts rows, COUNT(column) excludes NULL, and COUNT(DISTINCT key) counts
  entities. Choose deliberately from the wording and documented grain.
- Do not silently replace NULL with zero, discard NULL categories, or deduplicate
  rows unless the question requires that behavior.
- Add a LIMIT clause for global ranking/"top N" questions. For top N per group
  (for example, best product per month), calculate ROW_NUMBER, RANK, or DENSE_RANK
  in a CTE with PARTITION BY the group and metric ordering inside OVER(...), then
  filter that rank to = 1 or <= N in the outer query. Do not add a global LIMIT to
  a per-group result. For other questions the backend enforces a safety LIMIT.
- Use table aliases and qualify columns when joining more than one table.
- Join tables only through relationships present in the metadata. Relationships may
  be declared by SQLite or conservatively inferred from unique matching ID/key fields.
- A view can overlap a base table. Never combine a view with a table that may feed it
  unless the question explicitly requires both and the metadata proves the semantics.
- Preserve exact identifier spelling. Double-quote an identifier when it contains
  spaces, punctuation, or a SQLite keyword.
- For temporal grouping use SQLite date functions such as date(...) or strftime(...)
  against the exact temporal column supplied in metadata; never invent a calendar table.
- When a RESOLVED TIME CONTEXT is supplied, it is authoritative: copy its literal
  start and exclusive-end dates exactly, apply the same range to every relevant
  event/fact source, and never derive separate MAX(date) years per table.
- Use SQLite syntax only: never use TOP/TOP(...), :: casts, SQL Server brackets,
  or a trailing comma before FROM. Put LIMIT N at the very end of a ranking query.
- A business concept does not imply that a similarly named discriminator column
  exists. When comparable categories are stored in separate event tables, combine
  only schema-compatible measures at the same grain. Do not add a table twice or
  invent a discriminator column.
- For rankings, order by the requested metric in the correct direction and apply the
  exact requested N. For a singular GLOBAL row ranking use LIMIT 1; for a singular
  PER-GROUP ranking filter the partitioned window rank to 1. If the question asks
  only for the highest/lowest value (not its associated entity), use MAX/MIN instead
  of manufacturing a row ranking.
"""


def _user_prompt(
    question: str,
    metadata_text: str,
    prior_sql: str | None,
    prior_error: str | None,
    conversation_history_text: str | None,
) -> str:
    parts = [f"METADATA CONTEXT:\n{metadata_text}"]
    if conversation_history_text:
        parts.append(
            f"\n{conversation_history_text}\n"
            "(Only use this to resolve references like \"that\"/\"same but ...\" "
            "in the new question below -- do not answer the old question again.)"
        )
    parts.append(f"\nQUESTION:\n{question}")
    if prior_sql and prior_error:
        parts.append(
            "\nA previous attempt at this question failed validation/execution. "
            f"Previous SQL:\n{prior_sql}\n\nError:\n{prior_error}\n\n"
            "Rebuild the query from scratch as SQLite. Use only exact metadata names, "
            "LIMIT instead of TOP, no :: syntax, and no comma immediately before FROM. "
            "Address every reported semantic issue directly; do not evade type, join, "
            "time, grouping, or ranking checks with arbitrary casts or unused clauses."
        )
    return "\n".join(parts)


def generate_sql(
    llm_client: LLMClient,
    *,
    question: str,
    metadata_text: str,
    prior_sql: str | None = None,
    prior_error: str | None = None,
    conversation_history_text: str | None = None,
) -> SQLGenerationResult:
    return llm_client.complete_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_user_prompt(
            question, metadata_text, prior_sql, prior_error, conversation_history_text
        ),
        schema=SQLGenerationResult,
    )
