"""Deterministic schema discovery.

Introspects the live SQLite database (tables, columns, types, keys,
foreign-key relationships, row counts, and small samples of low-cardinality
categorical columns) with no LLM involvement. This is the factual backbone
that :mod:`app.metadata.store` enriches with descriptions/business meaning
and persists to disk.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

CATEGORICAL_SAMPLE_LIMIT = 12
CATEGORICAL_MAX_DISTINCT = 30

# Hints are matched against whole tokens (see _tokenize), never raw
# substrings, so e.g. "SalesOrderLineNumber" doesn't false-match "sales"
# and "TaxonomyCode" doesn't false-match "tax".
_AVG_MEASURE_HINTS = frozenset({"price", "cost"})
_SUM_MEASURE_HINTS = frozenset(
    {"amount", "total", "tax", "freight", "discount", "quantity", "qty"}
)
_MEASURE_HINTS = _AVG_MEASURE_HINTS | _SUM_MEASURE_HINTS
_TEMPORAL_HINTS = frozenset({"date", "time"})
_FLAG_HINTS = frozenset({"flag", "status", "active"})


def _tokenize(identifier: str) -> list[str]:
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", identifier)
    s = s.replace("_", " ")
    return [w.lower() for w in s.split() if w]


def humanize(identifier: str) -> str:
    """Turn ``CamelCase`` / ``snake_case`` identifiers into readable text."""
    s = " ".join(_tokenize(identifier))
    return s[:1].upper() + s[1:] if s else s


def infer_semantic_role(column_name: str, sql_type: str, is_pk: bool, is_fk: bool) -> str:
    words = set(_tokenize(column_name))
    name_l = column_name.lower()
    type_u = (sql_type or "").upper()

    if is_pk or is_fk or name_l.endswith("key") or name_l.endswith("_id") or name_l == "id":
        return "key"
    if words & _TEMPORAL_HINTS:
        return "temporal"
    if words & _FLAG_HINTS:
        return "flag"
    if type_u in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "INTEGER") and (
        words & _MEASURE_HINTS
    ):
        return "measure"
    if type_u in ("TEXT", "VARCHAR", "CHAR"):
        return "categorical_attribute"
    if type_u in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "INTEGER"):
        return "numeric_attribute"
    return "attribute"


def infer_default_aggregation(semantic_role: str, column_name: str = "") -> str | None:
    if semantic_role != "measure":
        return None
    words = set(_tokenize(column_name))
    if words & _AVG_MEASURE_HINTS:
        return "avg"
    return "sum"


@dataclass
class ColumnInfo:
    name: str
    sql_type: str
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    references: dict[str, str] | None
    semantic_role: str
    default_aggregation: str | None
    sample_values: list[Any] = field(default_factory=list)
    distinct_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sql_type": self.sql_type,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references": self.references,
            "semantic_role": self.semantic_role,
            "default_aggregation": self.default_aggregation,
            "sample_values": self.sample_values,
            "distinct_count": self.distinct_count,
        }


@dataclass
class TableInfo:
    name: str
    kind: str  # "dimension" | "fact" | "unknown"
    row_count: int
    columns: list[ColumnInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
        }


def _list_user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name;"
    ).fetchall()
    return [r["name"] for r in rows]


def _foreign_keys(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, str]]:
    rows = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
    return {
        row["from"]: {"table": row["table"], "column": row["to"]} for row in rows
    }


def _classify_table_kind(table: str, has_measures: bool, has_incoming_fk: bool) -> str:
    name_l = table.lower()
    # Naming convention is the strongest signal when present.
    if name_l.startswith("dim"):
        return "dimension"
    if name_l.startswith("fact"):
        return "fact"
    # Fallback for arbitrarily-named tables (extensibility): a table other
    # tables point a foreign key at is a dimension; one with monetary/qty
    # measures and no inbound references is a fact table.
    if has_incoming_fk:
        return "dimension"
    if has_measures:
        return "fact"
    return "unknown"


def _sample_categorical(conn: sqlite3.Connection, table: str, column: str) -> tuple[list[Any], int | None]:
    try:
        distinct_count = conn.execute(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
        ).fetchone()[0]
    except sqlite3.Error:
        return [], None

    if distinct_count is None or distinct_count > CATEGORICAL_MAX_DISTINCT:
        return [], distinct_count

    rows = conn.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL LIMIT {CATEGORICAL_SAMPLE_LIMIT}'
    ).fetchall()
    return [r[0] for r in rows], distinct_count


def discover_table(conn: sqlite3.Connection, table: str) -> TableInfo:
    col_rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    fk_map = _foreign_keys(conn, table)
    row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    columns: list[ColumnInfo] = []
    has_measures = False
    for row in col_rows:
        name = row["name"]
        sql_type = row["type"] or ""
        is_pk = bool(row["pk"])
        is_fk = name in fk_map
        role = infer_semantic_role(name, sql_type, is_pk, is_fk)
        if role == "measure":
            has_measures = True

        sample_values: list[Any] = []
        distinct_count: int | None = None
        if role == "categorical_attribute":
            sample_values, distinct_count = _sample_categorical(conn, table, name)

        columns.append(
            ColumnInfo(
                name=name,
                sql_type=sql_type,
                nullable=not bool(row["notnull"]),
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                references=fk_map.get(name),
                semantic_role=role,
                default_aggregation=infer_default_aggregation(role, name),
                sample_values=sample_values,
                distinct_count=distinct_count,
            )
        )

    kind = _classify_table_kind(table, has_measures, has_incoming_fk=False)
    return TableInfo(name=table, kind=kind, row_count=row_count, columns=columns)


def discover_schema(conn: sqlite3.Connection) -> dict[str, TableInfo]:
    """Introspect every user table in the database."""
    tables = {t: discover_table(conn, t) for t in _list_user_tables(conn)}

    # A table referenced by another table's FK is a dimension, even if the
    # naming convention doesn't say so (extensibility: works for any schema).
    referenced: set[str] = set()
    for info in tables.values():
        for col in info.columns:
            if col.references:
                referenced.add(col.references["table"])
    for name in referenced:
        if name in tables and tables[name].kind == "unknown":
            tables[name].kind = "dimension"

    return tables


def schema_signature(tables: dict[str, TableInfo]) -> str:
    """Stable hash of structural schema (names/types/keys) used to detect drift.

    Deliberately excludes row counts and sampled values so routine data
    refreshes don't trigger a metadata rebuild -- only actual DDL changes do.
    """
    structural = {
        name: {
            "kind": info.kind,
            "columns": [
                {
                    "name": c.name,
                    "sql_type": c.sql_type,
                    "is_primary_key": c.is_primary_key,
                    "references": c.references,
                }
                for c in info.columns
            ],
        }
        for name, info in sorted(tables.items())
    }
    payload = json.dumps(structural, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
