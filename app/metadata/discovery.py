"""Deterministic schema discovery.

Introspects the live SQLite database (tables, views, columns, declared and
observed types, keys, relationships, bounded row counts, and small samples of low-cardinality
categorical columns) with no LLM involvement. This is the factual backbone
that :mod:`app.metadata.store` enriches with descriptions/business meaning
and persists to disk.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

CATEGORICAL_SAMPLE_LIMIT = 12
CATEGORICAL_MAX_DISTINCT = 30
CATEGORICAL_SCAN_LIMIT = 2000
ROW_COUNT_SCAN_LIMIT = 100_000
PROFILE_TEXT_VALUE_LIMIT = 256

# Hints are matched against whole tokens (see _tokenize), never raw
# substrings, so e.g. "SalesOrderLineNumber" doesn't false-match "sales"
# and "TaxonomyCode" doesn't false-match "tax".
_AVG_MEASURE_HINTS = frozenset(
    {"price", "cost", "rate", "ratio", "percentage", "percent", "pct"}
)
_SUM_MEASURE_HINTS = frozenset(
    {
        "amount", "total", "tax", "freight", "discount", "quantity", "qty",
        "revenue", "sales", "profit", "income", "expense", "balance", "value",
        "units", "hours", "duration",
    }
)
_MEASURE_HINTS = _AVG_MEASURE_HINTS | _SUM_MEASURE_HINTS
_TEMPORAL_HINTS = frozenset(
    {"date", "time", "timestamp", "ts", "dt", "year", "month", "day", "created", "updated"}
)
_FLAG_HINTS = frozenset({"flag", "status", "active", "enabled", "deleted", "valid"})
_SENSITIVE_HINTS = frozenset(
    {
        "password", "passwd", "secret", "token", "credential", "api",
        "email", "phone", "mobile", "address", "ssn", "sin", "passport",
        "license", "dob", "birth", "firstname", "lastname", "fullname",
        "first", "last", "name",
    }
)


def _tokenize(identifier: str) -> list[str]:
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", identifier)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return [w.lower() for w in s.split() if w]


def infer_data_classification(column_name: str) -> str:
    """Classify whether representative values may be persisted in prompts/docs."""
    tokens = _tokenize(column_name)
    words = set(tokens)
    compact = "".join(tokens)
    if words & _SENSITIVE_HINTS or any(
        marker in compact
        for marker in ("apikey", "accesstoken", "refreshtoken", "creditcard")
    ):
        return "sensitive"
    return "ordinary"


def humanize(identifier: str) -> str:
    """Turn ``CamelCase`` / ``snake_case`` identifiers into readable text."""
    s = " ".join(_tokenize(identifier))
    return s[:1].upper() + s[1:] if s else s


def _type_family(sql_type: str) -> str:
    """Map arbitrary SQLite declared types to their storage/semantic family."""
    declared = (sql_type or "").strip().upper()
    if any(token in declared for token in ("DATE", "TIME")):
        return "temporal"
    if "BOOL" in declared:
        return "boolean"
    if any(token in declared for token in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC", "MONEY")):
        return "numeric"
    if any(token in declared for token in ("CHAR", "CLOB", "TEXT", "JSON", "UUID")):
        return "text"
    if "BLOB" in declared:
        return "blob"
    return "unknown"


def infer_semantic_role(
    column_name: str,
    sql_type: str,
    is_pk: bool,
    is_fk: bool,
    observed_value_family: str | None = None,
) -> str:
    words = set(_tokenize(column_name))
    name_l = column_name.lower()
    family = _type_family(sql_type)
    # SQLite commonly declares dates and imported numeric fields as TEXT.
    # A uniformly observed bounded sample is stronger than TEXT affinity, but
    # never overrides an explicitly incompatible numeric/blob declaration.
    if observed_value_family == "temporal_text" and family in {"unknown", "text"}:
        family = "temporal"
    elif observed_value_family in {"numeric", "numeric_text"} and family in {
        "unknown", "text"
    }:
        family = "numeric"
    elif observed_value_family == "text" and family == "unknown":
        family = "text"

    if is_pk or is_fk or name_l.endswith("key") or name_l.endswith("_id") or name_l == "id":
        return "key"
    if family == "temporal" or words & _TEMPORAL_HINTS:
        return "temporal"
    if family == "boolean" or words & _FLAG_HINTS:
        return "flag"
    if family == "numeric" and words & {"number", "code", "sequence", "index", "rank"}:
        return "numeric_attribute"
    if family == "numeric" and words & _MEASURE_HINTS:
        return "measure"
    if family == "text":
        return "categorical_attribute"
    if family == "numeric":
        return "numeric_attribute"
    return "attribute"


def infer_default_aggregation(semantic_role: str, column_name: str = "") -> str | None:
    if semantic_role != "measure":
        return None
    words = set(_tokenize(column_name))
    # Explicit additive wording wins over per-unit/ratio hints (e.g.
    # TotalProductCost is a total even though it also contains "cost").
    if words & {"amount", "total", "quantity", "qty", "revenue", "sales", "profit", "income", "expense"}:
        return "sum"
    if words & _AVG_MEASURE_HINTS or ({"unit", "cost"} <= words):
        return "avg"
    return "sum"


@dataclass
class ColumnInfo:
    name: str
    sql_type: str
    nullable: bool
    is_primary_key: bool
    is_unique: bool
    is_foreign_key: bool
    references: dict[str, Any] | None
    relationship_source: str | None
    semantic_role: str
    default_aggregation: str | None
    sample_values: list[Any] = field(default_factory=list)
    distinct_count: int | None = None
    observed_storage_types: list[str] = field(default_factory=list)
    sampled_non_null_count: int = 0
    sampled_null_fraction: float | None = None
    observed_value_family: str = "empty"
    data_classification: str = "ordinary"
    relationship_constraint_id: int | None = None
    relationship_constraint_sequence: int = 0
    relationship_constraint_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sql_type": self.sql_type,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_unique": self.is_unique,
            "is_foreign_key": self.is_foreign_key,
            "references": self.references,
            "relationship_source": self.relationship_source,
            "semantic_role": self.semantic_role,
            "default_aggregation": self.default_aggregation,
            "sample_values": self.sample_values,
            "distinct_count": self.distinct_count,
            "declared_type_family": _type_family(self.sql_type),
            "observed_storage_types": self.observed_storage_types,
            "sampled_non_null_count": self.sampled_non_null_count,
            "sampled_null_fraction": self.sampled_null_fraction,
            "observed_value_family": self.observed_value_family,
            "data_classification": self.data_classification,
            "relationship_constraint_id": self.relationship_constraint_id,
            "relationship_constraint_sequence": self.relationship_constraint_sequence,
            "relationship_constraint_size": self.relationship_constraint_size,
        }


@dataclass
class TableInfo:
    name: str
    kind: str  # structural hint: dimension/fact/entity/bridge/view/unknown
    row_count: int
    columns: list[ColumnInfo]
    object_type: str = "table"
    depends_on: list[str] = field(default_factory=list)
    row_count_is_lower_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "row_count": self.row_count,
            "object_type": self.object_type,
            "depends_on": self.depends_on,
            "row_count_is_lower_bound": self.row_count_is_lower_bound,
            "columns": [c.to_dict() for c in self.columns],
        }


def _list_user_objects(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name;"
    ).fetchall()
    return rows


def _list_user_tables(conn: sqlite3.Connection) -> list[str]:
    return [str(row["name"]) for row in _list_user_objects(conn)]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _foreign_keys(conn: sqlite3.Connection, table: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({_quote_literal(table)})").fetchall()
    group_sizes: dict[int, int] = {}
    for row in rows:
        group_sizes[int(row["id"])] = group_sizes.get(int(row["id"]), 0) + 1
    return {
        row["from"]: {
            "table": row["table"],
            "column": row["to"],
            "constraint_id": int(row["id"]),
            "constraint_sequence": int(row["seq"]),
            "constraint_size": group_sizes[int(row["id"])],
        }
        for row in rows
    }


def _single_column_unique_fields(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return declared single-column UNIQUE fields, excluding partial indexes."""
    unique: set[str] = set()
    try:
        rows = conn.execute(f"PRAGMA index_list({_quote_literal(table)})").fetchall()
        for row in rows:
            keys = set(row.keys())
            is_unique = bool(row["unique"] if "unique" in keys else row[2])
            is_partial = bool(row["partial"] if "partial" in keys else 0)
            if not is_unique or is_partial:
                continue
            index_name = str(row["name"] if "name" in keys else row[1])
            columns = conn.execute(
                f"PRAGMA index_info({_quote_literal(index_name)})"
            ).fetchall()
            names = [str(item["name"]) for item in columns if item["name"] is not None]
            if len(names) == 1:
                unique.add(names[0])
    except sqlite3.Error:
        return set()
    return unique


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
        rows = conn.execute(
            f"SELECT {_quote_identifier(column)} FROM {_quote_identifier(table)} "
            f"WHERE {_quote_identifier(column)} IS NOT NULL "
            f"LIMIT {CATEGORICAL_SCAN_LIMIT}"
        ).fetchall()
    except sqlite3.Error:
        return [], None
    unique = list(dict.fromkeys(row[0] for row in rows))
    if len(unique) > CATEGORICAL_MAX_DISTINCT:
        return [], None
    # Exact only when the bounded scan exhausted the table; otherwise leave the
    # count unknown rather than claim a sample-derived cardinality as fact.
    distinct_count = len(unique) if len(rows) < CATEGORICAL_SCAN_LIMIT else None
    return unique[:CATEGORICAL_SAMPLE_LIMIT], distinct_count


def _storage_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return "blob"
    if isinstance(value, bool):
        return "integer"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    return "text"


def _looks_like_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_iso_time(value: str) -> bool:
    candidate = value.strip().replace("Z", "+00:00")
    if not re.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:[T\s].*)?$", candidate):
        return False
    try:
        datetime.fromisoformat(candidate)
        return True
    except ValueError:
        return False


def _observed_value_family(values: list[Any]) -> str:
    if not values:
        return "empty"
    storage = {_storage_type(value) for value in values}
    if storage <= {"integer", "real"}:
        return "numeric"
    if storage == {"text"}:
        text_values = [str(value).strip() for value in values]
        if text_values and all(map(_looks_like_number, text_values)):
            return "numeric_text"
        if text_values and all(map(_looks_like_iso_time, text_values)):
            return "temporal_text"
        return "text"
    if storage == {"blob"}:
        return "blob"
    return "mixed"


def _sample_column_profiles(
    conn: sqlite3.Connection, table: str, column_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Profile all columns in one bounded read, independent of table size."""
    if not column_names:
        return {}
    selected_parts: list[str] = []
    for name in column_names:
        quoted = _quote_identifier(name)
        selected_parts.extend(
            [
                f"typeof({quoted})",
                f"CASE WHEN typeof({quoted}) = 'blob' THEN NULL "
                f"ELSE substr(CAST({quoted} AS TEXT), 1, {PROFILE_TEXT_VALUE_LIMIT}) END",
            ]
        )
    selected = ", ".join(selected_parts)
    try:
        rows = conn.execute(
            f"SELECT {selected} FROM {_quote_identifier(table)} "
            f"LIMIT {CATEGORICAL_SCAN_LIMIT}"
        ).fetchall()
    except sqlite3.Error:
        return {}
    sampled = len(rows)
    profiles: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(column_names):
        storage_types = [str(row[index * 2]) for row in rows]
        values = [row[index * 2 + 1] for row in rows]
        non_null = [value for value in values if value is not None]
        unique = list(dict.fromkeys(non_null))
        observed_types = sorted({value for value in storage_types if value != "null"})
        if observed_types and set(observed_types) <= {"integer", "real"}:
            observed_family = "numeric"
        elif observed_types == ["blob"]:
            observed_family = "blob"
        else:
            observed_family = _observed_value_family(non_null)
        profiles[name] = {
            "observed_storage_types": observed_types,
            "sampled_non_null_count": len(non_null),
            "sampled_null_fraction": (
                round((sampled - len(non_null)) / sampled, 6) if sampled else None
            ),
            "observed_value_family": observed_family,
            "sample_values": (
                unique[:CATEGORICAL_SAMPLE_LIMIT]
                if len(unique) <= CATEGORICAL_MAX_DISTINCT
                else []
            ),
            "distinct_count": len(unique) if sampled < CATEGORICAL_SCAN_LIMIT else None,
        }
    return profiles


def _view_dependencies(definition: str | None, known_objects: set[str]) -> list[str]:
    if not definition:
        return []
    try:
        parsed = sqlglot.parse_one(definition, dialect="sqlite")
    except Exception:  # noqa: BLE001 - malformed vendor SQL should not break discovery
        return []
    names_by_lower = {name.casefold(): name for name in known_objects}
    dependencies = {
        names_by_lower[table.name.casefold()]
        for table in parsed.find_all(exp.Table)
        if table.name.casefold() in names_by_lower
    }
    return sorted(dependencies)


def discover_table(conn: sqlite3.Connection, table: str) -> TableInfo:
    col_rows = conn.execute(f"PRAGMA table_info({_quote_literal(table)})").fetchall()
    fk_map = _foreign_keys(conn, table)
    unique_fields = _single_column_unique_fields(conn, table)
    row_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM {_quote_identifier(table)} "
            f"LIMIT {ROW_COUNT_SCAN_LIMIT + 1})"
        ).fetchone()[0]
    )
    row_count_is_lower_bound = row_count > ROW_COUNT_SCAN_LIMIT
    profiles = _sample_column_profiles(
        conn, table, [str(row["name"]) for row in col_rows]
    )

    columns: list[ColumnInfo] = []
    has_measures = False
    for row in col_rows:
        name = row["name"]
        sql_type = row["type"] or ""
        is_pk = bool(row["pk"])
        is_unique = is_pk or name in unique_fields
        is_fk = name in fk_map
        profile = profiles.get(name, {})
        role = infer_semantic_role(
            name,
            sql_type,
            is_pk,
            is_fk,
            profile.get("observed_value_family"),
        )
        if role == "measure":
            has_measures = True

        sample_values: list[Any] = []
        distinct_count: int | None = None
        data_classification = infer_data_classification(name)
        if (
            role in {"categorical_attribute", "flag"}
            and data_classification == "ordinary"
        ):
            sample_values = list(profile.get("sample_values") or [])
            distinct_count = profile.get("distinct_count")

        columns.append(
            ColumnInfo(
                name=name,
                sql_type=sql_type,
                nullable=not bool(row["notnull"]),
                is_primary_key=is_pk,
                is_unique=is_unique,
                is_foreign_key=is_fk,
                references=(
                    {
                        "table": fk_map[name]["table"],
                        "column": fk_map[name]["column"],
                    }
                    if is_fk
                    else None
                ),
                relationship_source="declared" if is_fk else None,
                semantic_role=role,
                default_aggregation=infer_default_aggregation(role, name),
                sample_values=sample_values,
                distinct_count=distinct_count,
                observed_storage_types=list(profile.get("observed_storage_types") or []),
                sampled_non_null_count=int(profile.get("sampled_non_null_count") or 0),
                sampled_null_fraction=profile.get("sampled_null_fraction"),
                observed_value_family=str(profile.get("observed_value_family") or "empty"),
                data_classification=data_classification,
                relationship_constraint_id=(
                    fk_map[name]["constraint_id"] if is_fk else None
                ),
                relationship_constraint_sequence=(
                    fk_map[name]["constraint_sequence"] if is_fk else 0
                ),
                relationship_constraint_size=(
                    fk_map[name]["constraint_size"] if is_fk else 1
                ),
            )
        )

    kind = _classify_table_kind(table, has_measures, has_incoming_fk=False)
    return TableInfo(
        name=table,
        kind=kind,
        row_count=row_count,
        columns=columns,
        row_count_is_lower_bound=row_count_is_lower_bound,
    )


def discover_schema(conn: sqlite3.Connection) -> dict[str, TableInfo]:
    """Introspect every user table in the database."""
    objects = _list_user_objects(conn)
    known_objects = {str(row["name"]) for row in objects}
    tables: dict[str, TableInfo] = {}
    for row in objects:
        name = str(row["name"])
        info = discover_table(conn, name)
        info.object_type = str(row["type"])
        if info.object_type == "view":
            info.depends_on = [
                dependency
                for dependency in _view_dependencies(row["sql"], known_objects)
                if dependency.casefold() != name.casefold()
            ]
        tables[name] = info

    _infer_undeclared_foreign_keys(conn, tables)

    # A table referenced by another table's FK is a dimension, even if the
    # naming convention doesn't say so (extensibility: works for any schema).
    referenced: set[str] = set()
    for info in tables.values():
        for col in info.columns:
            if col.references:
                referenced.add(col.references["table"])
    for name, table in tables.items():
        lowered = name.casefold()
        if table.object_type == "view":
            table.kind = "view"
            continue
        if lowered.startswith("dim"):
            table.kind = "dimension"
            continue
        if lowered.startswith("fact"):
            table.kind = "fact"
            continue
        outgoing_count = sum(bool(column.references) for column in table.columns)
        has_measures = any(column.semantic_role == "measure" for column in table.columns)
        has_temporal = any(column.semantic_role == "temporal" for column in table.columns)
        primary_key = {column.name for column in table.columns if column.is_primary_key}
        foreign_keys = {column.name for column in table.columns if column.references}
        if (
            outgoing_count >= 2
            and primary_key
            and primary_key <= foreign_keys
            and not has_measures
        ):
            table.kind = "bridge"
        elif name in referenced and not outgoing_count:
            table.kind = "dimension"
        elif has_measures or (outgoing_count and has_temporal):
            table.kind = "fact"
        elif outgoing_count:
            table.kind = "entity"
        else:
            table.kind = "unknown"

    return tables


def _relation_stem(identifier: str) -> str:
    tokens = _tokenize(identifier)
    while tokens and tokens[0] in {"dim", "fact", "tbl"}:
        tokens.pop(0)
    value = "".join(tokens)
    if value.endswith("ies"):
        value = value[:-3] + "y"
    elif value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _relationship_values_overlap(
    conn: sqlite3.Connection,
    source_table: str,
    source_column: ColumnInfo,
    target_table: str,
    target_column: ColumnInfo,
) -> bool:
    """Require compatible types and strong observed overlap before inferring a join."""
    source_family = _type_family(source_column.sql_type)
    target_family = _type_family(target_column.sql_type)
    if (
        source_family != "unknown"
        and target_family != "unknown"
        and source_family != target_family
    ):
        return False
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_quote_identifier(source_column.name)} "
            f"FROM {_quote_identifier(source_table)} "
            f"WHERE {_quote_identifier(source_column.name)} IS NOT NULL LIMIT 100"
        ).fetchall()
        values = [row[0] for row in rows]
        if not values:
            return False
        placeholders = ",".join("?" for _ in values)
        matched = conn.execute(
            f"SELECT COUNT(DISTINCT {_quote_identifier(target_column.name)}) "
            f"FROM {_quote_identifier(target_table)} "
            f"WHERE {_quote_identifier(target_column.name)} IN ({placeholders})",
            values,
        ).fetchone()[0]
    except sqlite3.Error:
        return False
    return int(matched or 0) / len(values) >= 0.8


def _infer_undeclared_foreign_keys(
    conn: sqlite3.Connection, tables: dict[str, TableInfo]
) -> None:
    """Conservatively infer common ID/key relationships absent from SQLite DDL.

    Many imported SQLite files omit FOREIGN KEY clauses. We infer only when a
    candidate target is unique: an exact non-generic primary/UNIQUE-key name
    match, or ``customer_id``/``CustomerKey`` whose prefix matches a table with
    a compatible unique key. Ambiguous candidates remain ordinary keys.
    """
    unique_keys: list[tuple[str, ColumnInfo]] = []
    for table_name, table in tables.items():
        unique_keys.extend(
            (table_name, column)
            for column in table.columns
            if column.is_unique
        )
    for source_name, source in tables.items():
        # A view may project keys from its base table, but that overlap is
        # lineage—not evidence that joining the view back to the base is safe.
        if source.object_type == "view":
            continue
        for column in source.columns:
            if column.is_primary_key or column.references:
                continue
            words = _tokenize(column.name)
            exact = [
                (table_name, target)
                for table_name, target in unique_keys
                if table_name != source_name
                and target.name.casefold() == column.name.casefold()
                and target.name.casefold() not in {"id", "key"}
            ]
            candidates = exact
            if not candidates:
                if not words or words[-1] not in {"id", "key"}:
                    continue
                column_stem = _relation_stem("".join(words[:-1]))
                candidates = [
                    (table_name, target)
                    for table_name, target in unique_keys
                    if table_name != source_name
                    and column_stem
                    and _relation_stem(table_name) == column_stem
                    and (
                        target.name.casefold() in {"id", "key", column.name.casefold()}
                        or _relation_stem(target.name) == column_stem
                    )
                ]
            if len(candidates) == 1:
                target_table, target_column = candidates[0]
                if not _relationship_values_overlap(
                    conn,
                    source_name,
                    column,
                    target_table,
                    target_column,
                ):
                    continue
                column.is_foreign_key = True
                column.references = {
                    "table": target_table,
                    "column": target_column.name,
                }
                column.relationship_source = "inferred"
                column.semantic_role = "key"


def schema_signature(tables: dict[str, TableInfo]) -> str:
    """Stable hash of structural schema (names/types/keys) used to detect drift.

    Deliberately excludes row counts and sampled values so routine data
    refreshes don't trigger a metadata rebuild -- only actual DDL changes do.
    """
    structural = {
        name: {
            "kind": info.kind,
            "object_type": info.object_type,
            "depends_on": info.depends_on,
            "columns": [
                {
                    "name": c.name,
                    "sql_type": c.sql_type,
                    "is_primary_key": c.is_primary_key,
                    "is_unique": c.is_unique,
                    "references": c.references,
                }
                for c in info.columns
            ],
        }
        for name, info in sorted(tables.items())
    }
    payload = json.dumps(structural, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
