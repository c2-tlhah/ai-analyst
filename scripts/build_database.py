#!/usr/bin/env python3
"""Import arbitrary tabular files into a SQLite database.

This utility is intentionally domain-neutral. It discovers CSV, TSV, JSON,
JSONL, and Parquet files, preserves their identifiers, infers conservative
SQLite types/keys, and optionally accepts a small manifest for exact schema
control. The analyst itself does not depend on this utility; it can connect to
any existing SQLite database directly.

Examples:
    python scripts/build_database.py --input data/raw --output data/local.db
    python scripts/build_database.py --input exports --manifest schema.json --force

Manifest shape (every field is optional except a referenced source):
    {
      "tables": {
        "orders": {
          "source": "orders.csv",
          "rename": {"Order ID": "order_id"},
          "types": {"order_id": "INTEGER", "ordered_at": "TEXT"},
          "primary_key": ["order_id"],
          "required": ["order_id"],
          "foreign_keys": {"customer_id": "customers.customer_id"}
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ai_analyst.db"
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
SQLITE_TYPES = {"INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC"}


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _tokens(identifier: str) -> list[str]:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", identifier)
    return [part.casefold() for part in re.split(r"[^A-Za-z0-9]+", spaced) if part]


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported source format: {path.suffix}")


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"tables": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tables", {}), dict):
        raise ValueError("Manifest must be a JSON object containing a 'tables' map.")
    return payload


def _discover_sources(input_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configured = manifest.get("tables") or {}
    if configured:
        tables: dict[str, dict[str, Any]] = {}
        for table_name, options in configured.items():
            if not isinstance(options, dict) or not options.get("source"):
                raise ValueError(f"Manifest table {table_name!r} requires a source file.")
            source = (input_dir / str(options["source"])).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Source for table {table_name!r} not found: {source}")
            tables[str(table_name)] = {**options, "path": source}
        return tables

    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No supported tabular files found in {input_dir} "
            f"({', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )
    tables: dict[str, dict[str, Any]] = {}
    for path in files:
        table_name = path.stem
        if table_name.casefold() in {name.casefold() for name in tables}:
            raise ValueError(f"Multiple source files resolve to table {table_name!r}.")
        tables[table_name] = {"path": path}
    return tables


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "TEXT"
    return "TEXT"


def _infer_primary_key(table_name: str, frame: pd.DataFrame) -> list[str]:
    """Infer only a single, non-null, unique ID/key field with strong naming evidence."""
    table_words = [word for word in _tokens(table_name) if word not in {"dim", "fact", "tbl"}]
    table_stem = "".join(table_words)
    candidates: list[tuple[int, str]] = []
    for column in frame.columns:
        series = frame[column]
        if series.empty or series.isna().any() or not series.is_unique:
            continue
        words = _tokens(str(column))
        compact = "".join(words)
        if not words or words[-1] not in {"id", "key"}:
            continue
        prefix = "".join(words[:-1])
        score = 3 if prefix and prefix == table_stem else 1
        if compact in {"id", "key"}:
            score = 2
        candidates.append((score, str(column)))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return []
    return [candidates[0][1]]


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _prepare_tables(input_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for table_name, options in _discover_sources(input_dir, manifest).items():
        frame = _read_source(options["path"])
        frame = frame.rename(columns=dict(options.get("rename") or {}))
        frame.columns = [str(column) for column in frame.columns]
        if not frame.columns.is_unique:
            raise ValueError(f"Table {table_name!r} has duplicate column names.")
        if frame.columns.empty:
            raise ValueError(f"Table {table_name!r} has no columns.")
        required = [str(value) for value in options.get("required") or []]
        unknown_required = sorted(set(required) - set(frame.columns))
        if unknown_required:
            raise ValueError(f"Table {table_name!r} requires missing columns: {unknown_required}.")
        if required:
            frame = frame.dropna(subset=required).copy()
        primary_key = [str(value) for value in options.get("primary_key") or []]
        if not primary_key:
            primary_key = _infer_primary_key(table_name, frame)
        unknown_pk = sorted(set(primary_key) - set(frame.columns))
        if unknown_pk:
            raise ValueError(f"Table {table_name!r} has unknown primary-key columns: {unknown_pk}.")
        if primary_key and (
            frame[primary_key].isna().any().any()
            or frame.duplicated(subset=primary_key).any()
        ):
            raise ValueError(f"Primary key for table {table_name!r} is null or duplicated.")
        type_overrides = {
            str(column): str(value).strip().upper()
            for column, value in (options.get("types") or {}).items()
        }
        unknown_type_columns = sorted(set(type_overrides) - set(frame.columns))
        if unknown_type_columns:
            raise ValueError(
                f"Table {table_name!r} has type overrides for unknown columns: "
                f"{unknown_type_columns}."
            )
        invalid_types = sorted(set(type_overrides.values()) - SQLITE_TYPES)
        if invalid_types:
            raise ValueError(f"Unsupported SQLite type(s) for {table_name!r}: {invalid_types}.")
        prepared[table_name] = {
            **options,
            "frame": frame,
            "primary_key": primary_key,
            "types": {
                column: type_overrides.get(column, _sqlite_type(frame[column]))
                for column in frame.columns
            },
        }
    return prepared


def _foreign_keys(table_name: str, tables: dict[str, dict[str, Any]]) -> dict[str, str]:
    options = tables[table_name]
    configured = {
        str(column): str(target)
        for column, target in (options.get("foreign_keys") or {}).items()
    }
    if configured:
        return configured
    frame = options["frame"]
    inferred: dict[str, str] = {}
    for column in frame.columns:
        matches: list[tuple[str, str]] = []
        for target_name, target in tables.items():
            if target_name == table_name or len(target["primary_key"]) != 1:
                continue
            target_column = target["primary_key"][0]
            if str(column).casefold() != target_column.casefold():
                continue
            source_values = set(frame[column].dropna().head(500).tolist())
            target_values = set(target["frame"][target_column].dropna().tolist())
            if source_values and len(source_values & target_values) / len(source_values) >= 0.8:
                matches.append((target_name, target_column))
        if len(matches) == 1:
            inferred[str(column)] = f"{matches[0][0]}.{matches[0][1]}"
    return inferred


def _validate_foreign_keys(
    table_name: str, foreign_keys: dict[str, str], tables: dict[str, dict[str, Any]]
) -> list[tuple[str, str, str]]:
    validated: list[tuple[str, str, str]] = []
    columns = set(tables[table_name]["frame"].columns)
    for source_column, target in foreign_keys.items():
        if source_column not in columns or "." not in target:
            raise ValueError(f"Invalid foreign key {table_name}.{source_column} -> {target!r}.")
        target_table, target_column = target.rsplit(".", 1)
        if target_table not in tables or target_column not in tables[target_table]["frame"].columns:
            raise ValueError(f"Unknown foreign-key target: {target!r}.")
        if target_column not in tables[target_table]["primary_key"]:
            raise ValueError(
                f"Foreign-key target {target!r} is not part of that table's primary key."
            )
        validated.append((source_column, target_table, target_column))
    return validated


def build(
    force: bool = False,
    *,
    input_dir: Path = DEFAULT_INPUT,
    output: Path = DEFAULT_OUTPUT,
    manifest_path: Path | None = None,
) -> Path:
    """Build one SQLite file atomically from arbitrary tabular sources."""
    input_dir = input_dir.resolve()
    output = output.resolve()
    if output.exists() and not force:
        print(f"[build_database] {output} already exists; use --force to rebuild.")
        return output
    manifest = _load_manifest(manifest_path.resolve() if manifest_path else None)
    tables = _prepare_tables(input_dir, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    conn = sqlite3.connect(temporary)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        relationships: dict[str, list[tuple[str, str, str]]] = {}
        for table_name, table in tables.items():
            frame: pd.DataFrame = table["frame"]
            primary_key: list[str] = table["primary_key"]
            definitions = [
                f"{_quote(column)} {table['types'][column]}"
                + (" NOT NULL" if column in primary_key else "")
                for column in frame.columns
            ]
            if primary_key:
                definitions.append(
                    "PRIMARY KEY (" + ", ".join(_quote(value) for value in primary_key) + ")"
                )
            relationships[table_name] = _validate_foreign_keys(
                table_name, _foreign_keys(table_name, tables), tables
            )
            definitions.extend(
                f"FOREIGN KEY ({_quote(source)}) REFERENCES "
                f"{_quote(target_table)} ({_quote(target_column)})"
                for source, target_table, target_column in relationships[table_name]
            )
            conn.execute(f"CREATE TABLE {_quote(table_name)} ({', '.join(definitions)})")
        # Source filenames have no guaranteed dependency order, and valid
        # relational datasets may even contain cycles. Defer all FK checks
        # until every table has been loaded, then verify before committing.
        conn.execute("PRAGMA defer_foreign_keys = ON")
        for table_name, table in tables.items():
            frame = table["frame"]
            columns = list(frame.columns)
            placeholders = ", ".join("?" for _ in columns)
            insert = (
                f"INSERT INTO {_quote(table_name)} "
                f"({', '.join(_quote(column) for column in columns)}) VALUES ({placeholders})"
            )
            rows = [
                tuple(_coerce_value(value) for value in row)
                for row in frame.itertuples(index=False, name=None)
            ]
            if rows:
                conn.executemany(insert, rows)
            for source, _target_table, _target_column in relationships[table_name]:
                index_name = f"ix_{table_name}_{source}"[:60]
                conn.execute(
                    f"CREATE INDEX {_quote(index_name)} ON "
                    f"{_quote(table_name)} ({_quote(source)})"
                )
            print(f"[build_database] loaded {table_name}: {len(frame)} row(s)")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(
                f"Imported data violates {len(violations)} foreign-key constraint(s)."
            )
        conn.execute("ANALYZE")
        conn.commit()
    except Exception:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    temporary.replace(output)
    print(f"[build_database] done -> {output} ({len(tables)} table(s))")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory containing tabular files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="SQLite file to create.")
    parser.add_argument("--manifest", type=Path, help="Optional JSON schema manifest.")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing output file.")
    args = parser.parse_args()
    try:
        build(
            force=args.force,
            input_dir=args.input,
            output=args.output,
            manifest_path=args.manifest,
        )
    except (FileNotFoundError, ImportError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"[build_database] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
