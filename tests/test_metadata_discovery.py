from app.db.connection import readonly_connection
from app.metadata import discovery, store


def test_discover_schema_classifies_tables_correctly():
    with readonly_connection() as conn:
        tables = discovery.discover_schema(conn)

    assert tables["DimProduct"].kind == "dimension"
    assert tables["FactInternetSales"].kind == "fact"
    assert tables["FactResellerSales"].kind == "fact"


def test_foreign_keys_detected():
    with readonly_connection() as conn:
        tables = discovery.discover_schema(conn)

    product_key_col = next(
        c for c in tables["FactInternetSales"].columns if c.name == "ProductKey"
    )
    assert product_key_col.is_foreign_key
    assert product_key_col.references == {"table": "DimProduct", "column": "ProductKey"}


def test_schema_signature_stable_across_calls():
    with readonly_connection() as conn:
        tables_a = discovery.discover_schema(conn)
        tables_b = discovery.discover_schema(conn)

    assert discovery.schema_signature(tables_a) == discovery.schema_signature(tables_b)


def test_measure_columns_get_sensible_default_aggregation():
    with readonly_connection() as conn:
        tables = discovery.discover_schema(conn)

    cols = {c.name: c for c in tables["FactInternetSales"].columns}
    assert cols["SalesAmount"].semantic_role == "measure"
    assert cols["SalesAmount"].default_aggregation == "sum"
    # Per-unit price columns should default to averaging, not summing.
    assert cols["UnitPrice"].default_aggregation == "avg"
    # Key columns should never be treated as measures.
    assert cols["ProductKey"].semantic_role == "key"
    assert cols["ProductKey"].default_aggregation is None


def test_build_metadata_produces_relationships_and_aggregation_rules():
    with readonly_connection() as conn:
        metadata, _ = store.build_metadata(conn)

    rel_pairs = {
        (r["from_table"], r["from_column"], r["to_table"], r["to_column"])
        for r in metadata["relationships"]
    }
    assert ("FactInternetSales", "ProductKey", "DimProduct", "ProductKey") in rel_pairs
    assert ("FactResellerSales", "ProductKey", "DimProduct", "ProductKey") in rel_pairs

    assert "SalesAmount" in metadata["aggregation_rules"]["FactInternetSales"]["measures"]
    assert metadata["aggregation_rules"]["FactInternetSales"]["default_measure"] == "SalesAmount"


def test_refresh_if_needed_is_cached_when_schema_unchanged():
    with readonly_connection() as conn:
        _, rebuilt_first = store.refresh_if_needed(conn, force=True)
        _, rebuilt_second = store.refresh_if_needed(conn, force=False)

    assert rebuilt_first is True
    assert rebuilt_second is False


def test_untyped_columns_values_and_view_lineage_are_discovered_generically():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE entities (entity_id INTEGER PRIMARY KEY, display_name TEXT);
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY,
            entity_id INTEGER,
            happened,
            gross_amount,
            payload BLOB
        );
        INSERT INTO entities VALUES (1, 'One'), (2, 'Two');
        INSERT INTO events VALUES
            (10, 1, '2026-01-01T10:00:00', 12.5, X'01'),
            (11, 2, '2026-01-02T11:00:00', 7.5, X'02');
        CREATE VIEW recent_events AS SELECT event_id, entity_id, happened, gross_amount
            FROM events WHERE happened >= '2026-01-02';
        """
    )

    tables = discovery.discover_schema(conn)
    event_columns = {column.name: column for column in tables["events"].columns}

    assert event_columns["gross_amount"].observed_value_family == "numeric"
    assert event_columns["gross_amount"].semantic_role == "measure"
    assert event_columns["happened"].observed_value_family == "temporal_text"
    assert event_columns["happened"].semantic_role == "temporal"
    assert event_columns["payload"].observed_value_family == "blob"
    assert event_columns["entity_id"].references == {
        "table": "entities",
        "column": "entity_id",
    }
    assert tables["entities"].kind == "dimension"
    assert tables["events"].kind == "fact"
    assert tables["recent_events"].object_type == "view"
    assert tables["recent_events"].depends_on == ["events"]
    conn.close()


def test_name_match_without_data_overlap_is_not_treated_as_a_join():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE people (person_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE messages (message_id INTEGER PRIMARY KEY, person_id INTEGER, body TEXT);
        INSERT INTO people VALUES (1, 'One'), (2, 'Two');
        INSERT INTO messages VALUES (10, 900, 'orphan'), (11, 901, 'orphan');
        """
    )

    tables = discovery.discover_schema(conn)
    person_id = next(
        column for column in tables["messages"].columns if column.name == "person_id"
    )
    assert person_id.references is None
    assert person_id.is_foreign_key is False
    conn.close()
import sqlite3
