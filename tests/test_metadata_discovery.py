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
