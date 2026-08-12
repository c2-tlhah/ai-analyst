from app.sql.validator import validate_sql

ALLOWED = {"DimProduct", "FactInternetSales", "FactResellerSales"}


def test_simple_select_is_valid():
    result = validate_sql("SELECT * FROM DimProduct", ALLOWED, max_rows=100)
    assert result.is_valid
    assert "LIMIT 100" in result.sanitized_sql


def test_join_with_aggregation_is_valid():
    sql = (
        "SELECT p.ProductLine, SUM(f.SalesAmount) AS total "
        "FROM FactInternetSales f JOIN DimProduct p ON f.ProductKey = p.ProductKey "
        "GROUP BY p.ProductLine ORDER BY total DESC"
    )
    result = validate_sql(sql, ALLOWED, max_rows=50)
    assert result.is_valid
    assert "LIMIT 50" in result.sanitized_sql


def test_cte_is_valid_and_alias_not_flagged_as_table():
    sql = "WITH recent AS (SELECT * FROM FactInternetSales) SELECT * FROM recent"
    result = validate_sql(sql, ALLOWED, max_rows=10)
    assert result.is_valid


def test_existing_limit_is_capped_not_expanded():
    result = validate_sql("SELECT * FROM DimProduct LIMIT 999999", ALLOWED, max_rows=25)
    assert result.is_valid
    assert "LIMIT 25" in result.sanitized_sql


def test_existing_limit_below_cap_is_preserved():
    result = validate_sql("SELECT * FROM DimProduct LIMIT 5", ALLOWED, max_rows=1000)
    assert result.is_valid
    assert "LIMIT 5" in result.sanitized_sql


def test_delete_is_rejected():
    result = validate_sql("DELETE FROM DimProduct", ALLOWED, max_rows=100)
    assert not result.is_valid


def test_drop_table_is_rejected():
    result = validate_sql("DROP TABLE DimProduct", ALLOWED, max_rows=100)
    assert not result.is_valid


def test_pragma_is_rejected():
    result = validate_sql("PRAGMA table_info(DimProduct)", ALLOWED, max_rows=100)
    assert not result.is_valid


def test_attach_is_rejected():
    result = validate_sql("ATTACH DATABASE 'evil.db' AS evil", ALLOWED, max_rows=100)
    assert not result.is_valid


def test_multiple_statements_rejected():
    result = validate_sql(
        "SELECT * FROM DimProduct; DROP TABLE DimProduct;", ALLOWED, max_rows=100
    )
    assert not result.is_valid


def test_unauthorized_table_rejected():
    result = validate_sql("SELECT * FROM sqlite_master", ALLOWED, max_rows=100)
    assert not result.is_valid
    assert any("sqlite_master" in e for e in result.errors)


def test_unauthorized_table_in_join_rejected():
    sql = "SELECT * FROM DimProduct JOIN SecretTable ON 1=1"
    result = validate_sql(sql, ALLOWED, max_rows=100)
    assert not result.is_valid


def test_empty_sql_rejected():
    result = validate_sql("", ALLOWED, max_rows=100)
    assert not result.is_valid


def test_insert_disguised_in_subquery_rejected():
    result = validate_sql(
        "SELECT * FROM DimProduct WHERE ProductKey IN (INSERT INTO DimProduct VALUES (1))",
        ALLOWED,
        max_rows=100,
    )
    assert not result.is_valid


def test_non_sqlite_dialect_errors_are_actionable():
    sql = (
        "SELECT TOP(10) DimProduct.ProductName, IssueType::InternetSales, "
        "FROM DimProduct"
    )
    result = validate_sql(sql, ALLOWED, max_rows=100)

    assert not result.is_valid
    combined = " ".join(result.errors)
    assert "does not support TOP" in combined
    assert "does not support PostgreSQL-style ::" in combined
    assert "trailing comma" in combined


def test_download_query_has_a_separate_larger_safety_cap():
    result = validate_sql(
        "SELECT * FROM FactInternetSales",
        ALLOWED,
        max_rows=10,
        download_max_rows=100,
    )

    assert result.is_valid
    assert "LIMIT 10" in result.sanitized_sql
    assert "LIMIT 101" in result.download_sql


def test_download_query_preserves_an_explicit_user_limit():
    result = validate_sql(
        "SELECT * FROM FactInternetSales LIMIT 5",
        ALLOWED,
        max_rows=10,
        download_max_rows=100,
    )

    assert result.is_valid
    assert "LIMIT 5" in result.sanitized_sql
    assert "LIMIT 5" in result.download_sql
