from scripts import build_database


def test_fact_loaders_reject_incomplete_source_rows():
    internet = build_database._load_fact(
        "FactInternetSales.csv", ["CustomerKey"], ["CustomerKey"]
    )
    reseller = build_database._load_fact(
        "FactResellerSales.csv",
        ["ResellerKey", "EmployeeKey"],
        ["ResellerKey"],
    )

    required_common = [
        "ProductKey",
        "SalesOrderNumber",
        "SalesOrderLineNumber",
        "OrderDate",
        "OrderQuantity",
        "SalesAmount",
    ]
    assert internet[[*required_common, "CustomerKey"]].notna().all().all()
    assert reseller[[*required_common, "ResellerKey"]].notna().all().all()
