#!/usr/bin/env python3
"""Build the sample analytics SQLite database from the packaged raw CSVs.

Source data: a trimmed extract of the classic AdventureWorks DW dataset
(``data/raw/*.csv``) -- one product dimension and two fact tables (direct
"internet" consumer sales and B2B "reseller" sales), related through
``ProductKey``. This gives the platform a realistic star-schema to reason
about out of the box.

Run:
    python scripts/build_database.py [--force]

The script is idempotent: it always rebuilds the DB file from the CSVs so
the packaged dataset and the live database never drift apart.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "ai_analyst.db"

DATE_COLUMNS = ["OrderDate", "DueDate", "ShipDate"]

DIM_PRODUCT_SCHEMA = """
CREATE TABLE DimProduct (
    ProductKey              INTEGER PRIMARY KEY,
    ProductAlternateKey     TEXT,
    ProductSubcategoryKey   INTEGER,
    ProductName             TEXT NOT NULL,
    ModelName                TEXT,
    ProductLine              TEXT,
    Class                     TEXT,
    Style                     TEXT,
    Color                     TEXT,
    Size                      TEXT,
    SizeRange                 TEXT,
    Weight                    REAL,
    StandardCost               REAL,
    ListPrice                  REAL,
    DealerPrice                REAL,
    SafetyStockLevel           INTEGER,
    ReorderPoint                INTEGER,
    DaysToManufacture           INTEGER,
    FinishedGoodsFlag            INTEGER,
    ProductDescription            TEXT,
    StartDate                     TEXT,
    EndDate                        TEXT,
    Status                          TEXT
);
"""

FACT_INTERNET_SALES_SCHEMA = """
CREATE TABLE FactInternetSales (
    InternetSalesKey       INTEGER PRIMARY KEY AUTOINCREMENT,
    ProductKey              INTEGER NOT NULL REFERENCES DimProduct(ProductKey),
    CustomerKey              INTEGER NOT NULL,
    SalesTerritoryKey         INTEGER,
    PromotionKey               INTEGER,
    CurrencyKey                 INTEGER,
    SalesOrderNumber              TEXT NOT NULL,
    SalesOrderLineNumber           INTEGER NOT NULL,
    OrderDateKey                    INTEGER,
    DueDateKey                       INTEGER,
    ShipDateKey                       INTEGER,
    OrderDate                          TEXT,
    DueDate                             TEXT,
    ShipDate                             TEXT,
    OrderQuantity                         INTEGER,
    UnitPrice                              REAL,
    ExtendedAmount                          REAL,
    UnitPriceDiscountPct                     REAL,
    DiscountAmount                            REAL,
    ProductStandardCost                        REAL,
    TotalProductCost                            REAL,
    SalesAmount                                  REAL,
    TaxAmt                                        REAL,
    Freight                                        REAL,
    CarrierTrackingNumber                           TEXT,
    CustomerPONumber                                 TEXT
);
"""

FACT_RESELLER_SALES_SCHEMA = """
CREATE TABLE FactResellerSales (
    ResellerSalesKey        INTEGER PRIMARY KEY AUTOINCREMENT,
    ProductKey               INTEGER NOT NULL REFERENCES DimProduct(ProductKey),
    ResellerKey                INTEGER NOT NULL,
    EmployeeKey                  INTEGER,
    SalesTerritoryKey              INTEGER,
    PromotionKey                     INTEGER,
    CurrencyKey                        INTEGER,
    SalesOrderNumber                     TEXT NOT NULL,
    SalesOrderLineNumber                   INTEGER NOT NULL,
    OrderDateKey                             INTEGER,
    DueDateKey                                 INTEGER,
    ShipDateKey                                  INTEGER,
    OrderDate                                      TEXT,
    DueDate                                          TEXT,
    ShipDate                                           TEXT,
    OrderQuantity                                        INTEGER,
    UnitPrice                                              REAL,
    ExtendedAmount                                           REAL,
    UnitPriceDiscountPct                                       REAL,
    DiscountAmount                                               REAL,
    ProductStandardCost                                            REAL,
    TotalProductCost                                                 REAL,
    SalesAmount                                                        REAL,
    TaxAmt                                                              REAL,
    Freight                                                              REAL,
    CarrierTrackingNumber                                                 TEXT,
    CustomerPONumber                                                        TEXT
);
"""

INDEXES = [
    "CREATE INDEX ix_product_subcategory ON DimProduct(ProductSubcategoryKey)",
    "CREATE INDEX ix_ifs_product ON FactInternetSales(ProductKey)",
    "CREATE INDEX ix_ifs_customer ON FactInternetSales(CustomerKey)",
    "CREATE INDEX ix_ifs_orderdate ON FactInternetSales(OrderDate)",
    "CREATE INDEX ix_frs_product ON FactResellerSales(ProductKey)",
    "CREATE INDEX ix_frs_reseller ON FactResellerSales(ResellerKey)",
    "CREATE INDEX ix_frs_orderdate ON FactResellerSales(OrderDate)",
]


def _load_dim_product() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "DimProduct.csv")
    df = df.rename(
        columns={
            "EnglishProductName": "ProductName",
            "EnglishDescription": "ProductDescription",
        }
    )
    keep = [
        "ProductKey",
        "ProductAlternateKey",
        "ProductSubcategoryKey",
        "ProductName",
        "ModelName",
        "ProductLine",
        "Class",
        "Style",
        "Color",
        "Size",
        "SizeRange",
        "Weight",
        "StandardCost",
        "ListPrice",
        "DealerPrice",
        "SafetyStockLevel",
        "ReorderPoint",
        "DaysToManufacture",
        "FinishedGoodsFlag",
        "ProductDescription",
        "StartDate",
        "EndDate",
        "Status",
    ]
    df = df[keep].copy()
    df["FinishedGoodsFlag"] = df["FinishedGoodsFlag"].astype(int)
    for col in ["StartDate", "EndDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _load_fact(
    filename: str,
    extra_cols: list[str],
    required_extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename)
    base_cols = [
        "ProductKey",
        *extra_cols,
        "SalesTerritoryKey",
        "PromotionKey",
        "CurrencyKey",
        "SalesOrderNumber",
        "SalesOrderLineNumber",
        "OrderDateKey",
        "DueDateKey",
        "ShipDateKey",
        "OrderDate",
        "DueDate",
        "ShipDate",
        "OrderQuantity",
        "UnitPrice",
        "ExtendedAmount",
        "UnitPriceDiscountPct",
        "DiscountAmount",
        "ProductStandardCost",
        "TotalProductCost",
        "SalesAmount",
        "TaxAmt",
        "Freight",
        "CarrierTrackingNumber",
        "CustomerPONumber",
    ]
    df = df[base_cols].copy()
    required_cols = [
        "ProductKey",
        "SalesOrderNumber",
        "SalesOrderLineNumber",
        "OrderDate",
        "OrderQuantity",
        "SalesAmount",
        *(required_extra_cols or extra_cols),
    ]
    # Raw extracts can contain partial lines that violate the fact table's
    # required business keys/measures. Exclude them before SQLite insertion.
    df = df.dropna(subset=required_cols).copy()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def build(force: bool) -> None:
    if DB_PATH.exists():
        if not force:
            print(f"[build_database] {DB_PATH} already exists; use --force to rebuild.")
            return
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[build_database] reading CSVs ...")
    dim_product = _load_dim_product()
    fact_internet = _load_fact(
        "FactInternetSales.csv", ["CustomerKey"], ["CustomerKey"]
    )
    fact_reseller = _load_fact(
        "FactResellerSales.csv", ["ResellerKey", "EmployeeKey"], ["ResellerKey"]
    )

    print("[build_database] creating schema ...")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            DIM_PRODUCT_SCHEMA + FACT_INTERNET_SALES_SCHEMA + FACT_RESELLER_SALES_SCHEMA
        )

        print(f"[build_database] loading DimProduct ({len(dim_product)} rows) ...")
        dim_product.to_sql("DimProduct", conn, if_exists="append", index=False)

        print(f"[build_database] loading FactInternetSales ({len(fact_internet)} rows) ...")
        fact_internet.to_sql("FactInternetSales", conn, if_exists="append", index=False)

        print(f"[build_database] loading FactResellerSales ({len(fact_reseller)} rows) ...")
        fact_reseller.to_sql("FactResellerSales", conn, if_exists="append", index=False)

        print("[build_database] creating indexes ...")
        for stmt in INDEXES:
            conn.execute(stmt)

        conn.execute("ANALYZE;")
        conn.commit()
    finally:
        conn.close()

    print(f"[build_database] done -> {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the database file already exists."
    )
    args = parser.parse_args()
    try:
        build(force=args.force)
    except FileNotFoundError as exc:
        print(f"[build_database] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
