"""Curated business-context seed for the packaged AdventureWorks-derived schema.

This is the *only* hand-written business knowledge in the system. Everything
else (new tables, new columns added later) is discovered structurally by
:mod:`app.metadata.discovery` and, when a description is missing, filled in
either heuristically or by an LLM enrichment call -- see
:mod:`app.metadata.store`. Curated entries here always take precedence.
"""

from __future__ import annotations

from typing import Any

SEED_BUSINESS_CONTEXT: dict[str, Any] = {
    "tables": {
        "DimProduct": {
            "description": (
                "Product dimension. One row per product sold by AdventureWorks, "
                "including pricing, cost, physical attributes and lifecycle status."
            ),
            "columns": {
                "ProductKey": "Surrogate primary key uniquely identifying a product.",
                "ProductAlternateKey": "Natural business product code (SKU) from the source system.",
                "ProductSubcategoryKey": "Key of the product subcategory this product belongs to (grouping key).",
                "ProductName": "Customer-facing product name.",
                "ModelName": "Product model family name; groups size/color variants of the same model.",
                "ProductLine": "High-level product line code (e.g. R=Road, M=Mountain, T=Touring, S=Standard).",
                "Class": "Quality tier classification of the product (e.g. L=Low, M=Medium, H=High).",
                "Style": "Target fit/style of the product (e.g. U=Unisex, M=Men's, W=Women's).",
                "Color": "Product color.",
                "Size": "Product size label.",
                "SizeRange": "Product size range/category.",
                "Weight": "Product weight.",
                "StandardCost": "Standard (budgeted) manufacturing cost per unit.",
                "ListPrice": "Manufacturer's suggested retail list price per unit.",
                "DealerPrice": "Wholesale price offered to dealers/resellers.",
                "SafetyStockLevel": "Inventory planning safety-stock threshold.",
                "ReorderPoint": "Inventory level at which the product should be reordered.",
                "DaysToManufacture": "Lead time in days required to manufacture one unit.",
                "FinishedGoodsFlag": "1 if the product is a sellable finished good, 0 if it is a component.",
                "ProductDescription": "Marketing description of the product.",
                "StartDate": "Date this product record became effective.",
                "EndDate": "Date this product record stopped being effective (NULL if still active).",
                "Status": "Current lifecycle status of the product (e.g. Current, Discontinued).",
            },
        },
        "FactInternetSales": {
            "description": (
                "Direct-to-consumer online ('Internet channel') sales order lines. "
                "Grain: one row per sales order line."
            ),
            "columns": {
                "InternetSalesKey": "Surrogate primary key of the sales order line.",
                "ProductKey": "Product sold on this line (foreign key to DimProduct).",
                "CustomerKey": "Identifier of the purchasing consumer customer.",
                "SalesTerritoryKey": "Identifier of the sales territory/region attributed to the order.",
                "PromotionKey": "Identifier of the promotion/discount campaign applied, if any.",
                "CurrencyKey": "Identifier of the transaction currency.",
                "SalesOrderNumber": "Business order number.",
                "SalesOrderLineNumber": "Line number within the order.",
                "OrderDateKey": "Integer YYYYMMDD key for the order date.",
                "DueDateKey": "Integer YYYYMMDD key for the due date.",
                "ShipDateKey": "Integer YYYYMMDD key for the ship date.",
                "OrderDate": "Calendar date the order was placed.",
                "DueDate": "Calendar date the order was due.",
                "ShipDate": "Calendar date the order shipped.",
                "OrderQuantity": "Number of units ordered on this line.",
                "UnitPrice": "Price charged per unit.",
                "ExtendedAmount": "UnitPrice * OrderQuantity before discount.",
                "UnitPriceDiscountPct": "Discount percentage applied to the unit price.",
                "DiscountAmount": "Discount amount applied to the line.",
                "ProductStandardCost": "Standard cost per unit at time of sale.",
                "TotalProductCost": "Total standard cost for the line (ProductStandardCost * OrderQuantity).",
                "SalesAmount": "Net revenue recognized for this line. Primary revenue measure.",
                "TaxAmt": "Tax charged on the line.",
                "Freight": "Freight/shipping charge on the line.",
                "CarrierTrackingNumber": "Shipping carrier tracking number.",
                "CustomerPONumber": "Customer purchase-order reference number.",
            },
            "aggregation_overrides": {
                "UnitPrice": "avg",
                "UnitPriceDiscountPct": "avg",
                "TotalProductCost": "sum",
            },
            "default_measure": "SalesAmount",
        },
        "FactResellerSales": {
            "description": (
                "B2B sales order lines sold through reseller/dealer partners "
                "('Reseller channel'). Grain: one row per sales order line."
            ),
            "columns": {
                "ResellerSalesKey": "Surrogate primary key of the sales order line.",
                "ProductKey": "Product sold on this line (foreign key to DimProduct).",
                "ResellerKey": "Identifier of the reseller/dealer that purchased the product.",
                "EmployeeKey": "Identifier of the AdventureWorks sales employee who owns the order.",
                "SalesTerritoryKey": "Identifier of the sales territory/region attributed to the order.",
                "PromotionKey": "Identifier of the promotion/discount campaign applied, if any.",
                "CurrencyKey": "Identifier of the transaction currency.",
                "SalesOrderNumber": "Business order number.",
                "SalesOrderLineNumber": "Line number within the order.",
                "OrderDateKey": "Integer YYYYMMDD key for the order date.",
                "DueDateKey": "Integer YYYYMMDD key for the due date.",
                "ShipDateKey": "Integer YYYYMMDD key for the ship date.",
                "OrderDate": "Calendar date the order was placed.",
                "DueDate": "Calendar date the order was due.",
                "ShipDate": "Calendar date the order shipped.",
                "OrderQuantity": "Number of units ordered on this line.",
                "UnitPrice": "Price charged per unit.",
                "ExtendedAmount": "UnitPrice * OrderQuantity before discount.",
                "UnitPriceDiscountPct": "Discount percentage applied to the unit price.",
                "DiscountAmount": "Discount amount applied to the line.",
                "ProductStandardCost": "Standard cost per unit at time of sale.",
                "TotalProductCost": "Total standard cost for the line (ProductStandardCost * OrderQuantity).",
                "SalesAmount": "Net revenue recognized for this line. Primary revenue measure.",
                "TaxAmt": "Tax charged on the line.",
                "Freight": "Freight/shipping charge on the line.",
                "CarrierTrackingNumber": "Shipping carrier tracking number.",
                "CustomerPONumber": "Customer purchase-order reference number.",
            },
            "aggregation_overrides": {
                "UnitPrice": "avg",
                "UnitPriceDiscountPct": "avg",
                "TotalProductCost": "sum",
            },
            "default_measure": "SalesAmount",
        },
    },
    "glossary": {
        "revenue": "SalesAmount in FactInternetSales / FactResellerSales.",
        "sales channel": "'Internet' = FactInternetSales (direct consumer), 'Reseller' = FactResellerSales (B2B/dealer).",
        "profit": "Approximated as SalesAmount - TotalProductCost (no explicit profit column is stored).",
        "customer": "Consumer customers only exist as CustomerKey on FactInternetSales; no customer dimension is loaded in this deployment.",
    },
}
