"""Conservative deterministic repairs for unambiguous ranking query shapes."""

from __future__ import annotations

import re

from sqlglot import exp


def ranking_request(question: str) -> tuple[bool, bool, int] | None:
    normalized = " ".join((question or "").casefold().split())
    normalized = re.sub(r"\bat\s+(?:least|most)\b", "", normalized)
    high = bool(
        re.search(r"\b(top|most|highest|largest|greatest|best|latest|newest)\b", normalized)
    )
    low = bool(
        re.search(r"\b(bottom|least|lowest|smallest|worst|earliest|oldest)\b", normalized)
    )
    if high == low:
        return None
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", normalized)
    return high, low, int(match.group(1)) if match else 1


def question_group_words(question: str) -> set[str]:
    """Extract likely partition concepts across common natural-language forms."""
    normalized = " ".join((question or "").casefold().split())
    words: set[str] = set()
    for phrase in re.findall(
        r"\b(?:by|per|for (?:each|every)|within (?:each|every)|"
        r"in (?:each|every)|each|every)\s+"
        r"([a-z0-9_]+(?:\s+[a-z0-9_]+){0,3})",
        normalized,
    ):
        words.update(re.findall(r"[a-z0-9]+", phrase))
    temporal_adverbs = {
        "daily": "day",
        "weekly": "week",
        "monthly": "month",
        "quarterly": "quarter",
        "yearly": "year",
        "annually": "year",
    }
    for adverb, group in temporal_adverbs.items():
        if re.search(rf"\b{adverb}\b", normalized):
            words.add(group)
    return words


def asks_for_per_group_ranking(question: str) -> bool:
    normalized = " ".join((question or "").casefold().split())
    if re.search(r"\b(?:each|every)\b", normalized):
        return True
    known_groups = {
        "day", "week", "month", "quarter", "year", "category", "group",
        "segment", "region", "country", "channel", "department", "team",
        "customer", "account", "product", "item", "store", "branch",
        "location", "territory", "class", "type",
    }
    return bool(question_group_words(question) & known_groups)


def _literal_limit(select: exp.Select) -> int | None:
    limit = select.args.get("limit")
    try:
        return int(limit.expression.this) if limit is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def repair_unambiguous_ranking(
    expression: exp.Expression,
    question: str,
) -> tuple[exp.Expression, list[str]]:
    """Add global ranking ORDER/LIMIT only when the metric is unambiguous.

    Per-group rankings are deliberately excluded: manufacturing a window and
    partition from incomplete SQL would require guessing user semantics. This
    repair handles only a global query with exactly one aggregate metric.
    """
    request = ranking_request(question)
    if request is None or asks_for_per_group_ranking(question):
        return expression, []
    if any(True for _ in expression.find_all(exp.Window)):
        return expression, []
    high, _low, requested = request
    root = expression if isinstance(expression, exp.Select) else None
    if root is None:
        return expression, []

    metric_projections = [
        projection
        for projection in root.expressions
        if projection.find(exp.AggFunc) is not None
    ]
    if len(metric_projections) != 1:
        return expression, []

    repaired = expression.copy()
    repaired_root = repaired if isinstance(repaired, exp.Select) else None
    if repaired_root is None:
        return expression, []
    notes: list[str] = []
    metric = next(
        projection
        for projection in repaired_root.expressions
        if projection.find(exp.AggFunc) is not None
    )
    metric_value = metric.this if isinstance(metric, exp.Alias) else metric
    order_expression: exp.Expression
    if isinstance(metric, exp.Alias) and metric.alias:
        order_expression = exp.column(metric.alias)
        metric_names = {metric.alias.casefold()}
    else:
        order_expression = metric_value.copy()
        metric_names = {metric_value.sql(dialect="sqlite").casefold()}

    order = repaired_root.args.get("order")
    ordered = list(order.expressions) if order is not None else []
    first = ordered[0] if ordered else None
    first_value = first.this if isinstance(first, exp.Ordered) else first
    first_name = (
        first_value.name.casefold()
        if isinstance(first_value, exp.Column)
        else first_value.sql(dialect="sqlite").casefold() if first_value is not None else ""
    )
    correct_direction = bool(first.args.get("desc")) == high if first is not None else False
    if first_name not in metric_names or not correct_direction:
        tie_breakers = [
            item
            for item in ordered
            if (
                item.this.name.casefold()
                if isinstance(item.this, exp.Column)
                else item.this.sql(dialect="sqlite").casefold()
            ) not in metric_names
        ]
        repaired_root.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(this=order_expression, desc=high),
                    *tie_breakers,
                ]
            ),
        )
        notes.append(
            "Applied descending metric ordering for the requested highest ranking."
            if high
            else "Applied ascending metric ordering for the requested lowest ranking."
        )

    current_limit = _literal_limit(repaired_root)
    if current_limit != requested:
        repaired_root.set(
            "limit", exp.Limit(expression=exp.Literal.number(requested))
        )
        notes.append(
            f"Applied LIMIT {requested} for the requested global ranking."
        )
    return repaired, notes
