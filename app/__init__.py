"""AI Analyst backend package.

All LLM calls, metadata processing, SQL generation/validation/execution,
data processing and visualization live in this package. The Streamlit
app (``ui/streamlit_app.py``) only calls into :mod:`app.orchestrator`
and renders the result.
"""
