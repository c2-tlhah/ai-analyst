# AI Analyst

A schema-aware, extensible AI analytics platform: ask a question in plain English,
get a validated read-only SQL query, a result table, an AI-generated chart, and a
concise natural-language insight -- all backed by deterministic security/validation
and orchestrated with LangGraph.

Streamlit is presentation-only. Every LLM call, metadata lookup, SQL generation/
validation/execution, Pandas processing, and chart-generation step happens in the
`app/` backend.

## Architecture

```
question
   -> understand_intent        (LLM)   classify + guess relevant tables
   -> retrieve_metadata         (deterministic)  trim schema to what's relevant
   -> generate_sql              (LLM)   propose a SELECT statement
   -> validate_sql               (deterministic)  security/allow-list gate
        |-- invalid --> handle_error --(retries left)--> generate_sql  [loop]
        |                             --(exhausted)---->  give_up -> END
   -> execute_sql                (deterministic)  read-only, limited, timed
        |-- failed --> handle_error --(retries left)--> generate_sql  [loop]
        |                            --(exhausted)---->  give_up -> END
   -> analyze_results            (LLM)   turn a DataFrame summary into an insight
   -> plan_visualization         (LLM)   propose chart_type/x/y/color/agg
   -> generate_chart             (deterministic)  Plotly, from a fixed function set
   -> respond -> END
```

See `app/graph/workflow.py` for the LangGraph wiring and `app/graph/nodes.py` for
each node.

### Modules

| Concern | Module |
|---|---|
| Config / env vars | `app/config.py` |
| Logging | `app/logging_config.py` |
| Read-only DB connection | `app/db/connection.py` |
| SQL execution (limits, timeout) | `app/db/executor.py` |
| Schema discovery | `app/metadata/discovery.py` |
| Metadata persistence + change detection | `app/metadata/store.py` |
| Curated business context (seed) | `app/metadata/business_context_seed.py` |
| LLM-assisted description of new tables | `app/metadata/enrichment.py` |
| Relevance-based metadata retrieval (lexical + vector/RAG) | `app/metadata/retrieval.py` |
| Local vector knowledge base for schema RAG (Chroma) | `app/metadata/vector_store.py` |
| Azure AI Foundry + Ollama + OpenRouter clients (structured JSON) | `app/llm/client.py` |
| Structured LLM output schemas | `app/llm/schemas.py` |
| SQL generation (LLM) | `app/sql/generator.py` |
| SQL security/validation | `app/sql/validator.py` |
| Result -> insight (LLM) | `app/analysis/insights.py` |
| Visualization plan (LLM) + sanitizer | `app/viz/planner.py` |
| Chart rendering (Plotly, controlled) | `app/viz/renderer.py` |
| LangGraph state/nodes/graph | `app/graph/` |
| Entrypoint for the UI | `app/orchestrator.py` |
| Streamlit UI (presentation only) | `ui/streamlit_app.py` |

## The metadata / context layer

`app/metadata/discovery.py` introspects the live SQLite schema (tables, columns,
types, primary/foreign keys, row counts, and small samples of low-cardinality
categorical columns) with no LLM involvement, and heuristically infers each
column's semantic role (`key`, `measure`, `temporal`, `categorical_attribute`, ...)
and a sensible default aggregation (`sum` for additive amounts, `avg` for
per-unit prices/rates).

`app/metadata/store.py` merges that with a business-context layer (human-curated
seed descriptions in `app/metadata/business_context_seed.py`, falling back to
LLM-generated or humanized-name descriptions for anything not curated) and
persists the result as JSON:

* `metadata_store/schema_metadata.json` -- the full merged metadata the app reads.
* `metadata_store/business_context.json` -- just the descriptions/glossary, which
  survive schema rebuilds so curation accumulates instead of being regenerated.

A structural hash of the schema is stored alongside the metadata. On every
question, `app.orchestrator.refresh_metadata()` re-hashes the live schema; if
nothing changed it's a no-op, and if a table/column was added or changed it
automatically re-discovers and updates the store (calling the LLM only to
describe what's new, never to redo work already cached).

`app/metadata/retrieval.py` then picks which tables are relevant to the
question and returns only that slice -- expanded to include anything
reachable via a foreign key so joins stay possible. This is what keeps the
LLM's context small and scoped instead of dumping the whole schema on every
call, and it works unmodified as more tables are added. Two selection
strategies are available and share the same output shape:

* **Vector/RAG** (preferred) -- nearest-neighbor search over table documents
  embedded in a local Chroma collection (`app/metadata/vector_store.py`).
  Generalizes to paraphrased questions that share no literal keywords with
  the schema (e.g. "top sellers" matching a table described as "product
  sales facts").
* **Lexical** (fallback) -- keyword/token overlap against table/column
  names, descriptions, and sample values. Used automatically whenever no
  vector index exists yet for the active database (e.g. before the first
  connect) or the vector backend errors.

Each answer's response reports which strategy actually served it
(`retrieval_mode`, surfaced as a badge in the UI) so this is never a silent
behavior difference.

## Database connection & the RAG knowledge base

The sidebar's **Database connection** panel lets you point the app at any
SQLite file (a plain path, or a `sqlite:///...` connection string) without
restarting -- `AI_ANALYST_DB_PATH` in `.env` only sets the *initial* default.
Clicking **Connect & build knowledge base** (`app.orchestrator.connect_database`):

1. Validates the file opens as a real, read-only SQLite database.
2. Makes it the active database for every subsequent query (backed by
   `app.db.connection.set_active_database_path`) and drops the session's
   metadata/answer caches, since they described a different database.
3. Crawls its schema and describes it -- the same deterministic discovery +
   LLM-assisted enrichment used at startup (`app/metadata/discovery.py`,
   `app/metadata/enrichment.py`), so curated/cached descriptions are reused
   and only genuinely new tables/columns cost an LLM call.
4. Renders each table's full business-enriched metadata (description,
   columns, types, foreign keys, sample values) to a short text document and
   upserts it into a Chroma collection scoped to that database
   (`app/metadata/vector_store.py`, keyed by
   `app.db.connection.get_active_database_identity`) -- the "knowledge base"
   RAG retrieval searches at question time.

Embeddings run entirely on-machine via Chroma 1.5+'s bundled ONNX MiniLM
model -- no API key, no network call per query, and no LLM token cost, so
RAG retrieval is strictly additive to the app's token budget. The first
index build may download the local embedding model once. Set
`VECTOR_RAG_ENABLED=false` in `.env` to always use the lexical fallback
instead (see `.env.example` for `VECTOR_STORE_DIR`/`VECTOR_TOP_K`).

### Versioning -- schema changes never overwrite the old knowledge base

The knowledge base is versioned per database, one version per *actual*
schema change, not per refresh. `app/metadata/vector_store.py` compares the
live schema's structural hash (already computed by
`app/metadata/discovery.py`) against the last version it built:

* **Unchanged schema** (e.g. clicking **Refresh schema** with nothing new,
  or reconnecting to the same file) -- the current version's documents are
  refreshed in place. No new version, nothing forked.
* **Real change** (a table or column added/changed/removed) -- a new,
  numbered version is created (`schema_<db identity>_v<N>` as its own Chroma
  collection). Every earlier version's collection and text export are left
  exactly as they were -- nothing is deleted or silently replaced. RAG
  retrieval always searches the *latest* version.

The sidebar reports one of four explicit states: **ready**, **not built**,
**disabled**, or **failed** (including the backend error and recovery steps).
Its searchable **Knowledge base explorer** lets you select any version,
inspect scrollable per-table documents, and download the complete version as
plain text. The same indexed document is also available beside each table in
**Available data**, with a per-table download
(`app.orchestrator.list_knowledge_base_versions` /
`get_knowledge_base_documents`). Building deterministic schema documents does
not require a configured LLM; an available LLM only improves generated
descriptions.

### Text export

Alongside the Chroma collection, every version's per-table documents are
also written as plain `.txt` files under
`<VECTOR_STORE_DIR>/knowledge_base_txt/<db identity>/v<N>/` -- one file per
table plus a combined `_all_tables.txt` -- so you can inspect, `grep`, or
`diff` the knowledge base directly from a terminal or text editor, without
going through the app or a Chroma client at all.

## Performance, caching & session memory

Three independent caches keep repeated work out of the hot path -- all of
them live in `app/orchestrator.py` / `app/llm/client.py`, never in the UI:

* **Metadata session cache.** Re-hashing the live schema (PRAGMA
  `table_info`/`foreign_key_list` plus sample-value queries per table) is
  cheap once, but `refresh_metadata()` used to redo it before *every*
  question. It's now cached in-process for `_METADATA_CACHE_TTL_SECONDS`
  (5 minutes, see `app/orchestrator.py`) -- a session's worth of questions
  pays for schema discovery once. The sidebar's "Refresh schema" button
  forces an immediate re-check (`force=True`) if you change the database
  mid-session.
* **Exact-question answer cache.** Re-asking a question with the same selected
  provider and model already answered
  this run (`app.orchestrator._answer_cache`, a bounded LRU keyed on the
  normalized question text) returns the same validated `AnalysisResponse`
  instantly, skipping the DB round-trip and all four LLM calls. Only
  successful answers are cached -- a failed attempt is always retried for
  real. The UI surfaces this as a "🔁 served from session cache" badge.
* **Bounded conversation memory.** The last few `{question, sql}` turns
  from the session are threaded into the intent-understanding and
  SQL-generation prompts (`app/graph/nodes.py::_format_history`) so short
  follow-ups ("now break that down by year", "same but for resellers")
  don't need to restate context the model already has. This is metadata
  about *past turns*, not the schema itself -- the per-question schema
  trimming in `app/metadata/retrieval.py` is unaffected and still runs
  every time, since which tables are relevant is genuinely
  question-dependent.

On top of caching, `app/llm/client.py` records real token usage
(`prompt_tokens`/`completion_tokens`/`total_tokens`) straight from each
Azure AI Foundry or Ollama response -- not an estimate -- so the sidebar's "Session
efficiency" panel reflects actual spend, and a cache hit visibly costs zero
additional tokens and zero additional latency.

## Security model

`app/sql/validator.py` is a hard, deterministic gate between LLM output and the
database (see its docstring for the full check list): exactly one statement,
must parse as a read-only query (`SELECT`/CTEs/set operations -- anything else
is rejected), a keyword blocklist as defense-in-depth, every referenced table
checked against an allow-list, and an enforced/capped `LIMIT`. On top of that,
the DB connection itself is opened SQLite `mode=ro` (plus `PRAGMA query_only`),
and query execution has a wall-clock timeout via SQLite's progress handler.

The LLM never generates chart code either -- only a structured
`VisualizationPlan` (chart type + column names), which `app/viz/planner.py`
sanitizes against the actual result columns before `app/viz/renderer.py` (a
fixed set of Plotly Express calls) ever touches it.

### Interactive result exploration

Each successful answer shows the retrieved table and CSV download beside its
generated SQL. The data grid and SQL viewer use matching fixed-height,
independently scrollable panels. The **Further analysis** workspace underneath
supports two paths:

* **Build a graph** includes a one-click general visualization recommended by
  the selected AI, a free-text AI request for specific needs (for example,
  "monthly sales as a line, split by channel"), and deterministic manual
  controls as a fallback. Every AI plan is validated against the retrieved
  columns before rendering. Date results can be regrouped by day, week, month,
  quarter, or year without rerunning SQL.
* **Ask a follow-up** sends the selected answer's question, SQL, result columns,
  and row count as conversation context. This is useful when the original result
  is too aggregated to graph—for example, ask for the same annual sales broken
  down monthly so the new result contains a date dimension.

Interactive chart requests are implemented in `app/viz/explorer.py`; invalid
axes, unsupported chart types, non-date time grouping, and unusable scalar
results return user-facing validation messages instead of UI exceptions.

## The sample dataset

`data/raw/*.csv` is a trimmed extract of the classic AdventureWorks DW sample
data: **one dimension table** (`DimProduct`) and **two fact tables**
(`FactInternetSales` -- direct consumer sales, `FactResellerSales` -- B2B/dealer
sales), related through `ProductKey`. `scripts/build_database.py` loads it into
`data/ai_analyst.db` (SQLite) with explicit types, primary/foreign keys, and
indexes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with Azure AI Foundry, Ollama, and/or OpenRouter settings (see below)

python scripts/build_database.py   # builds data/ai_analyst.db from data/raw/*.csv
```

### Azure AI Foundry (Kimi K2.6)

The app talks to a Kimi K2.6 deployment on Azure AI Foundry via the Azure AI
Model Inference API (`azure-ai-inference`), which is what every chat-completion
model on Foundry -- regardless of vendor -- speaks. Deploy Kimi K2.6 as a
serverless (Model-as-a-Service) endpoint in your Foundry project, then set:

```
AZURE_FOUNDRY_ENDPOINT=https://<your-resource>.services.ai.azure.com/models
AZURE_FOUNDRY_API_KEY=<your-key>
AZURE_FOUNDRY_MODEL=Kimi-K2.6
```

in `.env` (see `.env.example`). If you paste in a Foundry *project* endpoint
instead (`.../api/projects/<project>`, the form the portal often shows for the
`azure-ai-projects`/agents SDKs), `app/config.py` normalizes it to the
Model Inference API root automatically.

Structured outputs (intent, SQL, insight, viz plan) are enforced via
JSON-schema prompting + Pydantic validation with one bounded repair retry
(`app/llm/client.py`), rather than relying on provider-specific function-calling
support -- this keeps the client portable across whatever Foundry model you
point it at.

**Kimi K2.6 is a reasoning model**: it emits a hidden `reasoning_content`
chain-of-thought before its final `content`, and will return an empty/
truncated answer if `max_tokens` is too small to cover both -- so
`LLM_MAX_TOKENS` defaults to a generous 4096 rather than a typical chat-model
default. Reasoning also means real questions take noticeably longer than a
non-reasoning model: expect roughly 15-40s end to end for a question (four
sequential LLM calls: intent, SQL generation, insight, visualization plan),
confirmed against a live deployment. The UI's spinner sets this expectation
rather than looking hung.

### Ollama (local models)

Install Ollama and download at least one model, for example:

```bash
ollama pull qwen2.5:7b
```

Starting Streamlit now also starts `ollama serve` automatically when the local
API is not already running. Startup is idempotent across Streamlit reruns, and
the app never stops an Ollama process it did not launch. Set
`OLLAMA_AUTO_START=false` to manage Ollama yourself, or set
`OLLAMA_EXECUTABLE` when it is installed in a non-standard location.

The default local API URL is `http://localhost:11434`. Change
`OLLAMA_BASE_URL` in `.env` if your Ollama server is elsewhere. The Streamlit
sidebar reads Ollama's model inventory, shows every downloaded model, and has a
refresh button for models pulled while the app is open. Select **Ollama
(local)** and a model before asking a question. No Ollama Python dependency is
required; the backend uses Ollama's local HTTP API directly.

`OLLAMA_REQUEST_TIMEOUT_SECONDS` defaults to 300 because first-load and large
schema prompts can be slow on CPU-only systems; tune it for your hardware.

Set `LLM_DEFAULT_PROVIDER=ollama` and optionally `OLLAMA_MODEL=<model>` if
backend callers that do not pass an explicit selection should use Ollama.
Azure AI Foundry remains the default, preserving existing behavior.

### OpenRouter (North Mini Code)

Add an OpenRouter API key to `.env`:

```text
OPENROUTER_API_KEY=<your-openrouter-key>
OPENROUTER_MODEL=cohere/north-mini-code:free
OPENROUTER_REASONING_ENABLED=true
```

The **OpenRouter** provider and `cohere/north-mini-code:free` then appear in
the Streamlit sidebar. The backend uses OpenRouter's chat-completions endpoint,
requests strict JSON-schema responses for structured workflow steps, and sends
reasoning details back unchanged when it asks the model to repair malformed
structured output. `OPENROUTER_BASE_URL`, request timeout, optional app title,
and optional HTTP referrer can also be configured in `.env.example`.

Set `LLM_DEFAULT_PROVIDER=openrouter` to use it for backend calls that do not
provide an explicit UI selection.

### Data preview and complete CSV export

The result table renders only `UI_PREVIEW_ROWS` rows (200 by default) to keep
the Streamlit page responsive. **Download complete CSV** uses a separate copy
of the same validated, read-only SQL and is not limited by the visible preview
or the smaller LLM/chart analysis window. Downloads have their own configurable
`SQL_DOWNLOAD_MAX_ROWS` safety cap (250,000 by default); the UI clearly warns
when that cap is reached.

### Actionable errors

Errors in question input, provider configuration/authentication, service startup,
SQL validation, read-only execution, downloads, and graph generation are shown
with the actual reason and relevant next steps. Graph applicability is evaluated
from the retrieved column types: unsupported choices explain the missing data
shape, list usable chart alternatives, and suggest a follow-up query such as
adding a date, category, or second numeric measure.

## Run

```bash
streamlit run ui/streamlit_app.py
```

## Tests

```bash
pytest
```

Tests cover the SQL validator (destructive statements, multi-statement,
unauthorized tables, LIMIT enforcement), schema discovery/metadata persistence
against the real sample database, and a full offline run of the LangGraph
pipeline -- including the correction-retry loop and the give-up path -- using a
deterministic fake LLM client (`tests/fakes.py`), so the orchestration logic is
verified without any network access. `tests/test_vector_store.py` and
`tests/test_database_connect.py` exercise the RAG knowledge base and the
connect/crawl/index flow against a real (temp-directory) Chroma client and a
throwaway copy of the sample database, so they never touch the project's
real `metadata_store/`/`vector_store/` directories or leave the active
database pointed at a file that no longer exists once a test's temp
directory is cleaned up.

## Extending the schema

Add a table (or columns) to `data/ai_analyst.db` and just ask a question --
`refresh_metadata()` detects the structural change via its schema hash,
re-discovers it, asks the LLM for a business description of only what's new
(or falls back to a humanized column name if no LLM is configured), persists
the update, and re-syncs the vector knowledge base so the new/changed table is
searchable immediately. Nothing about table/column names is hardcoded
anywhere in the pipeline.

## Environment variables

See `.env.example` for the full list (Azure AI Foundry, Ollama, and OpenRouter
credentials/models, database path, metadata store directory, query
row/timeout/retry limits, logging).
