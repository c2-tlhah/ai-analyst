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
Connect & prepare database
   -> connect_database             (tool) read-only activation
   -> list_databases               (tool) SQLite catalog
   -> get_database_info            (tool) identity + file facts
   -> inspect_database_schema      (tool) tables + columns + profiles + relationships
   -> generate_descriptions        (tool + optional LLM enrichment)
   -> generate_knowledge_documents (tool) plain-text source of truth
   -> write_knowledge_documents    (tool + optional filesystem MCP/vector index)

question
   -> search_schema              (MCP + RAG) trim schema to what's relevant
   -> generate_sql              (LLM)   propose a SELECT statement
   -> validate_readonly_sql      (MCP)   security/allow-list gate
        |-- invalid --> handle_error --(retries left)--> generate_sql  [loop]
        |                             --(exhausted)---->  give_up -> END
   -> execute_readonly_sql       (MCP)   revalidated, read-only, limited, timed
        |-- failed --> handle_error --(retries left)--> generate_sql  [loop]
        |                            --(exhausted)---->  give_up -> END
   -> analyze_and_plan_results   (LLM)   insight + chart plan in one response
   -> generate_chart             (deterministic)  Plotly, from a fixed function set
   -> respond -> END
```

See `app/graph/workflow.py` for the LangGraph wiring and `app/graph/nodes.py` for
each node. Every deterministic tool invocation produces a bounded audit record
(name, stage, status, duration, safe arguments, summary, and error) that the UI
can display.

### Modules

| Concern | Module |
|---|---|
| Config / env vars | `app/config.py` |
| Logging | `app/logging_config.py` |
| Read-only DB connection | `app/db/connection.py` |
| SQL execution (limits, timeout) | `app/db/executor.py` |
| Audited database tool registry | `app/tools/database.py` |
| Database MCP server + in-memory protocol client | `app/mcp_server/database.py`, `app/mcp_client/database.py` |
| Schema discovery | `app/metadata/discovery.py` |
| Metadata persistence + change detection | `app/metadata/store.py` |
| Per-database generated semantic catalog | `app/metadata/store.py` |
| Optional neutral description enrichment | `app/metadata/enrichment.py` |
| Generic tabular-to-SQLite importer | `scripts/build_database.py` |
| Relevance-based metadata retrieval (lexical + vector/RAG) | `app/metadata/retrieval.py` |
| Versioned text documents + optional Chroma index | `app/metadata/vector_store.py` |
| Azure AI Foundry + Ollama + OpenRouter + NVIDIA NIM clients | `app/llm/client.py` |
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

`app/metadata/store.py` builds a semantic catalog from those live facts. It
uses neutral humanized descriptions without an LLM, or optional LLM-generated
descriptions constrained to the discovered schema. No packaged business seed
is loaded. Every database—including the configured startup database—uses
`metadata_store/databases/<database identity>/schema_metadata.json` and
`semantic_context.json`, so a database replaced at the same configured path
cannot inherit another schema's meaning.

A structural schema hash, metadata-format version, and cheap database/WAL
fingerprint are stored alongside the metadata. The in-process metadata cache
is periodically reverified; schema or source changes trigger bounded
re-profiling, while the LLM is called only for genuinely undocumented schema
objects. Answer-cache keys also include the live database/WAL revision, so an
updated database cannot receive a stale answer from an earlier data revision.

The database MCP server calls `app/metadata/retrieval.py` to pick which tables are relevant to the
question and returns only that slice -- expanded with relationship neighbors
and shortest-path connector tables so multi-hop joins stay possible. This keeps the
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

### Shared relative-time semantics

Before generating SQL for relative wording such as **last year**, **last
month**, **previous week**, **today**, or an explicit rolling period, the database MCP
server inspects the relevant fact/event date columns and resolves one explicit
shared range. Current databases use the ordinary previous calendar year;
historical snapshots use the calendar year preceding their latest observed
event date. The resolved inclusive-start/exclusive-end dates are supplied to
the SQL planner, displayed beside the answer, and checked again by MCP SQL
validation. Queries that calculate a separate `MAX(date)` year per table or
apply the range to only one event source are rejected and enter the bounded SQL
correction loop instead of returning a mixed-period result.

### Generic semantic correctness gates

Every generated statement is checked twice: first for read-only security, then
against capabilities discovered from the active database. The second gate
resolves every column, enforces documented join paths (including every part of
a composite key), blocks numeric/date operations on incompatible data, rejects
SQLite's arbitrary non-grouped aggregate projections, preserves event rows with
`UNION ALL` unless deduplication was requested, and verifies explicit ranking,
limit, direction, total, and average wording. Failed checks return precise
feedback to the bounded correction loop; they are never executed optimistically.

Scalar and ranking narratives are constructed directly from executed DataFrame
cells. Other LLM narratives are checked so every stated number exists in the
result or its deterministic statistics; unsupported numbers cause a local,
evidence-only summary. This separates fluent presentation from factual evidence.

## Tool-driven database preparation and documentation

The sidebar's **Database connection** panel lets you point the app at any
SQLite file (a plain path, or a `sqlite:///...` connection string) without
restarting -- `AI_ANALYST_DB_PATH` in `.env` only sets the *initial* default.
Clicking **Connect & prepare database** (`app.orchestrator.connect_database`)
runs a fixed, bounded workflow from `app/tools/database.py`:

1. Validates the file opens as a real, read-only SQLite database.
2. Makes it the active database for every subsequent query (backed by
   `app.db.connection.set_active_database_path`) and drops the session's
   metadata/answer caches, since they described a different database.
3. Lists attached databases and user tables, then inspects every table's
   schema, bounded column profile, declared/inferred unique keys, and relationships.
4. Builds the database-scoped semantic catalog using deterministic discovery +
   optional neutral LLM enrichment (`app/metadata/discovery.py`,
   `app/metadata/enrichment.py`); only genuinely new tables/columns cost a call.
5. Renders each table's business-enriched metadata to a versioned plain-text
   document. These files are the knowledge base's source of truth and remain
   usable without embeddings.
6. Optionally verifies those exact backend-selected files through the official
   filesystem MCP server when `MCP_FILESYSTEM_ALLOW_MUTATIONS=true`. Paths must
   remain inside both the managed version directory and a configured MCP root.
7. Optionally upserts the same documents into a Chroma collection scoped to
   the active database for semantic similarity search.

The LLM cannot skip connection validation, choose arbitrary documentation
paths, access unapproved tables, or bypass SQL policy. The sidebar shows every
preparation tool and its outcome.

### Unseen database behavior and scope

The agent does not require AdventureWorks table names. For any readable SQLite
database it discovers ordinary tables and views (including view dependencies), arbitrary quoted identifier names,
SQLite declared-type families (`BIGINT`, `DECIMAL(...)`, `DATETIME`, `BOOLEAN`,
`VARCHAR(...)`, and others), keys, categorical samples, measures, time fields,
flags, and aggregation hints. Declared foreign keys are preferred. When an
imported database omitted them, unique conventional matches such as
`orders.customer_id -> customers.customer_id` or matching UNIQUE business keys are conservatively inferred only
when declared types are compatible and sampled source keys have strong overlap
with a unique target key. They are labelled `inferred` with confidence in
metadata and prompts; ambiguous, composite, orphaned, and view-to-base guesses
are left unresolved.

There is no packaged-schema SQL template or runtime business-context seed.
Sensitive-looking fields retain structural metadata but never persist sample
values into prompts or knowledge documents. Every database uses its own identity-isolated generated context, deterministic
profiles, optional neutral LLM descriptions, and example questions generated
from its own tables and columns. Wide-table prompt context, profile scans, row
counts, result previews, and downloads are independently bounded. Switching
databases clears visible result/chart history as well as backend caches.

This connector currently targets SQLite and generates SQLite SQL. Supporting
PostgreSQL, SQL Server, MySQL, or cloud warehouses requires a separate dialect
adapter; pointing this SQLite connector at those engines is intentionally not
attempted.

Embeddings run entirely on-machine via Chroma 1.5+'s bundled ONNX MiniLM
model -- no API key, no network call per query, and no LLM token cost, so
RAG retrieval is strictly additive to the app's token budget. The first
index build may download the local embedding model once. Set
`VECTOR_RAG_ENABLED=false` in `.env` to skip Chroma entirely. Generated files
and documentation Q&A continue working through lexical document search (see
`.env.example` for `VECTOR_STORE_DIR`/`VECTOR_TOP_K`).

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

The sidebar distinguishes **semantically indexed**, **documents ready with
lexical search**, **not built**, and **semantic index failed with lexical
fallback** (including the backend error and recovery steps).
Its searchable **Knowledge base explorer** lets you select any version,
inspect scrollable per-table documents, and download the complete version as
plain text. The same indexed document is also available beside each table in
**Available data**, with a per-table download
(`app.orchestrator.list_knowledge_base_versions` /
`get_knowledge_base_documents`). Building deterministic schema documents does
not require a configured LLM; an available LLM only improves generated
descriptions.

### Ask generated documentation

The main page has separate **Query data** and **Ask knowledge base** tabs.
The knowledge tab is for questions such as “What does revenue mean?”, “How
are internet and reseller sales different?”, or “How should SalesAmount be
aggregated?”. The backend calls the MCP `search_knowledge_documents` tool, which uses a
healthy semantic index when available and otherwise ranks the generated text
files lexically. It gives up to four relevant documents to the selected Azure
AI Foundry, Ollama, OpenRouter, or NVIDIA NIM model. Answers are instructed to
cite their table sources, and the UI exposes the complete retrieved documents
and the search-tool audit record below every answer for verification.

Document RAG never executes SQL and does not invent live totals. Questions
that require actual calculations continue through **Query data**, where MCP
performs schema RAG, SQL policy validation, and read-only execution around the
LLM SQL planner. The in-memory MCP transport retains a real client/server
protocol boundary without starting a subprocess for every graph node. Successful
document answers are cached by database, knowledge-base
version, provider/model, and normalized question, so rebuilding the index
cannot return a stale answer from an earlier schema version.

### OpenRouter model catalog

The OpenRouter selector includes North Mini Code plus the curated free Liquid,
NVIDIA Nemotron, inclusionAI Ling, Poolside Laguna, and Google Gemma models
configured in `app/llm/client.py`. When OpenRouter is selected, the app refreshes
those entries from the public `/api/v1/models` catalog and displays each model's
current context window, modalities, pricing, reasoning/structured-output support,
supported request parameters, and the generation values this app will send. A
**Refresh OpenRouter models** button updates this metadata without restarting
Streamlit. If discovery is temporarily unavailable, model selection continues
using the built-in IDs and the UI clearly marks the metadata as unavailable.

Free-model availability and capabilities are provider-controlled and may change.
The existing `.env` default and any custom `OPENROUTER_MODEL` value remain
selectable, preserving current deployments.

Transient OpenRouter timeouts, HTTP 429 responses, and common HTTP 5xx failures
are retried with a short bounded backoff (`OPENROUTER_MAX_RETRIES` and
`OPENROUTER_RETRY_BACKOFF_SECONDS`). If retries are exhausted, the UI preserves
the HTTP status, selected model, upstream provider metadata, and nested provider
message instead of reducing the failure to “Provider returned error.” A provider
outage during intent classification also stops the workflow immediately instead
of spending another request on SQL generation that would fail for the same
reason.

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
  instantly, skipping the DB round-trip and both normal LLM calls. Only
  successful answers are cached -- a failed attempt is always retried for
  real. The UI surfaces this as a "🔁 served from session cache" badge.
* **Bounded conversation memory.** The last few `{question, sql}` turns
  from the session are threaded into the SQL-generation prompt
  (`app/graph/nodes.py::_format_history`) so short
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

Each successful answer shows the retrieved table and an on-demand complete CSV
export beside its generated SQL. The data grid and SQL viewer use matching fixed-height,
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

## Optional demo data and generic imports

`data/raw/*.csv` is a trimmed extract of the classic AdventureWorks DW sample
data and `data/ai_analyst.db` is only an optional runnable demo. Neither supplies
runtime metadata or special SQL behavior.

`scripts/build_database.py` is a generic importer for CSV, TSV, JSON, JSONL, or
Parquet directories. It preserves arbitrary table/column identifiers, infers
conservative SQLite types and keys, and accepts an optional JSON manifest for
exact renames, required fields, types, primary keys, and foreign keys:

```bash
python scripts/build_database.py --input C:/exports --output data/local.db
python scripts/build_database.py --input C:/exports --manifest schema.json --force
```

You do not need this importer for an existing SQLite database—connect its path
directly in the sidebar.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with Azure, Ollama, OpenRouter, and/or NVIDIA NIM settings (see below)

# Optional: import any tabular directory into SQLite
python scripts/build_database.py --input /path/to/exports --output data/local.db
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

Structured outputs (SQL and combined insight/viz plan) are enforced via
JSON-schema prompting + Pydantic validation with one bounded repair retry
(`app/llm/client.py`), rather than relying on provider-specific function-calling
support -- this keeps the client portable across whatever Foundry model you
point it at.

**Kimi K2.6 is a reasoning model**: it emits a hidden `reasoning_content`
chain-of-thought before its final `content`, and will return an empty/
truncated answer if `max_tokens` is too small to cover both -- so
`LLM_MAX_TOKENS` defaults to a generous 4096 rather than a typical chat-model
default. Reasoning also means real questions take noticeably longer than a
non-reasoning model. A normal question now makes two sequential calls (SQL,
then combined insight/visualization), halving the former four-call path. The
UI's spinner sets this expectation rather than looking hung.

The configured Azure `Kimi-K2.6` deployment also supports native function
calling. The filesystem assistant uses Azure AI Inference's `tools` and
`tool_choice` fields, preserves assistant tool-call IDs, executes approved MCP
tools, and returns each result as a tool message until Kimi produces its final
answer.

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

### OpenRouter

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

### NVIDIA NIM cloud models

Add an NVIDIA API Catalog key to `.env`:

```text
NVIDIA_API_KEY=<your-nvidia-key>
NVIDIA_NIM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

Select **NVIDIA NIM (cloud)** in the sidebar. The configured Nemotron model is
always shown together with `poolside/laguna-xs-2.1`, `z-ai/glm-5.2`, and
`minimaxai/minimax-m3`; after authentication,
**Refresh NVIDIA models** verifies only these explicitly approved model IDs
against the OpenAI-compatible `/v1/models` endpoint. Unrequested catalog models
are not added to the selector. The chat client sends NVIDIA's thinking controls
(`chat_template_kwargs.enable_thinking` and `reasoning_budget`) together with
the configured temperature, top-p, and output-token budget.

Laguna XS 2.1 follows its separate NVIDIA request profile: `max_tokens` is
8192, temperature is 1, top-p is 0.95, and Nemotron-specific reasoning fields
are omitted. GLM 5.2 uses temperature 1, top-p 1, max-tokens 16384, and seed 42.
MiniMax M3 uses temperature 1, top-p 0.95, and max-tokens 8192. All four
approved models advertise native tool use. Tool definitions are sent through
the same OpenAI-compatible `tools` interface when the filesystem assistant is
used.

Live endpoint tests confirmed that GLM 5.2 and MiniMax M3 support strict native
`json_schema` responses. The analytics workflow sends each Pydantic schema as
an OpenAI-compatible response format for those two models, then still parses
and validates the returned object locally. Nemotron and Laguna retain the
portable prompt + local validation + bounded repair path.

Although NVIDIA's example streams tokens, the analytics backend sets
`stream=false`: SQL and every other structured workflow result must be complete,
parsed, and validated before the application can use it safely. The free-form
answer remains identical; only incremental rendering is disabled. All relevant
request values are visible under **Model parameters & capabilities** in the UI.
See `.env.example` for timeout, discovery, retry, and reasoning settings. Set
`LLM_DEFAULT_PROVIDER=nvidia_nim` to make NVIDIA the backend default.

The NVIDIA client retries transient DNS, connection, timeout, rate-limit, and
5xx failures with bounded exponential backoff. Model discovery, chat calls, and
retries share one process-wide rolling budget of 60 requests per minute.
Requests are spaced slightly over one second apart, so concurrent Streamlit
sessions queue behind the same gate. An HTTP 429 honors `Retry-After`, or starts
the configured shared 60-second cooldown when that header is absent. Use
**Test connection** in the sidebar to verify local DNS resolution, authenticated
catalog access, and the selected model without displaying the API key. If Windows reports
`[Errno 11001] getaddrinfo failed`, run:

```powershell
Resolve-DnsName integrate.api.nvidia.com
ipconfig /flushdns
```

If the first command repeatedly times out, the configured router/VPN DNS or a
network policy is failing before NVIDIA can receive the request. Reconnect the
network/VPN, use a reliable DNS resolver approved for the machine, or ask the
network administrator to allow `integrate.api.nvidia.com`. Do not hardcode the
endpoint IP: NVIDIA uses multiple TLS/CDN addresses that can change.

The standard data-query pipeline now uses two provider requests: one generates
SQL and one combined request produces both the business insight and recommended
chart plan. Schema relevance is retrieved deterministically, so it no longer
spends a separate request on intent classification. SQL correction consumes an
additional request only when validation or execution finds a real issue.
Database onboarding inspects all schemas, profiles, keys, and relationships in
one read-only tool call, and descriptions of new tables are batched (12 tables
per LLM request by default). Configure the budget through the
`NVIDIA_NIM_REQUESTS_PER_MINUTE`, `NVIDIA_NIM_MIN_REQUEST_INTERVAL_SECONDS`,
`NVIDIA_NIM_RATE_LIMIT_MAX_WAIT_SECONDS`, and
`NVIDIA_NIM_429_COOLDOWN_SECONDS` settings. Configure onboarding batches with
`METADATA_LLM_ENRICH_BATCH_SIZE`.

### Filesystem MCP assistant

The **Work with files (MCP)** tab connects the selected tool-capable model to
the official `@modelcontextprotocol/server-filesystem` server. Install both
dependency sets once:

```powershell
pip install -r requirements.txt
cmd /c npm install
```

The database-preparation workflow can also send its generated documentation
through this MCP server when `MCP_FILESYSTEM_ALLOW_MUTATIONS=true`. This path
does not accept model-generated filenames: it permits only the exact files in
the backend-managed knowledge-version directory and falls back to the local
managed writer if MCP is unavailable. Interactive file changes still require
the separate per-request UI approval.

By default, the server can access only the project directory and only read/list/
search/metadata tools are exposed. Configure one or more roots with
`MCP_FILESYSTEM_ROOTS` (semicolon-separated on Windows). The backend validates
every model-generated path itself in addition to the MCP server's root checks.

Create, write, edit, and move tools require two independent approvals:

1. Set `MCP_FILESYSTEM_ALLOW_MUTATIONS=true` and restart Streamlit.
2. Enable and confirm mutations for the individual request in the UI.

Delete tools and arbitrary MCP/npm packages are never exposed. File contents
are treated as untrusted data, tool rounds and returned content are bounded,
and the UI shows every MCP call, its arguments, result, and error status.

### Data preview and complete CSV export

The result table renders only `UI_PREVIEW_ROWS` rows (200 by default) to keep
the Streamlit page responsive. The selected LLM receives only five raw result
rows plus deterministic local statistics, regardless of whether SQLite fetched
five rows or five thousand. **Prepare complete CSV** runs a separate copy of
the same validated, read-only SQL only when clicked; it never delays the initial
answer and is not limited by the visible preview or the smaller LLM/chart
analysis window. Downloads have their own configurable
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

## Connecting or changing a schema

Connect any populated SQLite file in the sidebar, or add/change tables in the
active database, then refresh the schema. `refresh_metadata()` detects the structural change via its schema hash,
re-discovers it, asks the LLM for a business description of only what's new
(or falls back to a humanized column name if no LLM is configured), persists
the update, refreshes the versioned knowledge documents, and optionally
re-syncs their vector index so the new/changed table is searchable immediately.
Nothing about table/column names is hardcoded
anywhere in the pipeline.

## Environment variables

See `.env.example` for the full list (Azure AI Foundry, Ollama, OpenRouter,
and NVIDIA NIM credentials/models, database path, metadata store directory, query
row/timeout/retry limits, logging).

### Logs and agent traces

The application writes two rotating, secret-redacted diagnostics streams:

- `logs/ai_analyst.log` is the readable backend log, including trace IDs.
- `logs/agent_traces.jsonl` contains structured events for complete requests,
  LLM/provider attempts, reasoning stages, database tools, SQL validation and
  execution, retries, knowledge retrieval, and filesystem MCP tools.

Open **Live agent logs** in Streamlit to filter one request by trace ID, watch
new events refresh every two seconds, inspect the latest structured event, and
download redacted traces or the application-log tail. Rotation sizes and backup
counts are configured through `LOG_*` and `AGENT_TRACE_*` settings in
`.env.example`. API keys, authorization headers, passwords, secrets, and tokens
are redacted before events are retained in memory or written to disk. NVIDIA
`request_budget` events report queue waits, calls used in the rolling window,
remaining local permits, cooldowns, retries, and safe rate-limit headers.
