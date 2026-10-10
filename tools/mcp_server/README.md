# Signal MCP server

A client-side adapter that plugs innate c3 audit data into the client's own
AI assistant (Claude, ChatGPT desktop, Cursor — anything MCP-capable).

## Design contract
Facts, never conclusions (whitelisted fields; narrative payload fields are
never returned). Every response carries convention metadata (scope, counting
method, denominators). Per-client slug ALLOWLIST — no wildcards; a client's
config reaches their audits only. Read-only: request_rerun files a request,
nothing collects data. compare_runs recomputes both runs from raw answers
under one stated convention at query time — stored aggregates are
era-dependent and never diffed.

## Tools
list_audits · get_audit · query_citations · outlet_profile · get_responses
(verbatim, labeled as agent output) · get_page_evidence · compare_runs ·
request_rerun

## Install (client)
    python3 -m venv venv && venv/bin/pip install mcp certifi
    # config at ~/.signal_mcp/config.json (or SIGNAL_MCP_CONFIG):
    # {"base_url": "https://signal.innatec3.com", "client": "<name>",
    #  "audits": [{"slug": "…", "label": "…", "run_date": "YYYY-MM-DD"}]}
Then register in the assistant's MCP settings:
    command: <path>/venv/bin/python   args: [<path>/signal_mcp.py]

Payloads are cached at ~/.signal_mcp/cache (TTL 24h) so steady-state load on
the hosting instance is a handful of conditional fetches per day.

## Delivery QA steps (every delivered audit)
Run in this order on the audit's payload JSON, then build the bundle:

    python3 page_check_full.py   <payload.json> <checks_full.json>     # every distinct cited page fetched, client's name checked
    python3 classify_full.py     <payload.json> <classes.json>         # every distinct cited root typed; exceptions listed
    python3 classify_full.py     <payload.json> <classes.json> --check --decide <root>=accept|keep|<type> ...
    python3 export_raw_run.py    <payload.json> <out_dir> <classes.json>

classify_full.py (standard since 2026-10-10): owned roots from brand_domains,
the payload's own domain_types, then one Haiku pass over the rest with the
app's classifier. --check fails until every exception carries a decision and
every root has a type. The exporter's manifest states the coverage
("classified roots: N of N | full classification pass: reviewed").

Within-answer repeats: from 2026-10-10 the pipeline counts an identical final
URL once per answer and stamps the payload with `citation_counting`. Older
payloads export as stored; the manifest counts the repeats they still hold,
and --merge-repeats merges them to compare an older run with a newer one.
