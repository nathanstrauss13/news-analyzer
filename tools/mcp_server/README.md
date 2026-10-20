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

query_citations filters (2026-10-17): domain, agent, prompt_contains,
prompt_class (branded | unbranded), brand_in_answer (named | named_prose |
named_first | absent — the compare_runs convention: prose = answer text with
URLs stripped; named_first = brand's first prose mention precedes every
competitor's, competitors from the audit's own list, exact names only),
source_type (classified type of the cited root; 'owned' = the audit's
brand_domains), url_contains. group_by='domain' returns a ranked table over
the full matching set (citations and distinct answers per root) instead of a
row slice, so "most cited" questions are never capped by `limit`.

get_page_evidence filters: only_missing_brand, url_contains, title_contains,
source_type; sort='times_cited'; offset + limit page through the full set
(next_offset in the response). Each page row carries times_cited and
answers_citing from the run's own answers.

Worked example (a client's four questions, all answerable in one call each):
most-cited outlets where the brand was named first in an unbranded answer →
query_citations(slug, prompt_class="unbranded", brand_in_answer="named_first",
group_by="domain"); where the brand was absent → brand_in_answer="absent";
cited pages with the brand in the URL or title → get_page_evidence(slug,
url_contains="<brand>", sort="times_cited") and title_contains; own site vs
competitors' sites in unbranded answers → source_type="owned" and
source_type="competitor" with prompt_class="unbranded", group_by="domain".

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
    python3 export_raw_run.py    <payload.json> <out_dir> <classes.json> --merge-repeats
    python3 build_data_workbook.py <payload.json> <classes.json> <out_dir> <Client_Data_Completeness.xlsx> --client <who> --run-date "<date>"

Before any of the above: scan all_responses for empty or degenerate answers
(the collector stores an error row for them since 2026-10-20, but older
payloads can hold empty strings or runaway repetitions). Re-collect those
answers through the product call path, swap them in with a ledger
(`rerun_ledger`), then run the steps. A payload with an empty answer is not
client-ready.

build_data_workbook.py (standard since 2026-10-20, every run, internal and
client-facing): the data completeness workbook, three tabs (Read me, Answers
one row per prompt x platform with retrieval evidence and a collapsible full
response, Citations one row per kept citation with delivery channel, demasking,
duplicate handling, page scrape and source type). Styled like the project
trackers; opens in Google Sheets with filters and coloured dropdowns intact.
Inject the full page checks into the payload as `citation_checks` first (or
pass --checks). Every figure is recomputed from all_responses; nothing is
copied from the platform summary fields.

classify_full.py (standard since 2026-10-10): owned roots from brand_domains,
the payload's own domain_types, then one Haiku pass over the rest with the
app's classifier. --check fails until every exception carries a decision and
every root has a type. The exporter's manifest states the coverage
("classified roots: N of N | full classification pass: reviewed").

Within-answer repeats: from 2026-10-10 the pipeline counts an identical final
URL once per answer and stamps the payload with `citation_counting`. Older
payloads export as stored; the manifest counts the repeats they still hold,
and --merge-repeats merges them to compare an older run with a newer one.
