"""Cost tracking for LOCAL collector-kit runs (the production app has its own
durable ApiUsage ledger; this covers batch scripts that call platforms/*.py
directly, which previously left engagements like the Xsight baseline with no
cost record at all).

Usage in a runner script:

    from tools.collector_costs import CostTracker
    costs = CostTracker()
    ...
    r = plat.get_citations(prompt)
    costs.record(pname, plat)          # reads plat.last_usage (set on success)
    ...
    print(costs.summary())             # per-model tokens + $ at list pricing
    payload["run_cost"] = costs.as_dict()

Prices are list per-1M tokens; web-search fees per 1k searches. Perplexity
sonar-pro bills per-token plus per-request; the request fee is folded into
'search' at its per-1k rate. Override any price via env (COST_<PROVIDER>_IN /
_OUT / _SEARCH). Numbers are MEASURED tokens from the providers' own usage
blocks, priced at list: label output as estimated cost, exact tokens."""
import os
from collections import defaultdict

# (input $/1M, output $/1M, search-ish fee $/1k) by matching model substring.
_PRICES = [
    ("claude-3-haiku", (0.25, 1.25, 10.0)),
    ("claude-haiku", (1.00, 5.00, 10.0)),
    ("claude", (3.00, 15.00, 10.0)),          # sonnet-class default
    ("gpt-4o", (2.50, 10.00, 30.0)),
    ("gpt", (2.50, 10.00, 30.0)),
    ("gemini-2.5-flash", (0.30, 2.50, 35.0)),  # grounding billed per request
    ("gemini", (1.25, 10.00, 35.0)),
    ("sonar-pro", (3.00, 15.00, 6.0)),         # + per-request fee as search
    ("sonar", (1.00, 1.00, 5.0)),
    ("grok", (3.00, 15.00, 4.0)),              # via OpenRouter, :online-ish
]


def _price_for(model):
    m = (model or "").lower()
    for key, p in _PRICES:
        if key in m:
            return p
    return (3.0, 15.0, 10.0)


class CostTracker:
    def __init__(self):
        self.by_model = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0,
                                             "search": 0, "missing": 0})

    def record(self, platform_name, plat):
        """Call right after get_citations(). Reads and clears plat.last_usage."""
        u = getattr(plat, "last_usage", None)
        key = f"{platform_name}:{(u or {}).get('model', '?')}"
        d = self.by_model[key]
        d["calls"] += 1
        if u:
            d["in"] += int(u.get("in") or 0)
            d["out"] += int(u.get("out") or 0)
            d["search"] += int(u.get("search") or 0)
            plat.last_usage = None
        else:
            d["missing"] += 1     # errored call or a path that predates usage capture

    def rows(self):
        out = []
        total = 0.0
        for key in sorted(self.by_model):
            d = self.by_model[key]
            pin, pout, psearch = _price_for(key)
            cost = (d["in"] / 1e6 * pin + d["out"] / 1e6 * pout
                    + d["search"] / 1000.0 * psearch)
            total += cost
            out.append({"model": key, **d, "est_usd": round(cost, 2)})
        return out, round(total, 2)

    def as_dict(self):
        rows, total = self.rows()
        return {"total_est_usd": total, "by_model": rows,
                "note": "measured tokens from provider usage blocks, list pricing"}

    def summary(self):
        rows, total = self.rows()
        lines = [f"  {r['model']:36s} calls={r['calls']:4d} in={r['in']:>9,} "
                 f"out={r['out']:>9,} search={r['search']:3d}"
                 + (f" (no usage on {r['missing']})" if r['missing'] else "")
                 + f"  ${r['est_usd']:.2f}"
                 for r in rows]
        return "RUN COST (measured tokens, list pricing):\n" + "\n".join(lines) + \
               f"\n  TOTAL ~${total:.2f}"
