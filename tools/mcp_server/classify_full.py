#!/usr/bin/env python3
"""Full source classification for one audit payload: the standard delivery
step (Nathan, 2026-10-10) that makes "every citation classified" true.

The live run types only the most-cited roots (_domain_types_for_report keeps
the top 180), so a big run leaves a tail with no type. This pass gives EVERY
distinct cited root a type, in this order:
  1. owned: roots of the payload's brand_domains (no model call);
  2. the payload's own domain_types, where present (same classifier, kept);
  3. one model pass over everything left, with the app's own classifier
     (_classify_domains_llm: same prompt, same types, observed domains only).
Then it lists EXCEPTIONS for an analyst to review before delivery:
  - the model returned no valid type
  - the root matches the brand's name (or an alias) but isn't in brand_domains
  - a named competitor's own site typed as something else
  - link plumbing (search and redirect hosts), which is never a source
  - the computed root looks like a bare public suffix
The analyst decides every exception, either with --decide on the command
line or a "decision" key in the JSON: accept (take the proposed type), keep
(keep the current type) or a type name. --check applies the decisions and
fails unless every exception is decided and every root carries a valid type;
it then stamps _meta.status = "reviewed".

Output: {root: type, ..., "_meta": {...}, "_exceptions": [...]} plus
<out>_exceptions.csv. export_raw_run.py takes the JSON as its host-classes
file and reports coverage in its manifest.

Usage:
    python3 classify_full.py <payload.json> <out_classes.json>
    python3 classify_full.py <payload.json> <out_classes.json> --check \
        [--decide nyulangone.org=accept --decide t.co=keep --decide x.com=social ...]
Spend: one Haiku call per 60 unclassified roots (cents per audit).
"""
import argparse, csv, datetime, hashlib, io, json, os, re, sys, contextlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conventions import root_of, is_bare_suffix   # noqa: E402

PLUMBING = {"google.com", "bing.com", "duckduckgo.com", "t.co", "bit.ly", "lnkd.in",
            "goo.gl", "ow.ly", "tinyurl.com", "googleusercontent.com", "gstatic.com",
            "googleapis.com"}


def _app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root
    os.environ.setdefault("FLASK_SECRET_KEY", "offline")
    with contextlib.redirect_stdout(io.StringIO()):               # DB-migration noise
        import app as A   # noqa: E402  (load_dotenv(override=True) supplies the real key)
    return A


def _own_site(A, root, name):
    """Stricter than the app's _is_brand_own_domain, which is tuned for
    single-word brands and fires on short or generic labels (ny.gov for NYU
    Langone, cancer.gov for Memorial Sloan Kettering, health.nz for Northwell
    Health). Matches the whole name, a long part of it, or an affixed
    distinctive token; a bare token only for one-token names (Azzaro, Kilian)."""
    label = root.split(".")[0]
    slug = re.sub(r"[^a-z0-9]", "", A._ascii_fold((name or "").lower()))
    if len(slug) < 3 or len(label) < 3:
        return False
    if label == slug or slug in label:
        return True
    if label in slug and len(label) >= max(6, 0.6 * len(slug)):
        return True
    toks = A._brand_distinctive_tokens(name)
    for tok in toks:
        if label == tok and len(toks) == 1:
            return True
        if any(label == pre + tok for pre in A._OWN_DOMAIN_PREFIXES) or \
           any(label == tok + suf for suf in A._OWN_DOMAIN_SUFFIXES):
            return True
    return False


def _universe(p):
    counts = Counter()
    for r in p.get("all_responses") or []:
        for c in r.get("citations") or []:
            u = c.get("url") if isinstance(c, dict) else None
            root = root_of(u) if u else ""
            if root and "." in root:
                counts[root] += 1
    return counts


def classify(payload_path, out_path):
    raw = open(payload_path, "rb").read()
    p = json.loads(raw)
    A = _app()
    valid = set(A._DOMAIN_TYPES_VALID) | {"owned"}
    brand = (p.get("brand") or "").strip()
    aliases = [a for a in (p.get("brand_aliases") or []) if isinstance(a, str) and a.strip()]
    comps = [c.get("name") for c in (p.get("competitors") or []) if isinstance(c, dict) and c.get("name")]
    counts = _universe(p)
    owned_roots = {root_of("https://" + d) for d in (p.get("brand_domains") or []) if d}
    payload_types = dict(p.get("domain_types") or {})

    out, exc, source = {}, {}, Counter()

    def flag(root, proposed, reason):
        e = exc.setdefault(root, {"root": root, "citations": counts.get(root, 0),
                                  "proposed": proposed or "", "reason": []})
        if proposed and not e["proposed"]:
            e["proposed"] = proposed
        e["reason"].append(reason)

    todo = []
    for root in sorted(counts, key=lambda r: (-counts[r], r)):
        if root in owned_roots:
            out[root] = "owned"; source["owned"] += 1
            continue
        t = payload_types.get(root)
        if t in valid:
            out[root] = t; source["payload"] += 1
        else:
            todo.append(root)
    if todo:
        got = A._classify_domains_llm(todo, brand, p.get("category") or "", comps)
        for root in todo:
            t = got.get(root)
            if t in valid:
                out[root] = t; source["model"] += 1
            else:
                flag(root, "", "model returned no valid type")

    for root in sorted(counts, key=lambda r: (-counts[r], r)):
        t = out.get(root)
        if root not in owned_roots and any(_own_site(A, root, n) for n in [brand] + aliases if n):
            flag(root, "owned", f"matches the brand's name but isn't in brand_domains (typed {t or 'none'}); confirm owned")
        named_comp = [n for n in comps if _own_site(A, root, n)]
        if named_comp and t not in ("competitor", None):
            flag(root, "competitor", f"looks like {named_comp[0]}'s own site but typed {t}; confirm")
        if root in PLUMBING:
            flag(root, t, "search or redirect host, usually link plumbing rather than a source; confirm")
        if is_bare_suffix(root):
            flag(root, t, "computed root looks like a bare public suffix; unrelated sites may be merged")

    exc = [dict(e, reason="; ".join(e["reason"])) for e in exc.values()]
    meta = {"payload": os.path.basename(payload_path),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "slug": p.get("slug"), "brand": brand,
            "roots": len(counts), "classified": sum(1 for r in counts if out.get(r) in valid),
            "from_owned": source["owned"], "from_payload": source["payload"],
            "from_model": source["model"], "exceptions": len(exc),
            "status": "needs_review" if exc else "complete",
            "created": datetime.date.today().isoformat(),
            "types": sorted(valid)}
    doc = dict(out)
    doc["_meta"], doc["_exceptions"] = meta, exc
    json.dump(doc, open(out_path, "w"), indent=1)
    csv_path = os.path.splitext(out_path)[0] + "_exceptions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["root", "citations", "proposed", "reason"])
        w.writeheader(); w.writerows(exc)
    print(f"{meta['slug']} | {meta['roots']} distinct cited roots | classified {meta['classified']} "
          f"(owned {source['owned']}, payload {source['payload']}, model {source['model']}) | "
          f"exceptions {len(exc)} -> {os.path.basename(csv_path)} | status {meta['status']}")
    return meta


def check(payload_path, out_path, decisions=None):
    raw = open(payload_path, "rb").read()
    p = json.loads(raw)
    doc = json.load(open(out_path))
    meta = doc.get("_meta") or {}
    valid = set(meta.get("types") or [])
    problems = []
    if meta.get("payload_sha256") != hashlib.sha256(raw).hexdigest():
        problems.append("classes file was built from a different payload (sha mismatch)")
    decide = dict(x.split("=", 1) for x in (decisions or []) if "=" in x)
    unresolved = []
    for e in doc.get("_exceptions") or []:
        d = decide.get(e["root"], e.get("decision"))
        if not d:
            unresolved.append(e["root"])
            continue
        e["decision"] = d
        if d == "accept":
            if e.get("proposed") in valid:
                doc[e["root"]] = e["proposed"]
            else:
                problems.append(f"{e['root']}: 'accept' needs a proposed type; give a type name instead")
        elif d in valid:
            doc[e["root"]] = d
        elif d != "keep":
            problems.append(f"{e['root']}: unknown decision '{d}' (accept, keep or one of {sorted(valid)})")
    if unresolved:
        problems.append(f"{len(unresolved)} exception(s) without a decision: {unresolved[:12]}")
    counts = _universe(p)
    missing = [r for r in counts if doc.get(r) not in valid]
    if missing:
        problems.append(f"{len(missing)} root(s) without a valid type: {missing[:12]}")
    if problems:
        print("CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    meta.update(status="reviewed", reviewed=datetime.date.today().isoformat(),
                classified=len(counts), roots=len(counts))
    doc["_meta"] = meta
    json.dump(doc, open(out_path, "w"), indent=1)
    print(f"CHECK PASSED: all {len(counts)} distinct cited roots typed; status reviewed")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("payload"); ap.add_argument("out")
    ap.add_argument("--check", action="store_true", help="apply decisions and validate a reviewed classes file")
    ap.add_argument("--decide", action="append", metavar="ROOT=accept|keep|TYPE",
                    help="decision for one exception (repeatable); also read from the JSON")
    ap.add_argument("--force", action="store_true", help="overwrite a file already marked reviewed")
    a = ap.parse_args()
    if a.check:
        check(a.payload, a.out, a.decide)
    else:
        if os.path.exists(a.out) and not a.force:
            if (json.load(open(a.out)).get("_meta") or {}).get("status") == "reviewed":
                sys.exit(f"{a.out} is already reviewed; pass --force to rebuild it from scratch")
        classify(a.payload, a.out)
