#!/usr/bin/env python3
"""Approved prompt rows (intake workbook) -> the run's prompt_sets input.

Reads the 'Prompt set' tab, keeps only rows with Approved = TRUE for one
brand, validates the counts against the operator limits (3-60 per half,
5-120 total), and writes the JSON the operator run posts as `prompt_sets`.
Questions are never retyped: what the client approved is what runs.

Usage:
  python3 import_prompt_set.py <workbook.xlsx> --brand "Azzaro" [--out prompt_sets_azzaro.json]
  python3 import_prompt_set.py <workbook.xlsx> --brand "Azzaro" --post "<problem statement>"
      (posts to signal.innatec3.com/citation-audit from an operator IP / ops+ email;
       prints the slug; costs a real run)
"""
import argparse, json, sys, re
from openpyxl import load_workbook

LIMITS = {"half_min": 3, "half_max": 60, "total_min": 5, "total_max": 120}


def read_rows(path, brand):
    wb = load_workbook(path, data_only=True)
    ws = wb["Prompt set"]
    hdr = None
    rows = []
    for r in ws.iter_rows(values_only=True):
        if hdr is None:
            if r and r[0] == "#":
                hdr = [str(c).strip() if c is not None else "" for c in r]
            continue
        rec = dict(zip(hdr, r))
        if not rec.get("Question") or str(rec.get("Brand") or "").strip().lower() != brand.lower():
            continue
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook"); ap.add_argument("--brand", required=True)
    ap.add_argument("--out"); ap.add_argument("--post", help="problem statement; posts the run")
    ap.add_argument("--email", default="ops+intake@innatec3.com")
    ap.add_argument("--include-unapproved", action="store_true", help="dry-run shape check only")
    a = ap.parse_args()

    rows = read_rows(a.workbook, a.brand)
    appr = [r for r in rows if a.include_unapproved or str(r.get("Approved")).strip().upper() == "TRUE"]
    b = [str(r["Question"]).strip() for r in appr if str(r.get("Type") or "").strip().lower() == "branded"]
    u = [str(r["Question"]).strip() for r in appr if str(r.get("Type") or "").strip().lower() == "unbranded"]
    untyped = [r["#"] for r in appr if str(r.get("Type") or "").strip().lower() not in ("branded", "unbranded")]
    dups = {q for q in b + u if (b + u).count(q) > 1}
    leaks = [q for q in u if re.search(re.escape(a.brand), q, re.I)]
    print(f"{a.brand}: {len(rows)} rows, {len(appr)} approved -> branded {len(b)}, unbranded {len(u)}")
    problems = []
    if untyped:
        problems.append(f"rows without a Type: {untyped}")
    if dups:
        problems.append(f"duplicate questions: {sorted(dups)[:5]}")
    if leaks:
        problems.append(f"unbranded questions that name the brand: {leaks[:5]}")
    for name, n in (("branded", len(b)), ("unbranded", len(u))):
        if not (LIMITS["half_min"] <= n <= LIMITS["half_max"]):
            problems.append(f"{name} count {n} outside {LIMITS['half_min']}-{LIMITS['half_max']}")
    if not (LIMITS["total_min"] <= len(b) + len(u) <= LIMITS["total_max"]):
        problems.append(f"total {len(b) + len(u)} outside {LIMITS['total_min']}-{LIMITS['total_max']}")
    for p in problems:
        print("  PROBLEM:", p)
    sets = {"branded": b, "unbranded": u}
    out = a.out or f"prompt_sets_{re.sub(r'[^a-z0-9]+', '_', a.brand.lower())}.json"
    json.dump(sets, open(out, "w"), indent=1)
    print("wrote", out)
    if problems:
        sys.exit("fix the problems above before posting")
    if a.post:
        import requests
        r = requests.post("https://signal.innatec3.com/citation-audit",
                          data={"problem_statement": a.post, "email": a.email, "prompt_sets": json.dumps(sets)},
                          stream=True, timeout=3600)
        slug = None
        for line in r.iter_lines():
            s = line.decode() if line else ""
            m = re.search(r'"slug":\s*"([0-9a-f]{10})"', s)
            if m:
                slug = m.group(1)
            if '"progress"' in s and "Completed" in s:
                mm = re.search(r"Completed (\d+)/(\d+)", s)
                if mm and int(mm.group(1)) % 25 == 0:
                    print(" ", mm.group(0), flush=True)
        print("SLUG:", slug)


if __name__ == "__main__":
    main()
