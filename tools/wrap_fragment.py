#!/usr/bin/env python3
"""Wrap an artifact-style HTML fragment (starts at <title>/<meta>, no <html>)
in a minimal, deterministic document so /admin/upload-static-report accepts
it. The wrapper is fixed text, so the served bytes are reproducible from the
source file: served = HEAD + fragment + TAIL. Prints the wrapped document."""
import sys
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n')
TAIL = '\n</body></html>\n'
frag = open(sys.argv[1], encoding='utf-8').read()
if '<html' in frag[:2000].lower():
    sys.stdout.write(frag)          # already a full document; pass through
else:
    # split head-ish tags from body: everything up to the first <div|<main|<section|<body-content
    import re
    m = re.search(r'<(div|main|section|header|nav|canvas|script(?![^>]*src)|style)\b', frag)
    cut = m.start() if m else 0
    sys.stdout.write(HEAD + frag[:cut] + '</head><body>\n' + frag[cut:] + TAIL)
