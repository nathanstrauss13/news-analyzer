#!/usr/bin/env python3
"""Build a one-click .mcpb bundle for a hosted Signal MCP key.

Usage:
    python3 build_mcpb.py <key> [--name "Signal — Innate c³"] [--out dist/]

Emits dist/<name-slug>.mcpb — a zip Claude Desktop installs with one click.
The bundle contains a dependency-free Node stdio<->HTTP bridge pointed at
https://signal.innatec3.com/mcp/<key>. The key is baked into the file:
a bundle is a bearer credential, so per-client bundles are delivered
person-to-person, never posted or broadly emailed.
"""
import argparse, json, pathlib, re, sys, zipfile

BASE_URL = "https://signal.innatec3.com/mcp/"

BRIDGE_JS = r"""#!/usr/bin/env node
// Minimal stdio <-> HTTP JSON-RPC bridge for the Signal MCP hosted endpoint.
// No dependencies. Reads newline-delimited JSON-RPC on stdin, POSTs each
// message to the endpoint, writes responses (when any) to stdout.
const https = require("https");
const { URL } = require("url");
const ENDPOINT = new URL(process.env.SIGNAL_MCP_URL || "%ENDPOINT%");

let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (line) forward(line);
  }
});

function forward(line) {
  let msg;
  try { msg = JSON.parse(line); } catch (e) { return; }
  const body = Buffer.from(JSON.stringify(msg), "utf8");
  const req = https.request(ENDPOINT, {
    method: "POST",
    headers: { "content-type": "application/json", "content-length": body.length },
  }, (res) => {
    const parts = [];
    res.on("data", (d) => parts.push(d));
    res.on("end", () => {
      if (res.statusCode === 202) return; // notification accepted, no reply
      const text = Buffer.concat(parts).toString("utf8");
      if (!text.trim()) return;
      try { process.stdout.write(JSON.stringify(JSON.parse(text)) + "\n"); }
      catch (e) { fail(msg, -32603, "bad response from server"); }
    });
  });
  req.on("error", (e) => fail(msg, -32001, "connection failed: " + e.message));
  req.end(body);
}

function fail(msg, code, message) {
  if (msg && msg.id !== undefined && msg.id !== null) {
    process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: msg.id, error: { code, message } }) + "\n");
  }
}
"""


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "signal"


def build(key: str, display_name: str, out_dir: pathlib.Path, description: str) -> pathlib.Path:
    endpoint = BASE_URL + key
    manifest = {
        "manifest_version": "0.2",
        "name": slugify(display_name),
        "display_name": display_name,
        "version": "1.0.0",
        "description": description,
        "author": {"name": "Innate c³", "url": "https://www.innatec3.com"},
        "server": {
            "type": "node",
            "entry_point": "server/index.js",
            "mcp_config": {
                "command": "node",
                "args": ["${__dirname}/server/index.js"],
                "env": {"SIGNAL_MCP_URL": endpoint},
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(display_name)}.mcpb"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("server/index.js", BRIDGE_JS.replace("%ENDPOINT%", endpoint))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--name", default="Signal — Innate c³")
    ap.add_argument("--description", default="Your AI visibility data, with the counting conventions attached.")
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{24}", a.key):
        sys.exit("key must be a 24-hex connector key")
    p = build(a.key, a.name, pathlib.Path(a.out), a.description)
    print(f"built {p} ({p.stat().st_size} bytes) -> {BASE_URL}{a.key[:6]}…")
