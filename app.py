import os
import json
import re
import random
import requests
import html
import uuid
import io
import csv
import queue
import threading
import secrets
import ipaddress
import socket
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Use the OS native trust store (macOS keychain / Windows cert store / Linux /etc/ssl).
# Fixes "self-signed certificate in certificate chain" on Python.org Python builds
# whose bundled CA path is missing. Harmless no-op on platforms where it's not needed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta, date
import hashlib
import hmac
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response, session, abort
from markupsafe import Markup
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from dateutil.parser import parse
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from utils.simple_file_processor import SimpleMediaFileProcessor

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")

# Cookie/session hardening — kills the basic cross-site CSRF vector on
# credit-spending POSTs and protects the magic-link session cookie in transit.
# SESSION_COOKIE_SECURE is fine for prod (HTTPS-only); local dev over plain
# HTTP just means the cookie won't be set, which is the safer default.
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter for /signal/login. Keyed by email + IP so a
# single bad actor can't burn through tokens for many emails, and a single
# email can't get spammed from many IPs. Process-local — fine for Render
# Starter's single-instance gunicorn; switch to Redis if we scale horizontally.
# ---------------------------------------------------------------------------
_login_rate = defaultdict(list)  # key (email or ip) → [unix_ts, ...]
_LOGIN_RATE_LIMIT = 5            # attempts per window
_LOGIN_RATE_WINDOW = 600         # seconds (10 min)


def _check_login_rate(key):
    """Returns True if the request is under the limit (and records this hit).
    Returns False if the caller should be rejected."""
    now = datetime.utcnow().timestamp()
    _login_rate[key] = [t for t in _login_rate[key] if now - t < _LOGIN_RATE_WINDOW]
    if len(_login_rate[key]) >= _LOGIN_RATE_LIMIT:
        return False
    _login_rate[key].append(now)
    return True


# ---------------------------------------------------------------------------
# SSRF guard for outbound URL verification (_resolve_and_verify_urls).
# Citation URLs are pulled from LLM output, so we can't trust them. Resolve
# the hostname to an IP and reject anything that points at private/loopback/
# link-local / reserved space — this prevents the production HEAD requests
# from being weaponized to probe internal services (Render's metadata, etc.).
# ---------------------------------------------------------------------------
def _is_safe_url(url):
    try:
        host = url.split('/')[2].split(':')[0]
        if not host:
            return False
        # Resolve all A/AAAA records — reject if any of them is unsafe space.
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True
    except Exception:
        # If we can't resolve safely, don't fetch — fail closed.
        return False


# File upload configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'pdf', 'pptx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize file processor
file_processor = SimpleMediaFileProcessor(os.environ.get("ANTHROPIC_API_KEY"))

# Initialize SQLAlchemy
# Render's Postgres provides URLs starting with `postgres://`, but SQLAlchemy 2.x
# requires `postgresql://`. Normalize so the existing app config works against
# either local SQLite or production Postgres without further edits.
_db_url = os.environ.get('DATABASE_URL', 'sqlite:///waitlist.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Postgres-friendly pool settings: recycle stale connections, validate on checkout.
# Harmless on SQLite (the pool is single-connection anyway).
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,  # under Render Postgres's default 5-min idle timeout
}
db = SQLAlchemy(app)

# Define the WaitingList model
class WaitingList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100))
    bespoke_analysis = db.Column(db.Boolean, default=False)
    historical_data = db.Column(db.Boolean, default=False)
    additional_sources = db.Column(db.Boolean, default=False)
    more_results = db.Column(db.Boolean, default=False)
    consulting_services = db.Column(db.Boolean, default=False)
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SharedResult(db.Model):
    __tablename__ = 'shared_results'
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(32), unique=True, index=True, nullable=False)
    payload = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class LeadCapture(db.Model):
    __tablename__ = 'leads'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    slug = db.Column(db.String(32), nullable=True, index=True)
    app_name = db.Column(db.String(64), nullable=False, default='media_analyzer')
    extra = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class CitedPage(db.Model):
    """Cross-audit cache of scraped citation pages. One row per URL: which
    brand names appear on the page (JSON {name: count}), plus title/author/
    published metadata. Audits in the same category reuse each other's
    fetches, and over time this becomes a map of the specific pages AI
    relies on per category."""
    __tablename__ = 'cited_pages'
    id = db.Column(db.Integer, primary_key=True)
    url_hash = db.Column(db.String(40), unique=True, index=True, nullable=False)
    url = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(255), index=True)
    status = db.Column(db.String(16))   # 'ok' | 'blocked' | 'error'
    via = db.Column(db.String(16))      # 'direct' | 'reader'
    title = db.Column(db.Text)
    author = db.Column(db.String(255))
    published = db.Column(db.String(64))
    content_len = db.Column(db.Integer)
    name_counts = db.Column(db.Text)    # JSON {brand_or_competitor_name: count}
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    slug = db.Column(db.String(32), nullable=False, index=True)
    frequency = db.Column(db.String(16), nullable=False)  # 'realtime' | 'daily'
    params = db.Column(db.Text, nullable=True)  # JSON string of query params/signature
    active = db.Column(db.Boolean, default=True, nullable=False)
    last_checked_at = db.Column(db.DateTime, nullable=True)
    last_seen_published_at = db.Column(db.DateTime, nullable=True)
    unsubscribe_token = db.Column(db.String(64), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# ---------------------------------------------------------------------------
# PR Signal Finder — paid-tier models
# ---------------------------------------------------------------------------

class SignalUser(db.Model):
    __tablename__ = 'signal_users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = db.Column(db.DateTime, nullable=True)

class CreditBalance(db.Model):
    __tablename__ = 'credit_balances'
    user_id = db.Column(db.Integer, db.ForeignKey('signal_users.id'), primary_key=True)
    credits_remaining = db.Column(db.Integer, default=0, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Purchase(db.Model):
    __tablename__ = 'purchases'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signal_users.id'), index=True, nullable=False)
    stripe_session_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    amount_cents = db.Column(db.Integer, nullable=False)
    credits_granted = db.Column(db.Integer, nullable=False)
    product_label = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class AuditRun(db.Model):
    __tablename__ = 'audit_runs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('signal_users.id'), index=True, nullable=True)
    slug = db.Column(db.String(32), nullable=True, index=True)
    tier = db.Column(db.String(16), default='free', nullable=False)
    prompt_count = db.Column(db.Integer, nullable=True)
    llm_count = db.Column(db.Integer, nullable=True)
    credits_consumed = db.Column(db.Integer, default=0, nullable=False)
    problem_statement = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class InboundAudit(db.Model):
    """One row per self-serve audit — the operator's warm-lead pipeline. The
    problem_statement IS the lead: it names the brand + what they want AI to
    surface them for. Attribution (UTM / external referrer) is captured on the
    landing GET via the session; coarse geo is derived from the IP. Surfaced only
    at the operator-gated /inbound view + the daily digest; never public."""
    __tablename__ = 'inbound_audits'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    slug = db.Column(db.String(32), nullable=True, index=True)
    brand = db.Column(db.String(200), nullable=True)
    category = db.Column(db.String(400), nullable=True)
    problem_statement = db.Column(db.Text, nullable=True)
    ip = db.Column(db.String(64), nullable=True)
    geo = db.Column(db.String(160), nullable=True)
    utm_source = db.Column(db.String(120), nullable=True, index=True)
    utm_medium = db.Column(db.String(120), nullable=True)
    utm_campaign = db.Column(db.String(120), nullable=True)
    referrer = db.Column(db.String(400), nullable=True)
    # Added after launch (prod ALTER in _ensure_inbound_columns). status tracks the
    # audit lifecycle so incomplete/abandoned/errored/rate-limited starts are
    # captured as leads, not only completions. email is the optional opt-in lead
    # contact. is_operator flags biz-dev batch / operator runs so the default
    # /inbound view shows real DIY demand, not our own testing.
    status = db.Column(db.String(16), nullable=True, default='started', index=True)
    email = db.Column(db.String(254), nullable=True, index=True)
    is_operator = db.Column(db.Boolean, nullable=True, default=False, index=True)


class PageVisit(db.Model):
    """One row per HUMAN visit to a tool page (homepage or a shared report) —
    first-party traffic analytics for the operator /traffic view. Bots and
    operator/self IPs are skipped at log time. Unique visitors are counted as
    distinct IPs over a window. Never public; surfaced only at /traffic."""
    __tablename__ = 'page_visits'
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    kind = db.Column(db.String(16), nullable=True, index=True)      # 'home' | 'report'
    path = db.Column(db.String(200), nullable=True)
    slug = db.Column(db.String(32), nullable=True, index=True)      # report slug if kind='report'
    ip = db.Column(db.String(64), nullable=True, index=True)        # for unique-visitor counting
    referrer = db.Column(db.String(400), nullable=True)
    ref_host = db.Column(db.String(160), nullable=True, index=True) # parsed referrer host (linkedin.com)
    utm_source = db.Column(db.String(120), nullable=True, index=True)
    utm_medium = db.Column(db.String(120), nullable=True)
    utm_campaign = db.Column(db.String(120), nullable=True)


class LoginToken(db.Model):
    __tablename__ = 'login_tokens'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    token_hash = db.Column(db.String(128), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# ---------------------------------------------------------------------------
# MVP branch only: per-IP free-tier rate limit + feedback widget
# ---------------------------------------------------------------------------

class FreeAuditUse(db.Model):
    __tablename__ = 'free_audit_use'
    id = db.Column(db.Integer, primary_key=True)
    ip = db.Column(db.String(64), nullable=False, index=True)
    day = db.Column(db.Date, nullable=False, index=True)
    count = db.Column(db.Integer, default=0, nullable=False)
    __table_args__ = (db.UniqueConstraint('ip', 'day', name='uniq_ip_day'),)


class AuditLead(db.Model):
    """Per-email lead capture + hard cap on free audits per email. Each audit
    costs real LLM API spend; without this, a single shared link could blow up
    into hundreds of dollars in casual self-serve runs. Doubles as the lead-
    capture for follow-up (we know who's running audits and on what brand).
    Cap is set by EMAIL_AUDIT_CAP env var (default 1). IPs in
    FREE_AUDIT_BYPASS_IPS skip the email gate entirely — for the operator's
    own testing without polluting the lead table.
    """
    __tablename__ = 'audit_leads'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    audit_count = db.Column(db.Integer, default=0, nullable=False)
    first_seen = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_ip = db.Column(db.String(64), nullable=True)
    last_slug = db.Column(db.String(32), nullable=True, index=True)
    last_problem_statement = db.Column(db.Text, nullable=True)


class AuditFeedback(db.Model):
    __tablename__ = 'audit_feedback'
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(32), nullable=False, index=True)
    rating = db.Column(db.String(16), nullable=False)  # 'up' | 'down'
    ip = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class TrackedLink(db.Model):
    """A per-recipient short link (/r/<token>) that redirects to a
    /signal/<slug> report and logs genuine human opens. Lets the operator see
    whether a specific pitch recipient opened their report — and when — so
    follow-up can be timed while the brand is top of mind."""
    __tablename__ = 'tracked_links'
    id = db.Column(db.Integer, primary_key=True)
    # 64 (widened from 16 at startup by _ensure_inbound_columns) so vanity
    # tokens like 'citi-flagship-proposal' fit; random tokens are 8 hex chars.
    token = db.Column(db.String(64), unique=True, index=True, nullable=False)
    slug = db.Column(db.String(32), nullable=False, index=True)
    recipient = db.Column(db.String(160), nullable=True)   # e.g. "Jennifer McGuire — Swarovski"
    campaign = db.Column(db.String(80), nullable=True)      # optional grouping label
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class LinkClick(db.Model):
    """One row per hit on a TrackedLink. is_bot separates link-preview crawlers
    (LinkedIn/Slack/iMessage/Outlook fetch the URL to build a preview card
    BEFORE any human clicks) from real human opens — so the 'did they open it'
    signal isn't inflated by automated fetches."""
    __tablename__ = 'link_clicks'
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), index=True, nullable=False)   # matches TrackedLink.token width
    is_bot = db.Column(db.Boolean, default=False, nullable=False, index=True)
    ip = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    referer = db.Column(db.String(500), nullable=True)
    clicked_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


class Outreach(db.Model):
    """Lightweight outreach CRM: one row per prospect we've pitched a report to.
    Tracks pipeline status, ties to the prospect's tracked link (so a genuine
    open auto-advances status to 'opened'), and drives the follow-up reminder
    cadence ('days from sent', default 5,14)."""
    __tablename__ = 'outreach'
    id = db.Column(db.Integer, primary_key=True)
    prospect_name = db.Column(db.String(160), nullable=False)
    prospect_title = db.Column(db.String(200), nullable=True)
    company = db.Column(db.String(120), nullable=True)
    slug = db.Column(db.String(32), nullable=False)                    # report slug
    link_token = db.Column(db.String(64), nullable=True, index=True)   # -> TrackedLink.token (matches width)
    channel = db.Column(db.String(24), default='linkedin', nullable=False)  # linkedin | email
    # queued | sent | opened | replied | call_scheduled | won | cold | passed
    status = db.Column(db.String(24), default='queued', nullable=False, index=True)
    cadence = db.Column(db.String(40), default='5,14', nullable=False)  # comma days from sent
    followup_count = db.Column(db.Integer, default=0, nullable=False)
    insight = db.Column(db.Text, nullable=True)        # one-line lead insight, seeds proposed text
    relationship = db.Column(db.String(240), nullable=True)  # shared-connection hook (e.g. "fellow Oregon alum")
    message = db.Column(db.Text, nullable=True)        # full suggested initial outreach message
    notes = db.Column(db.Text, nullable=True)
    sent_at = db.Column(db.DateTime, nullable=True)
    last_activity_at = db.Column(db.DateTime, nullable=True)
    next_followup_due = db.Column(db.Date, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('prospect_name', 'slug', name='uniq_prospect_slug'),)


# Create the database tables
with app.app_context():
    db.create_all()

# API keys and configuration
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NYT_API_KEY = os.environ.get("NYT_API_KEY")
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY")

# Debug logging for API keys
print(f"NEWS_API_KEY is {'set' if NEWS_API_KEY else 'NOT SET'}")
print(f"ANTHROPIC_API_KEY is {'set' if ANTHROPIC_API_KEY else 'NOT SET'}")
print(f"GA_MEASUREMENT_ID is {'set' if GA_MEASUREMENT_ID else 'NOT SET'}")
print(f"OPENAI_API_KEY is {'set' if OPENAI_API_KEY else 'NOT SET'}")
print(f"NYT_API_KEY is {'set' if NYT_API_KEY else 'NOT SET'}")
print(f"GUARDIAN_API_KEY is {'set' if GUARDIAN_API_KEY else 'NOT SET'}")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

# Centralized Claude model IDs. Anthropic retires dated model snapshots ~12
# months out, and the failure mode is a silent 404 NotFoundError at call time
# (observed June 2026: claude-sonnet-4-20250514 retired, killing prompt-gen +
# exec-summary). Keeping every call site on these constants means the next
# migration is a one-line change, not a 13-site grep. Override per-deploy via
# env vars without a code change.
CLAUDE_SONNET = os.environ.get("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6")
CLAUDE_HAIKU = os.environ.get("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    gemini_client = None
print(f"GEMINI_API_KEY is {'set' if GEMINI_API_KEY else 'NOT SET'}")

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("PPLX_API_KEY")
try:
    perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai", timeout=60.0) if PERPLEXITY_API_KEY else None
except Exception:
    perplexity_client = None
print(f"PERPLEXITY_API_KEY is {'set' if PERPLEXITY_API_KEY else 'NOT SET'}")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
try:
    # Tight client-level timeout for OpenRouter — Grok via x-ai is the worst-case
    # latency on this stack and we don't want a hung Grok call to block the audit.
    openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", timeout=30.0) if OPENROUTER_API_KEY else None
except Exception:
    openrouter_client = None
print(f"OPENROUTER_API_KEY is {'set' if OPENROUTER_API_KEY else 'NOT SET'}")

# ---------------------------------------------------------------------------
# Stripe configuration for paid tier
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_SINGLE = os.environ.get("STRIPE_PRICE_SINGLE")
STRIPE_PRICE_PACK_5 = os.environ.get("STRIPE_PRICE_PACK_5")
SIGNAL_BASE_URL = os.environ.get("SIGNAL_BASE_URL")  # e.g. https://signal.innatec3.com, falls back to request origin

STRIPE_PRODUCTS = {
    "single": {
        "label": "Single audit",
        "credits": 1,
        "amount_display": "$25",
        "price_id": STRIPE_PRICE_SINGLE,
        "description": "One paid audit: 100 prompts × 5 LLMs, top 25 media targets.",
    },
    "pack_5": {
        "label": "5 audit credits",
        "credits": 5,
        "amount_display": "$100",
        "price_id": STRIPE_PRICE_PACK_5,
        "description": "5 paid audits at $20 each. Credits never expire.",
    },
}

try:
    import stripe as stripe_lib
    if STRIPE_SECRET_KEY:
        stripe_lib.api_key = STRIPE_SECRET_KEY
except Exception:
    stripe_lib = None
print(f"STRIPE_SECRET_KEY is {'set' if STRIPE_SECRET_KEY else 'NOT SET'}")

def analyze_articles(articles, query):
    """Extract key metrics and patterns from articles."""
    # Batch sentiment analysis for all articles
    texts = [f"{article['title']} {article['description'] or ''}" for article in articles]
    
    # Create a numbered list for Claude to reference
    numbered_texts = "\n\n".join(f"Text {i+1}:\n{text}" for i, text in enumerate(texts))
    
    # Disable sentiment scoring entirely (set neutral for all)
    if True:
        for article in articles:
            article['sentiment'] = 0
    else:
        try:
            response = anthropic.messages.create(
                model=CLAUDE_HAIKU,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze the sentiment of each numbered text and respond with a JSON array of sentiment scores between -1 (most negative) and 1 (most positive).

For each text:
- Consider the overall tone, word choice, and context
- Score negative news/criticism closer to -1
- Score positive news/achievements closer to +1
- Score neutral/factual content closer to 0

IMPORTANT: Your response must begin with a valid JSON array containing only numbers, like this:
[-0.8, 0.5, 0.2, -0.4, 0.1]

Do not include any explanations before the array. You can add explanations after the array if needed.

Here are the texts to analyze:

{numbered_texts}"""
                }]
            )
            
            # Extract the array from Claude's response by finding text between [ and ]
            sentiment_text = response.content[0].text
            print("Anthropic API Response:", sentiment_text)  # Log the response for debugging
            array_match = re.search(r'\[(.*?)\]', sentiment_text, re.DOTALL)
            if array_match:
                # Parse the comma-separated values into floats
                sentiment_values = re.findall(r'-?\d+(?:\.\d+)?', array_match.group(1))
                sentiments = []
                for value in sentiment_values:
                    try:
                        parsed_value = float(value)
                        sentiments.append(parsed_value)
                    except ValueError:
                        pass
                # Use available sentiments, pad with 0 if needed
                for i, article in enumerate(articles):
                    if i < len(sentiments):
                        sentiment = max(-1, min(1, sentiments[i]))
                        article['sentiment'] = sentiment
                    else:
                        article['sentiment'] = 0
            else:
                # If no array found, use neutral sentiment
                for article in articles:
                    article['sentiment'] = 0
        except Exception as e:
            print("Error calling or parsing Anthropic sentiment response:", e)
            # Default to neutral if API call or parsing fails
            for article in articles:
                article['sentiment'] = 0
    
    # Publication timeline with articles
    dates = {}
    articles_by_date = {}
    for article in articles:
        date = parse(article['publishedAt']).strftime('%Y-%m-%d')
        dates[date] = dates.get(date, 0) + 1
        
        # Store articles for each date
        if date not in articles_by_date:
            articles_by_date[date] = []
        articles_by_date[date].append({
            'title': article['title'],
            'source': article['source']['name'],
            'url': article['url'],
            'sentiment': article['sentiment']
        })
    
    # Create timeline with articles
    timeline = []
    for date, count in sorted(dates.items()):
        # Get the article with the highest absolute sentiment score for this date
        date_articles = articles_by_date[date]
        peak_article = max(date_articles, key=lambda x: abs(x['sentiment']))
        
        timeline.append({
            'date': date,
            'count': count,
            'peak_article': peak_article
        })
    
    # News source distribution
    sources = Counter(article['source']['name'] for article in articles)
    top_sources = [{'name': name, 'count': count} 
                   for name, count in sources.most_common(10)]
    
    # Topic extraction (multi-word phrases and filtered unigrams)
    top_topics = extract_topics(articles, query)

    # Calculate average sentiment
    sentiments = [article['sentiment'] for article in articles]
    avg_sentiment = sum(sentiments) / len(articles) if articles else 0

    return {
        'timeline': timeline,
        'sources': top_sources,
        'topics': top_topics,
        'total_articles': len(articles),
        'date_range': {
            'start': timeline[0]['date'] if timeline else None,
            'end': timeline[-1]['date'] if timeline else None
        },
        'avg_sentiment': avg_sentiment
    }

def fetch_rss_articles(query, from_date_str=None, to_date_str=None, max_items=50):
    """
    Fallback: fetch recent articles from Google News RSS without requiring NEWS_API_KEY.
    Tries a few query variants (quoted, with when:Xd) and returns a list of article dicts
    compatible with analyze_articles().
    """
    import urllib.parse
    import xml.etree.ElementTree as ET
    from datetime import datetime
    import email.utils as eut

    if not query:
        return []

    # Parse date bounds (YYYY-MM-DD) if provided
    def parse_iso_date(dstr):
        try:
            return datetime.fromisoformat(dstr).date() if dstr else None
        except Exception:
            return None

    from_date = parse_iso_date(from_date_str)
    to_date = parse_iso_date(to_date_str)

    # Compute an approximate day window (1..60) for Google News "when:Xd" hint
    day_window = None
    try:
        if from_date and to_date:
            delta_days = (to_date - from_date).days + 1
            if delta_days > 0:
                day_window = max(1, min(60, delta_days))
    except Exception:
        day_window = None

    # Build query variants to improve recall
    cleaned = query.strip()
    variants = [cleaned]

    # Quoted variant (helps for multi-word brands)
    if " " in cleaned:
        variants.append(f'"{cleaned}"')

    # when:Xd variant to hint recency (if user selected a date range)
    if day_window:
        variants.append(f'{cleaned} when:{day_window}d')

    # Dedupe while preserving order
    seen = set()
    query_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            query_variants.append(v)

    all_articles = []
    seen_keys = set()

    for q in query_variants:
        qs = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={qs}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, timeout=12, headers={
                "User-Agent": "Mozilla/5.0 (compatible; InnateC3/1.0; +https://innatec3.com)"
            })
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"RSS fetch error for '{q}': {e}")
            continue

        channel = root.find('channel')
        if channel is None:
            continue

        for item in channel.findall('item'):
            try:
                title = (item.findtext('title') or '').strip()
                link = (item.findtext('link') or '').strip()
                description = (item.findtext('description') or '').strip()
                pub_raw = item.findtext('pubDate') or ''
                # pubDate like: Wed, 13 Aug 2025 15:04:05 GMT
                dt = eut.parsedate_to_datetime(pub_raw) if pub_raw else None
                dt_date = dt.date() if dt else None

                # Date filtering (inclusive)
                if from_date and dt_date and dt_date < from_date:
                    continue
                if to_date and dt_date and dt_date > to_date:
                    continue

                source_tag = item.find('source')
                source_name = (source_tag.text.strip() if source_tag is not None and source_tag.text else 'Google News')
                try:
                    source_name = html.unescape(source_name)
                except Exception:
                    pass

                # Clean/normalize for display and dedupe
                try:
                    norm_link = normalize_url(link) or link
                except Exception:
                    norm_link = link

                try:
                    cleaned_title = html.unescape(title or '').strip()
                    if source_name:
                        cleaned_title = re.sub(r'\s+[-—]\s*' + re.escape(source_name) + r'\s*$', '', cleaned_title)
                    cleaned_desc = re.sub(r'<[^>]+>', '', description or '')
                    cleaned_desc = html.unescape(cleaned_desc).strip()
                except Exception:
                    cleaned_title = (title or '').strip()
                    cleaned_desc = description

                key = (cleaned_title, norm_link)
                if key in seen_keys:
                    continue

                all_articles.append({
                    'title': cleaned_title,
                    'description': cleaned_desc,
                    'publishedAt': (dt.isoformat() if dt else datetime.utcnow().isoformat()),
                    'source': {'name': source_name},
                    'url': norm_link,
                    'api_source': 'google_news_rss'
                })
                seen_keys.add(key)

                if len(all_articles) >= max_items:
                    break
            except Exception:
                continue

        if len(all_articles) >= max_items:
            break

    return all_articles


def fetch_news_api_articles(query, from_date_str=None, to_date_str=None, language="en", sources=None, page_size=50):
    """
    Fetch recent articles from NewsAPI.org using the 'everything' endpoint.
    Returns a list of article dicts compatible with analyze_articles().
    """
    if not query:
        return []
    if not NEWS_API_KEY:
        return []

    # Build ISO date-times if provided (NewsAPI expects RFC3339/ISO8601)
    def to_iso(dt_str, end=False):
        try:
            if not dt_str:
                return None
            # Pad time to start or end of day
            return f"{dt_str}T23:59:59Z" if end else f"{dt_str}T00:00:00Z"
        except Exception:
            return None

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": (language or "en"),
        "pageSize": max(1, min(100, page_size)),
        "apiKey": NEWS_API_KEY,
    }
    from_iso = to_iso(from_date_str, end=False)
    to_iso_str = to_iso(to_date_str, end=True)
    if from_iso:
        params["from"] = from_iso
    if to_iso_str:
        params["to"] = to_iso_str
    if sources:
        # NewsAPI expects a comma-separated list of allowed sources
        params["sources"] = sources

    url = "https://newsapi.org/v2/everything"
    try:
        resp = requests.get(url, params=params, timeout=12, headers={
            "User-Agent": "Mozilla/5.0 (compatible; InnateC3/1.0; +https://innatec3.com)"
        })
        resp.raise_for_status()
        data = resp.json()
        items = data.get("articles", []) or []
    except Exception as e:
        print(f"NewsAPI fetch error for '{query}': {e}")
        return []

    articles = []
    seen = set()
    for it in items:
        try:
            title = (it.get("title") or "").strip()
            link = (it.get("url") or "").strip()
            if not title or not link:
                continue
            key = (title, link)
            if key in seen:
                continue
            seen.add(key)
            desc = (it.get("description") or "").strip()
            pub = it.get("publishedAt") or datetime.utcnow().isoformat()
            source_name = (it.get("source", {}) or {}).get("name") or "NewsAPI"

            articles.append({
                "title": title,
                "description": desc,
                "publishedAt": pub,
                "source": {"name": source_name},
                "url": link,
                "api_source": "newsapi"
            })
            if len(articles) >= page_size:
                break
        except Exception:
            continue

    return articles


def fetch_news_api_articles_sliced(
    query,
    from_date_str=None,
    to_date_str=None,
    language="en",
    sources=None,
    per_slice=8,
    slice_days=1
):
    """
    Fetch NewsAPI articles distributed across the selected date range to avoid
    recency clumping. Partition the time range into slices (default 1 day) and
    request a few items per slice, deduping afterward.
    """
    if not query or not NEWS_API_KEY:
        return []

    def parse_date_only(d):
        try:
            return datetime.fromisoformat(d).date() if d else None
        except Exception:
            return None

    start_date = parse_date_only(from_date_str)
    end_date = parse_date_only(to_date_str)

    # Default to last 30 days if bounds missing
    if not start_date or not end_date:
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=29)
    else:
        start_dt = start_date
        end_dt = end_date

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Build date slices
    slices = []
    d = start_dt
    while d <= end_dt:
        slice_start = d
        slice_end = min(end_dt, d + timedelta(days=slice_days - 1))
        slices.append((slice_start.isoformat(), slice_end.isoformat()))
        d = slice_end + timedelta(days=1)

    out = []
    seen = set()
    for (fs, ts) in slices:
        items = fetch_news_api_articles(query, fs, ts, language=language, sources=sources, page_size=per_slice)
        for it in items:
            key = (it.get("title") or "", it.get("url") or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(it)

    return sort_articles_desc(out)

def fetch_nyt_articles(query, from_date_str=None, to_date_str=None, max_items=60):
    """
    Fetch articles from The New York Times Article Search API.
    Maps to our unified article schema.
    """
    if not query or not (NYT_API_KEY):
        return []
    try:
        # NYT expects YYYYMMDD
        def ymd(s):
            try:
                return datetime.fromisoformat(s).strftime("%Y%m%d") if s else None
            except Exception:
                return None
        begin = ymd(from_date_str)
        end = ymd(to_date_str)
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        out, seen = [], set()
        page = 0
        while len(out) < max_items and page < 10:
            params = {
                "q": query,
                "sort": "newest",
                "api-key": NYT_API_KEY,
                "page": page
            }
            if begin: params["begin_date"] = begin
            if end: params["end_date"] = end
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code != 200:
                break
            data = resp.json() or {}
            docs = ((data.get("response") or {}).get("docs")) or []
            if not docs:
                break
            for d in docs:
                title = ((d.get("headline") or {}).get("main")) or ""
                link = d.get("web_url") or ""
                if not title or not link:
                    continue
                key = (title.strip(), link.strip())
                if key in seen:
                    continue
                seen.add(key)
                desc = (d.get("abstract") or "").strip()
                pub = d.get("pub_date") or datetime.utcnow().isoformat()
                out.append({
                    "title": title.strip(),
                    "description": desc,
                    "publishedAt": pub,
                    "source": {"name": "The New York Times"},
                    "url": link.strip(),
                    "api_source": "nyt"
                })
                if len(out) >= max_items:
                    break
            page += 1
        return sort_articles_desc(out)
    except Exception as e:
        print("NYT fetch error:", e)
        return []

def fetch_guardian_articles(query, from_date_str=None, to_date_str=None, max_items=60):
    """
    Fetch articles from The Guardian Content API.
    Maps to our unified article schema.
    """
    if not query or not (GUARDIAN_API_KEY):
        return []
    try:
        url = "https://content.guardianapis.com/search"
        out, seen = [], set()
        page = 1
        page_size = 50
        while len(out) < max_items and page <= 10:
            params = {
                "q": query,
                "order-by": "newest",
                "show-fields": "trailText",
                "api-key": GUARDIAN_API_KEY,
                "page": page,
                "page-size": page_size
            }
            if from_date_str: params["from-date"] = from_date_str
            if to_date_str: params["to-date"] = to_date_str
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code != 200:
                break
            data = resp.json() or {}
            results = ((data.get("response") or {}).get("results")) or []
            if not results:
                break
            for r in results:
                title = (r.get("webTitle") or "").strip()
                link = (r.get("webUrl") or "").strip()
                if not title or not link:
                    continue
                key = (title, link)
                if key in seen:
                    continue
                seen.add(key)
                desc = (((r.get("fields") or {}).get("trailText")) or "").strip()
                pub = r.get("webPublicationDate") or datetime.utcnow().isoformat()
                out.append({
                    "title": title,
                    "description": desc,
                    "publishedAt": pub,
                    "source": {"name": "The Guardian"},
                    "url": link,
                    "api_source": "guardian"
                })
                if len(out) >= max_items:
                    break
            page += 1
        return sort_articles_desc(out)
    except Exception as e:
        print("Guardian fetch error:", e)
        return []

def normalize_url(u: str) -> str:
    """Normalize URLs to improve duplicate detection across sources."""
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        if not u:
            return ""
        p = urlparse(u)
        netloc = p.netloc.lower()

        # If it's a Google News redirect link, try to extract the real URL
        if "news.google.com" in netloc:
            try:
                qs = dict(parse_qsl(p.query))
                for k in ("url", "u"):
                    if k in qs and qs[k]:
                        nested = qs[k] if isinstance(qs[k], str) else qs[k][0]
                        return normalize_url(nested)
            except Exception:
                pass

        # Strip known tracking params
        filtered_q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl.startswith("utm_") or kl in {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ved"}:
                continue
            filtered_q.append((k, v))

        path = p.path.rstrip("/")
        return urlunparse((p.scheme, netloc, path, "", urlencode(filtered_q), ""))
    except Exception:
        return u or ""


def merge_articles_dedupe(primary, supplement, max_total=60):
    """
    Merge two article lists, prioritizing 'primary', then supplementing with 'supplement'.
    De-duplicate by normalized URL and (lowercased) title.
    """
    result = []
    seen_urls = set()
    seen_titles = set()

    def add_all(lst):
        for a in lst or []:
            url = (a.get("url") or "").strip()
            norm = normalize_url(url)
            title = (a.get("title") or "").strip().lower()

            url_key = (norm or url).lower()
            if url_key and url_key in seen_urls:
                continue
            if title and title in seen_titles:
                continue

            seen_urls.add(url_key)
            if title:
                seen_titles.add(title)
            result.append(a)
            if len(result) >= max_total:
                return

    add_all(primary)
    if len(result) < max_total:
        add_all(supplement)
    return result


def sort_articles_desc(articles):
    """Sort articles by publishedAt desc when possible."""
    def ts(a):
        try:
            return parse(a.get("publishedAt") or "").timestamp()
        except Exception:
            return 0.0
    return sorted(articles or [], key=ts, reverse=True)


# Helper: extract domain from URL
def _domain(u: str) -> str:
    try:
        from urllib.parse import urlparse
        d = (urlparse(u).netloc or "").lower()
        if d.startswith("www."):
            d = d[4:]
        # strip port if present
        return d.split(":")[0]
    except Exception:
        return ""

# Seeds (editable later or move to JSON config)
PR_DOMAINS = {
    "prnewswire.com", "businesswire.com", "globenewswire.com", "prweb.com",
    "newswire.com", "einnews.com"
}
TIER1_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com",
    "ft.com", "apnews.com", "cnbc.com", "washingtonpost.com",
    "theguardian.com", "techcrunch.com", "forbes.com"
}

# Load domain config sets from ./config if available
def _load_cfg_set(rel_path: str, key: str) -> set:
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "config", rel_path)
        with open(cfg_path, "r") as f:
            data = json.load(f)
        arr = data.get(key) or []
        return set([x.strip().lower() for x in arr if isinstance(x, str) and x.strip()])
    except Exception:
        return set()

def _init_domain_config():
    global TIER1_DOMAINS, PR_DOMAINS
    try:
        tier1 = _load_cfg_set("domains_tier1.json", "tier1")
        if tier1:
            TIER1_DOMAINS.clear()
            TIER1_DOMAINS.update(tier1)
    except Exception:
        pass
    try:
        pr = _load_cfg_set("domains_pr.json", "pr")
        if pr:
            PR_DOMAINS.clear()
            PR_DOMAINS.update(pr)
    except Exception:
        pass
    # Optional groups
    globals()["AGGREGATOR_DOMAINS"] = _load_cfg_set("domains_aggregators.json", "aggregators")
    cm = _load_cfg_set("domains_community.json", "community")
    corp = _load_cfg_set("domains_community.json", "corporate")
    globals()["COMMUNITY_DOMAINS"] = cm
    globals()["CORPORATE_DOMAINS"] = corp

_init_domain_config()

def _load_sectors():
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "sectors.json")
        with open(cfg_path, "r") as f:
            data = json.load(f)
        globals()["SECTORS"] = data if isinstance(data, dict) else {}
    except Exception:
        globals()["SECTORS"] = {}

_load_sectors()

EDITORIAL_ONLY_DEFAULT = False
NARRATIVE_CAP_PER_SIDE = 300

def classify_domain(domain: str) -> str:
    """
    Return one of: editorial_tier1, editorial, pr, aggregator, community, corporate, other
    """
    d = (domain or "").lower()
    if not d:
        return "other"
    if d in PR_DOMAINS:
        return "pr"
    if "AGGREGATOR_DOMAINS" in globals() and d in AGGREGATOR_DOMAINS:
        return "aggregator"
    if "COMMUNITY_DOMAINS" in globals() and d in COMMUNITY_DOMAINS:
        return "community"
    if "CORPORATE_DOMAINS" in globals() and d in CORPORATE_DOMAINS:
        return "corporate"
    if d in TIER1_DOMAINS:
        return "editorial_tier1"
    # treat unknowns as editorial unless clearly non-editorial (conservative)
    return "editorial"

def filter_editorial_articles(articles):
    out = []
    for a in (articles or []):
        dom = _domain(a.get("url") or "")
        cls = classify_domain(dom)
        if cls in ("editorial", "editorial_tier1"):
            out.append(a)
    return out

def filter_by_sector(articles, sector_key: str):
    """
    Filter articles by sector definition loaded from config/sectors.json.
    Keep an article if its domain is in sector.domains OR title/description contains any sector keyword.
    """
    try:
        sectors = globals().get("SECTORS") or {}
        sect = sectors.get(sector_key or "")
        if not sect:
            return articles or []
        domains = set((sect.get("domains") or []))
        domains = {d.strip().lower() for d in domains if isinstance(d, str)}
        keywords = [k.strip().lower() for k in (sect.get("keywords") or []) if isinstance(k, str) and k.strip()]
        if not domains and not keywords:
            return articles or []
        out = []
        for a in (articles or []):
            try:
                title = (a.get("title") or "").lower()
                desc = (a.get("description") or "").lower()
                dom = _domain(a.get("url") or "")
                domain_hit = dom in domains if dom else False
                keyword_hit = any(((k in title) or (k in desc)) for k in keywords) if (title or desc) and keywords else False
                if domain_hit or keyword_hit:
                    out.append(a)
            except Exception:
                continue
        return out
    except Exception:
        return articles or []

def timeline_from_articles(articles):
    by_date = {}
    for a in (articles or []):
        try:
            d = parse(a.get("publishedAt") or "").strftime("%Y-%m-%d")
        except Exception:
            continue
        by_date.setdefault(d, {"date": d, "count": 0, "peak_article": None})
        by_date[d]["count"] += 1
        # choose first as representative (no sentiment)
        if not by_date[d]["peak_article"]:
            by_date[d]["peak_article"] = {
                "title": a.get("title"),
                "source": (a.get("source") or {}).get("name"),
                "url": a.get("url"),
                "sentiment": 0
            }
    return [by_date[k] for k in sorted(by_date.keys())]

def _dt_from_iso(iso_s: str):
    try:
        return parse(iso_s)
    except Exception:
        return None

def _recency_share_48h(articles):
    try:
        now = datetime.utcnow()
        start = now - timedelta(hours=48)
        recent = 0
        for a in articles or []:
            dt = _dt_from_iso(a.get("publishedAt") or "")
            if dt and dt.replace(tzinfo=None) >= start:
                recent += 1
        total = max(1, len(articles or []))
        return recent / total
    except Exception:
        return 0.0

def _top_sources_by_domain(articles, top_n=15):
    from collections import Counter
    c = Counter()
    for a in articles or []:
        d = _domain(a.get("url") or "")
        if d:
            c[d] += 1
    out = []
    for d, cnt in c.most_common(top_n):
        out.append({
            "name": d,
            "count": cnt,
            "tier": "tier1" if d in TIER1_DOMAINS else "other",
            "is_pr": d in PR_DOMAINS
        })
    return out, c

def _peaks_from_timeline(analysis, top_n=5):
    try:
        tl = (analysis or {}).get("timeline") or []
        top = sorted(tl, key=lambda x: x.get("count", 0), reverse=True)[:top_n]
        out = []
        for t in top:
            pa = t.get("peak_article") or {}
            out.append({
                "date": t.get("date"),
                "count": t.get("count"),
                "headline": pa.get("title"),
                "source": pa.get("source")
            })
        return out
    except Exception:
        return []

def aggregate_query_payload(query, articles, analysis):
    # Editorial-only set (default ON)
    editorial_articles = filter_editorial_articles(articles) if EDITORIAL_ONLY_DEFAULT else (articles or [])
    total = len(articles or [])
    total_editorial = len(editorial_articles)

    # Rebuild timeline from editorial articles for accuracy
    daily_counts = timeline_from_articles(editorial_articles)

    # Top sources by domain (editorial only)
    top_sources, domain_counts = _top_sources_by_domain(editorial_articles, top_n=15)
    distinct_sources = len(domain_counts)

    # Shares
    pr_hits = 0
    tier1_hits = 0
    for a in (articles or []):
        d = _domain(a.get("url") or "")
        if d in PR_DOMAINS:
            pr_hits += 1
    for a in editorial_articles:
        if _domain(a.get("url") or "") in TIER1_DOMAINS:
            tier1_hits += 1
    pr_share = (pr_hits / max(1, total))
    editorial_share = (total_editorial / max(1, total))
    tier1_share = (tier1_hits / max(1, total_editorial)) if total_editorial else 0.0

    # Topics from non-PR items (avoid PR keyword pollution)
    non_pr_articles = [a for a in editorial_articles if _domain(a.get("url") or "") not in PR_DOMAINS]
    keyphrases = extract_topics(non_pr_articles, query)

    # Peaks from editorial timeline
    peaks = []
    if daily_counts:
        # Attach representative article from editorial set
        for t in daily_counts:
            peaks.append({
                "date": t["date"],
                "count": t["count"],
                "headline": (t.get("peak_article") or {}).get("title"),
                "source": (t.get("peak_article") or {}).get("source")
            })

    # Representative headlines (editorial, most recent)
    reps = []
    for a in editorial_articles[:12]:
        reps.append({
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "date": (a.get("publishedAt") or "").split("T")[0],
            "url": a.get("url")
        })

    # Date range (fallback to analysis if computed; else infer from editorial timeline)
    date_range = (analysis or {}).get("date_range") or {}
    if not date_range.get("start") or not date_range.get("end"):
        if daily_counts:
            date_range = {"start": daily_counts[0]["date"], "end": daily_counts[-1]["date"]}

    payload = {
        "summary": {
            "total_articles": total_editorial if EDITORIAL_ONLY_DEFAULT else total,
            "date_range": {"start": date_range.get("start"), "end": date_range.get("end")},
            "notes": []
        },
        "volume": {
            "daily_counts": daily_counts,
            "peaks": peaks,
            "recency_share_48h": _recency_share_48h(editorial_articles)
        },
        "outlets": {
            "top_sources": top_sources,
            "pr_share": pr_share,
            "editorial_share": editorial_share,
            "tier1_share": tier1_share,
            "distinct_sources": distinct_sources
        },
        "topics": {
            "keyphrases": [{"phrase": kv["topic"], "count": kv["count"]} for kv in keyphrases],
            "regulatory": []
        },
        "entities": {
            "spokespeople": [],
            "partners": [],
            "competitors": []
        },
        "headlines": {
            "representative": reps
        }
    }
    return payload

def compute_overlap(left_payload, right_payload):
    # Overlap/distinct by sources (domain) and keyphrases (phrase)
    ls = {s["name"]: s["count"] for s in (left_payload.get("outlets", {}).get("top_sources") or [])}
    rs = {s["name"]: s["count"] for s in (right_payload.get("outlets", {}).get("top_sources") or [])}
    lset, rset = set(ls.keys()), set(rs.keys())
    overlap_sources = []
    for d in sorted(lset & rset):
        overlap_sources.append({"name": d, "left": ls[d], "right": rs[d]})
    distinct_left_sources = [{"name": d, "count": ls[d]} for d in sorted(lset - rset)]
    distinct_right_sources = [{"name": d, "count": rs[d]} for d in sorted(rset - lset)]

    lt = {t["phrase"]: t["count"] for t in (left_payload.get("topics", {}).get("keyphrases") or [])}
    rt = {t["phrase"]: t["count"] for t in (right_payload.get("topics", {}).get("keyphrases") or [])}
    lpt, rpt = set(lt.keys()), set(rt.keys())
    overlap_topics = []
    for p in sorted(lpt & rpt):
        overlap_topics.append({"phrase": p, "left": lt[p], "right": rt[p]})
    distinct_left_topics = [{"phrase": p, "count": lt[p]} for p in sorted(lpt - rpt)]
    distinct_right_topics = [{"phrase": p, "count": rt[p]} for p in sorted(rpt - lpt)]

    return {
        "sources": {
            "overlap": overlap_sources,
            "distinct_left": distinct_left_sources,
            "distinct_right": distinct_right_sources
        },
        "topics": {
            "overlap": overlap_topics,
            "distinct_left": distinct_left_topics,
            "distinct_right": distinct_right_topics
        }
    }

def build_insights_payload(query1, query2, articles1, articles2, analysis1, analysis2):
    left = aggregate_query_payload(query1, articles1, analysis1)

    def sov_series(left_dc, right_dc):
        # merge by date
        lmap = {x["date"]: x["count"] for x in (left_dc or [])}
        rmap = {x["date"]: x["count"] for x in (right_dc or [])}
        dates = sorted(set(list(lmap.keys()) + list(rmap.keys())))
        out = []
        for d in dates:
            l = lmap.get(d, 0)
            r = rmap.get(d, 0)
            tot = l + r
            out.append({"date": d, "left": l, "right": r, "sov": (l / tot) if tot else 0})
        return out

    def tier1_big_stories(payload):
        # pick tier1 domains with >=2 articles; list representative headlines
        res = []
        groups = {}
        for h in (payload.get("headlines", {}).get("representative") or []):
            dom = _domain(h.get("url") or "")
            if dom in TIER1_DOMAINS:
                groups.setdefault(dom, []).append(h)
        for dom, items in groups.items():
            if len(items) >= 2:
                res.append({
                    "domain": dom,
                    "dates": sorted({i.get("date") for i in items if i.get("date")}),
                    "headlines": items[:3],
                    "count": len(items)
                })
        # sort by count desc
        res.sort(key=lambda x: x.get("count", 0), reverse=True)
        return res[:5]

    def rising_topics(payload):
        # Compare first half vs second half on editorial-only daily articles
        # Use representative headlines dates to approximate halves
        reps = payload.get("headlines", {}).get("representative") or []
        if not reps:
            return []
        # Get dates present
        ds = sorted({r.get("date") for r in reps if r.get("date")})
        if len(ds) < 4:
            return []
        mid = ds[len(ds)//2]
        first = [a for a in articles1 if (a.get("publishedAt") or "").startswith(tuple([d for d in ds if d <= mid]))]
        second = [a for a in articles1 if (a.get("publishedAt") or "").startswith(tuple([d for d in ds if d > mid]))]
        # Fallback if empty
        if not first or not second:
            return []
        f_top = {t["topic"]: t["count"] for t in extract_topics(first, query1)}
        s_top = {t["topic"]: t["count"] for t in extract_topics(second, query1)}
        deltas = []
        for k, v in s_top.items():
            prev = f_top.get(k, 0)
            dv = v - prev
            if dv > 0:
                deltas.append({"phrase": k, "delta": dv})
        deltas.sort(key=lambda x: x["delta"], reverse=True)
        return deltas[:10]

    if query2:
        right = aggregate_query_payload(query2, articles2, analysis2 or {})
        ov = compute_overlap(left, right)
        total_left = left["summary"]["total_articles"]
        total_right = right["summary"]["total_articles"]
        denom = max(1, total_left + total_right)
        payload = {
            "scenario": "comparative",
            "left_label": query1,
            "right_label": query2,
            "left": left,
            "right": right,
            "sov": {"left": total_left/denom, "right": total_right/denom},
            "sov_by_day": sov_series((left.get("volume") or {}).get("daily_counts"), (right.get("volume") or {}).get("daily_counts")),
            "overlap": ov,
            "tier1_big_stories": {
                "left": tier1_big_stories(left),
                "right": tier1_big_stories(right)
            },
            "trends": {
                "rising_topics_left": rising_topics(left),
                "rising_topics_right": rising_topics(right)
            }
        }
    else:
        payload = {
            "scenario": "company",
            "label": query1,
            "single": left,
            "tier1_big_stories": {"single": tier1_big_stories(left)},
            "trends": {"rising_topics": rising_topics(left)}
        }
    return payload.get("scenario"), payload

def openai_insights_json(payload: dict):
    """
    Ask OpenAI to return a JSON insights object. No narrative sprawl.
    """
    if not ('openai_client' in globals() and openai_client):
        return None
    system = (
        "You are a communications analyst. Produce STRICTLY factual insights based ONLY on the provided JSON. "
        "No sentiment, no subjective language. Output MUST be a JSON object. "
        "Cover, when present: summary (totals, SoV), outlet mix (editorial share, tier1 share, top editorial sources), "
        "top narratives/topics, timeline highlights (peaks), SoV by day for comparative, tier1 big stories (domain, dates, headlines), "
        "and rising topics (with +delta counts). If a field cannot be substantiated, omit it."
    )
    user = (
        "Payload JSON for analysis:\n" + json.dumps(payload, default=str)
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.1,
            max_tokens=900
        )
        txt = (resp.choices[0].message.content or "").strip()
        return json.loads(txt)
    except Exception as e:
        print("openai_insights_json error:", e)
        return None

def render_insights_html(ins: dict) -> str:
    """
    Convert insights JSON into conservative HTML sections.
    """
    if not isinstance(ins, dict):
        return ""
    h = []
    scen = ins.get("scenario") or ""
    def esc(s): return html.escape(str(s)) if s is not None else ""

    # Coverage Summary
    h.append("<h3>Coverage Summary</h3>")
    if scen == "comparative":
        left_label = ins.get("left_label") or "Left"
        right_label = ins.get("right_label") or "Right"
        left = ins.get("left") or {}
        right = ins.get("right") or {}
        sov = ins.get("sov") or {}
        h.append("<ul>")
        h.append(f"<li>{esc(left_label)}: {int((left.get('summary') or {}).get('total_articles') or 0)} articles</li>")
        h.append(f"<li>{esc(right_label)}: {int((right.get('summary') or {}).get('total_articles') or 0)} articles</li>")
        if "left" in sov and "right" in sov:
            h.append(f"<li>Share of Voice: {round((sov.get('left') or 0)*100)}% vs {round((sov.get('right') or 0)*100)}%</li>")
        h.append("</ul>")
    else:
        single = ins.get("single") or {}
        ssum = single.get("summary") or {}
        dr = ssum.get("date_range") or {}
        h.append("<ul>")
        h.append(f"<li>Total articles: {int(ssum.get('total_articles') or 0)}</li>")
        if dr.get("start") or dr.get("end"):
            h.append(f"<li>Date range: {esc(dr.get('start'))} to {esc(dr.get('end'))}</li>")
        h.append("</ul>")

    # Top Sources
    def render_sources(srcs):
        out = []
        out.append("<ul>")
        for s in (srcs or [])[:10]:
            name = esc(s.get("name"))
            cnt = int(s.get("count") or 0)
            tier = esc(s.get("tier") or "")
            badge = f" <span style='font-size:11px;color:#666'>({tier})</span>" if tier == "tier1" else ""
            out.append(f"<li>{name}: {cnt}{badge}</li>")
        out.append("</ul>")
        return "".join(out)

    h.append("<h3>Top Sources</h3>")
    if scen == "comparative":
        left = ins.get("left") or {}
        right = ins.get("right") or {}
        h.append("<div style='display:flex;gap:24px;flex-wrap:wrap'>")
        h.append("<div><strong>Left</strong>" + render_sources((left.get("outlets") or {}).get("top_sources")) + "</div>")
        h.append("<div><strong>Right</strong>" + render_sources((right.get("outlets") or {}).get("top_sources")) + "</div>")
        h.append("</div>")
    else:
        single = ins.get("single") or {}
        h.append(render_sources((single.get("outlets") or {}).get("top_sources")))

    # Top Topics
    def render_topics(tps):
        out = []
        out.append("<ul>")
        for t in (tps or [])[:15]:
            out.append(f"<li>{esc(t.get('phrase'))}: {int(t.get('count') or 0)}</li>")
        out.append("</ul>")
        return "".join(out)

    h.append("<h3>Top Topics</h3>")
    if scen == "comparative":
        left = ins.get("left") or {}
        right = ins.get("right") or {}
        h.append("<div style='display:flex;gap:24px;flex-wrap:wrap'>")
        h.append("<div><strong>Left</strong>" + render_topics(((left.get("topics") or {}).get("keyphrases"))) + "</div>")
        h.append("<div><strong>Right</strong>" + render_topics(((right.get("topics") or {}).get("keyphrases"))) + "</div>")
        h.append("</div>")
    else:
        single = ins.get("single") or {}
        h.append(render_topics((single.get("topics") or {}).get("keyphrases")))

    # Tier1 Big Stories (optional)
    if "tier1_big_stories" in ins:
        h.append("<h3>Tier‑1 Big Stories</h3>")
        def render_bigstories(bs):
            out = ["<ul>"]
            for item in (bs or [])[:5]:
                dom = esc(item.get("domain"))
                cnt = int(item.get("count") or 0)
                dates = ", ".join([esc(d) for d in (item.get("dates") or [])[:3]])
                out.append(f"<li><strong>{dom}</strong> — {cnt} stories • {dates}")
                heads = item.get("headlines") or []
                if heads:
                    out.append("<ul>")
                    for hli in heads[:3]:
                        title = esc(hli.get("title"))
                        url = esc(hli.get("url"))
                        date = esc(hli.get("date"))
                        out.append(f"<li><a href=\"{url}\" target=\"_blank\">{title}</a> • {date}</li>")
                    out.append("</ul>")
                out.append("</li>")
            out.append("</ul>")
            return "".join(out)

        if ins.get("scenario") == "comparative":
            left_bs = (ins.get("tier1_big_stories") or {}).get("left")
            right_bs = (ins.get("tier1_big_stories") or {}).get("right")
            h.append("<div style='display:flex;gap:24px;flex-wrap:wrap'>")
            h.append("<div><strong>Left</strong>" + render_bigstories(left_bs) + "</div>")
            h.append("<div><strong>Right</strong>" + render_bigstories(right_bs) + "</div>")
            h.append("</div>")
        else:
            single_bs = (ins.get("tier1_big_stories") or {}).get("single")
            h.append(render_bigstories(single_bs))

    # Timeline Highlights
    h.append("<h3>Timeline Highlights</h3>")
    if scen == "comparative":
        left = ins.get("left") or {}
        right = ins.get("right") or {}
        def peak_list(peaks):
            out = []
            out.append("<ul>")
            for p in (peaks or [])[:5]:
                out.append(f"<li>{esc(p.get('date'))}: {int(p.get('count') or 0)} — {esc(p.get('headline'))} ({esc(p.get('source'))})</li>")
            out.append("</ul>")
            return "".join(out)
        h.append("<div style='display:flex;gap:24px;flex-wrap:wrap'>")
        h.append("<div><strong>Left</strong>" + peak_list(((left.get("volume") or {}).get("peaks"))) + "</div>")
        h.append("<div><strong>Right</strong>" + peak_list(((right.get("volume") or {}).get("peaks"))) + "</div>")
        h.append("</div>")
    else:
        single = ins.get("single") or {}
        peaks = ((single.get("volume") or {}).get("peaks"))
        h.append("<ul>")
        for p in (peaks or [])[:5]:
            h.append(f"<li>{esc(p.get('date'))}: {int(p.get('count') or 0)} — {esc(p.get('headline'))} ({esc(p.get('source'))})</li>")
        h.append("</ul>")

    # Trends (optional)
    if "trends" in ins:
        tr = ins.get("trends") or {}
        h.append("<h3>Trends</h3>")
        h.append("<ul>")
        if ins.get("scenario") == "comparative":
            lt = tr.get("rising_topics_left") or []
            rt = tr.get("rising_topics_right") or []
            if lt:
                h.append("<li><strong>Rising topics (Left):</strong> " + ", ".join([f"{esc(x.get('phrase'))} (+{int(x.get('delta') or 0)})" for x in lt[:8]]) + "</li>")
            if rt:
                h.append("<li><strong>Rising topics (Right):</strong> " + ", ".join([f"{esc(x.get('phrase'))} (+{int(x.get('delta') or 0)})" for x in rt[:8]]) + "</li>")
        else:
            r = tr.get("rising_topics") or []
            if r:
                h.append("<li><strong>Rising topics:</strong> " + ", ".join([f"{esc(x.get('phrase'))} (+{int(x.get('delta') or 0)})" for x in r[:10]]) + "</li>")
        h.append("</ul>")

    # Representative Headlines
    h.append("<h3>Representative Headlines</h3>")
    reps = []
    if scen == "comparative":
        # combine a few from each side
        left = ins.get("left") or {}
        right = ins.get("right") or {}
        reps = ((left.get("headlines") or {}).get("representative") or [])[:6] + ((right.get("headlines") or {}).get("representative") or [])[:6]
    else:
        single = ins.get("single") or {}
        reps = ((single.get("headlines") or {}).get("representative") or [])[:10]
    h.append("<ul>")
    for r in reps:
        title = esc(r.get("title"))
        src = esc(r.get("source"))
        date = esc(r.get("date"))
        url = esc(r.get("url"))
        h.append(f"<li><a href=\"{url}\" target=\"_blank\">{title}</a> — {src} • {date}</li>")
    h.append("</ul>")

    return "".join(h)


def extract_topics(articles, query):
    """
    Extract top topics with a preference for multi-word keyphrases (bigrams/trigrams).
    - No sentiment. Purely frequency-based.
    - Filters common/low-signal tokens.
    Returns: list[{"topic": str, "count": int}], top 30.
    """
    from collections import Counter
    import re as _re

    # 1) Build corpus of titles + descriptions
    docs = []
    for a in (articles or []):
        t = (a.get("title") or "")
        d = (a.get("description") or "")
        docs.append((t + " " + d).lower())

    # 2) Tokenize
    token_re = _re.compile(r"\b[a-zA-Z][a-zA-Z0-9\-]+\b")
    tokens_list = [token_re.findall(doc) for doc in docs]

    # 3) Stop words (extended)
    base_sw = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','up',
        'about','into','over','after','is','are','was','were','be','been','being','have','has',
        'had','do','does','did','will','would','should','can','could','may','might','must','it',
        'its','this','that','these','those','as','than','their','there','here','you','your'
    }
    # Generic low-signal words frequently appearing in headlines
    low_signal = {
        'new','news','latest','update','updates','today','report','reports','breaking','live',
        'video','watch','reveals','announces','launch','launches','says','said','saying',
        'how','what','why','when','where','who',
        'model','models','data','system','systems','policy','policies','rules','rule','plan','plans',
        'tech','technology','digital','online','platform',
        'industry','market','company','business','press',
        'year','years','month','months','week','weeks','day','days',
        'guide','review','analysis','opinion'
    }
    # Domain-specific low-signal; avoids skew like "liquid", "edge", etc.
    domain_low = {
        'ai','artificial','intelligence','edge','cloud','cooling','cooled','coolers','center','centers',
        'efficient','efficiency','trend','trends','leap','apollo','blackwell',
        'liquid','air','deployments',
        # Filter PR distributors and related tokens from becoming topics
        'globenewswire','businesswire','prnewswire','newswire','press','release','pressrelease'
    }
    stop_words = base_sw | low_signal | domain_low

    # Include search terms to avoid echoing the query itself as a "topic"
    try:
        for tok in (query or "").lower().split():
            if tok:
                stop_words.add(tok)
    except Exception:
        pass

    def cleaned(seq):
        out = []
        for w in seq:
            wl = w.lower()
            if wl in stop_words: 
                continue
            if len(wl) <= 2:
                continue
            # filter tokens that are mostly digits or hyphens with little signal
            if sum(ch.isalpha() for ch in wl) < 2:
                continue
            out.append(wl)
        return out

    cleaned_tokens = [cleaned(toks) for toks in tokens_list]

    # 4) Count bigrams/trigrams
    bigrams = Counter()
    trigrams = Counter()
    for toks in cleaned_tokens:
        for i in range(len(toks) - 1):
            bigrams[f"{toks[i]} {toks[i+1]}"] += 1
        for i in range(len(toks) - 2):
            trigrams[f"{toks[i]} {toks[i+1]} {toks[i+2]}"] += 1

    # Keep phrases that appear at least twice
    phrase_counts = Counter()
    for k, v in trigrams.items():
        if v >= 2:
            phrase_counts[k] += v
    for k, v in bigrams.items():
        if v >= 2:
            # Only add if not overshadowed by a stronger trigram that contains it
            phrase_counts[k] += v

    # 5) If not enough phrases, fallback to unigrams
    unigrams = Counter()
    if len(phrase_counts) < 10:
        for toks in cleaned_tokens:
            unigrams.update(toks)

    # 6) Merge and select top
    combined = phrase_counts.copy()
    # Add top unigrams (down-weight a bit so phrases are preferred)
    for w, c in unigrams.most_common(50):
        combined[w] += max(1, c // 2)

    top = [{"topic": k, "count": int(v)} for k, v in combined.most_common(30) if v > 1]
    return top

def generate_openai_summary(analysis1, analysis2, articles1, articles2, query1, query2):
    """
    Build a concise, factual HTML summary of coverage using OpenAI.
    Rules:
      - NO sentiment/tone/opinion words; strictly quantitative and descriptive.
      - Focus on counts, outlets, topics, date ranges, and peak days.
      - Output valid, minimal HTML (<h3>, <ul><li>, <p>), no external CSS/JS.
    """
    try:
        if not ('openai_client' in globals() and openai_client):
            return None

        # Reduce payload size: keep only top parts
        def trim_analysis(a):
            if not isinstance(a, dict):
                return {}
            out = {
                "total_articles": a.get("total_articles"),
                "date_range": a.get("date_range"),
                "timeline": (a.get("timeline") or [])[:20],
                "sources": (a.get("sources") or [])[:15],
                "topics": (a.get("topics") or [])[:20],
            }
            return out

        # Pre-compute peaks and sample headlines for higher precision output
        def peaks(a):
            try:
                tl = (a or {}).get("timeline") or []
                top = sorted(tl, key=lambda x: x.get("count", 0), reverse=True)[:5]
                out = []
                for t in top:
                    pa = t.get("peak_article") or {}
                    out.append({
                        "date": t.get("date"),
                        "count": t.get("count"),
                        "peak_title": pa.get("title"),
                        "peak_source": pa.get("source")
                    })
                return out
            except Exception:
                return []

        def sample_headlines(arr, n=12):
            out = []
            for a in (arr or [])[:n]:
                out.append({
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "date": (a.get("publishedAt") or "").split("T")[0]
                })
            return out

        payload = {
            "query1": query1,
            "query2": query2,
            "analysis1": trim_analysis(analysis1),
            "analysis2": trim_analysis(analysis2) if analysis2 else None,
            "peaks1": peaks(analysis1),
            "peaks2": peaks(analysis2) if analysis2 else None,
            "headlines1": sample_headlines(articles1, 12),
            "headlines2": sample_headlines(articles2, 12) if articles2 else None
        }

        system = (
            "You are a media coverage analyst. Produce a concise, factual HTML summary of the provided coverage data. "
            "STRICTLY AVOID sentiment, tone, subjective adjectives, or speculation. "
            "Only use facts present in the JSON. Do not invent sources, numbers, or dates. "
            "Return raw HTML only (no markdown/code fences). "
            "Sections to include when applicable: "
            "<h3>Coverage Summary</h3>, <h3>Top Sources</h3>, <h3>Top Topics</h3>, <h3>Timeline Highlights</h3>, <h3>Representative Headlines</h3>. "
            "Guidelines for Top Topics: Prefer multi‑word keyphrases and named entities. Avoid generic/filler terms "
            "like: new, launches, that, model, data, liquid, air, cooling, coolers, policy, rules. "
            "For each bullet, include counts when provided. "
            "If both query1 and query2 are present, add a compact comparison in Coverage Summary (totals, overlapping/distinct sources, distinct topics)."
        )
        user = (
            "Here is the JSON with computed analysis. Summarize factually:\n"
            + json.dumps(payload, default=str)
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip potential markdown code fences the model may include
        try:
            text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"```$", "", text).strip()
        except Exception:
            pass
        if "<" not in text:
            # Ensure HTML if model returned plain text
            text = "<div><pre>" + html.escape(text) + "</pre></div>"
        return text
    except Exception as e:
        print("OpenAI summary generation error:", e)
        return None


# File upload utility functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# File upload routes
@app.route("/upload", methods=["GET", "POST"])
def upload_files():
    """Handle file uploads and process media coverage data."""
    if request.method == "POST":
        # Check if files were uploaded
        if 'files' not in request.files:
            flash('No files selected')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            flash('No files selected')
            return redirect(request.url)
        
        # Process uploaded files
        all_articles = []
        processed_files = []
        
        for file in files:
            if file and file.filename != '' and allowed_file(file.filename):
                try:
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    
                    # Save the file
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Process the file
                    articles = file_processor.process_file(file_path, file.filename)
                    
                    if articles:
                        all_articles.extend(articles)
                        processed_files.append({
                            'filename': file.filename,
                            'articles_count': len(articles)
                        })
                        print(f"Processed {file.filename}: {len(articles)} articles extracted")
                    else:
                        flash(f"No data could be extracted from {file.filename}")
                    
                    # Clean up the uploaded file
                    try:
                        os.remove(file_path)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"Error processing file {file.filename}: {str(e)}")
                    flash(f"Error processing {file.filename}: {str(e)}")
            else:
                flash(f"File type not allowed: {file.filename}")
        
        if not all_articles:
            flash("No articles could be extracted from the uploaded files")
            return redirect(request.url)
        
        # Analyze the extracted articles
        try:
            # Use a generic query for file-based analysis
            query = "Local File Analysis"
            analysis = analyze_articles(all_articles, query)
            
            # Generate analysis text
            def summarize_articles(articles):
                return [{
                    'title': article['title'],
                    'description': article['description'],
                    'publishedAt': article['publishedAt']
                } for article in articles]
            
            summarized_articles = summarize_articles(all_articles)
            
            # Get analysis from Claude
            analysis_prompt = f"""Analyze this media coverage data extracted from uploaded files.

Key points to address:
1. Major Coverage Themes: Identify the main themes, tones, and focus areas in the coverage
2. Key Trends: Analyze patterns in coverage volume, sentiment evolution, and source diversity
3. Business Implications: Discuss market perception, competitive positioning, and strategic opportunities

Articles: {json.dumps(summarized_articles[:50])}"""
            
            response = anthropic.messages.create(
                model=CLAUDE_HAIKU,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": analysis_prompt
                }]
            )
            
            # Simple formatting for the response
            analysis_text = response.content[0].text.replace('\n\n', '</p><p>')
            analysis_text = '<p>' + analysis_text + '</p>'
            analysis_text = Markup(analysis_text)
            
            # Create form data for template compatibility
            form_data = {
                'analysis_type': 'file_upload',
                'processed_files': processed_files
            }
            
            # Persist shareable result with short slug
            payload = {
                "query1": "File Upload Analysis",
                "query2": None,
                "enhanced_query1": {"enhanced_query": "File Upload Analysis", "entity_type": "file_analysis", "reasoning": "Analysis of uploaded files"},
                "enhanced_query2": None,
                "textual_analysis": str(analysis_text),
                "analysis1": analysis,
                "analysis2": None,
                "articles1": all_articles,
                "articles2": [],
                "form_data": form_data
            }
            slug = uuid.uuid4().hex[:10]
            try:
                rec = SharedResult(slug=slug, payload=json.dumps(payload, default=str))
                db.session.add(rec)
                db.session.commit()
            except Exception as e:
                print(f"Error saving media upload share result to DB: {e}")
            share_url = (request.url_root.rstrip('/') + f"/results/{slug}")
            return render_template(
                "result.html",
                query1=payload["query1"],
                query2=None,
                enhanced_query1=payload["enhanced_query1"],
                enhanced_query2=None,
                textual_analysis=analysis_text,
                analysis1=analysis,
                analysis2=None,
                articles1=all_articles,
                articles2=[],
                request=type('obj', (object,), {'form': form_data}),
                ga_measurement_id=GA_MEASUREMENT_ID,
                share_url=share_url,
                slug=slug
            )
            
        except Exception as e:
            print(f"Error analyzing articles: {str(e)}")
            flash(f"Error analyzing data: {str(e)}")
            return redirect(request.url)
    
    return render_template("upload.html", ga_measurement_id=GA_MEASUREMENT_ID)

@app.route("/")
def root_redirect():
    # Preserve the query string so LinkedIn attribution survives: a visitor landing
    # on signal.innatec3.com/?utm_source=linkedin must arrive at
    # /citation-audit?utm_source=linkedin, where the GET handler stashes the UTM.
    qs = request.query_string.decode('utf-8', 'ignore')
    target = "/citation-audit" + (("?" + qs) if qs else "")
    return redirect(target, code=308)


@app.route("/legacy-index", methods=["GET", "POST"])
def index():
    # Allow POST from the search form to avoid 405 Method Not Allowed
    if request.method == "POST":
        query1 = (request.form.get("query1") or "").strip()
        query2 = (request.form.get("query2") or "").strip() or None
        from_date1 = request.form.get("from_date1")
        to_date1 = request.form.get("to_date1")
        from_date2 = request.form.get("from_date2")
        to_date2 = request.form.get("to_date2")

        # Convenience: if user types "brandA vs brandB" in a single field, split into two queries
        if not query2 and re.search(r"\bvs\.?\b", query1, flags=re.IGNORECASE):
            parts = re.split(r"\bvs\.?\b", query1, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                query1, query2 = parts[0], parts[1]

        if not query1:
            flash("Please enter at least one search term")
            return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID, sectors=globals().get("SECTORS"))

        # If live news API isn't configured, try RSS fallback for a useful demo result
        if not NEWS_API_KEY:
            articles1 = fetch_rss_articles(query1, from_date1, to_date1, max_items=60)
            articles2 = fetch_rss_articles(query2, from_date2, to_date2, max_items=60) if query2 else []

            if articles1 or articles2:
                try:
                    analysis1 = analyze_articles(articles1, query1)
                    analysis2 = analyze_articles(articles2, query2) if query2 else None
                    info_html = Markup(
                        "<p><strong>Note:</strong> Using RSS fallback (no NEWS_API_KEY set). "
                        "Results are for demonstration and may be limited compared to premium sources.</p>"
                    )
                    # Persist sharable result with short slug
                    form_data = {
                        'language1': request.form.get("language1"),
                        'language2': request.form.get("language2"),
                        'source1': request.form.get("source1"),
                        'source2': request.form.get("source2"),
                        'from_date1': from_date1, 'to_date1': to_date1,
                        'from_date2': from_date2, 'to_date2': to_date2
                    }
                    payload = {
                        "query1": query1, "query2": query2,
                        "enhanced_query1": {"enhanced_query": query1, "entity_type": "brand", "reasoning": "RSS fallback"},
                        "enhanced_query2": ({"enhanced_query": query2, "entity_type": "brand", "reasoning": "RSS fallback"} if query2 else None),
                        "textual_analysis": str(info_html),
                        "analysis1": analysis1, "analysis2": analysis2,
                        "articles1": articles1, "articles2": articles2,
                        "form_data": form_data
                    }
                    slug = uuid.uuid4().hex[:10]
                    try:
                        rec = SharedResult(slug=slug, payload=json.dumps(payload, default=str))
                        db.session.add(rec)
                        db.session.commit()
                    except Exception as e:
                        print(f"Error saving media share result to DB: {e}")
                    share_url = (request.url_root.rstrip('/') + f"/results/{slug}")
                    return redirect(share_url)
                except Exception as e:
                    print(f"Error analyzing RSS fallback articles: {e}")

            # If RSS also found nothing, render a graceful guidance message
            info_html = Markup(
                "<p><strong>Live news search is not configured.</strong> "
                "Please add a NEWS_API_KEY to enable fetching coverage, or use the "
                "<a href='/upload' style='text-decoration: underline;'>Upload Files</a> "
                "flow to analyze your media spreadsheets/PDFs.</p>"
            )
            analysis1 = {
                "timeline": [],
                "sources": [],
                "topics": [],
                "total_articles": 0,
                "date_range": {"start": from_date1, "end": to_date1},
                "avg_sentiment": 0,
            }
            analysis2 = None
            if query2:
                analysis2 = {
                    "timeline": [],
                    "sources": [],
                    "topics": [],
                    "total_articles": 0,
                    "date_range": {"start": from_date2, "end": to_date2},
                    "avg_sentiment": 0,
                }

            return render_template(
                "result.html",
                query1=query1,
                query2=query2,
                enhanced_query1={"enhanced_query": query1, "entity_type": "brand", "reasoning": "No live API configured"},
                enhanced_query2=({"enhanced_query": query2, "entity_type": "brand", "reasoning": "No live API configured"} if query2 else None),
                textual_analysis=info_html,
                analysis1=analysis1,
                analysis2=analysis2,
                articles1=[],
                articles2=[],
                ga_measurement_id=GA_MEASUREMENT_ID,
            )

        # If NEWS_API_KEY is present, fetch live coverage via NewsAPI; fallback to RSS if needed
        language1 = (request.form.get("language1") or "en").strip()
        language2 = (request.form.get("language2") or "en").strip() if query2 else None
        sources1 = (request.form.get("source1") or "").strip() or None
        sources2 = (request.form.get("source2") or "").strip() if query2 else None
        # New filters
        provider1 = (request.form.get("provider1") or "").strip()
        provider2 = (request.form.get("provider2") or "").strip() if query2 else None
        sector1 = (request.form.get("sector1") or "").strip()
        sector2 = (request.form.get("sector2") or "").strip() if query2 else None

        try:
            # Conditionally use NewsAPI depending on provider selection and date span
            def _days_span(f, t):
                try:
                    if not f or not t:
                        return None
                    fdt = datetime.fromisoformat(f).date()
                    tdt = datetime.fromisoformat(t).date()
                    return abs((tdt - fdt).days) + 1
                except Exception:
                    return None

            span1 = _days_span(from_date1, to_date1)
            span2 = _days_span(from_date2, to_date2) if query2 else None

            def _should_use_newsapi(provider, span):
                # Always use NewsAPI if explicitly selected
                if provider == "newsapi":
                    return True
                # Never use NewsAPI if another explicit provider is selected
                if provider in ("nyt", "guardian", "rss"):
                    return False
                # Provider unspecified (All): only use NewsAPI for <= 31 days
                return (span is None) or (span <= 31)

            na1 = []
            if NEWS_API_KEY and _should_use_newsapi(provider1, span1):
                na1 = fetch_news_api_articles_sliced(query1, from_date1, to_date1, language=language1, sources=sources1, per_slice=8, slice_days=1)

            na2 = []
            if query2 and NEWS_API_KEY and _should_use_newsapi(provider2, span2):
                na2 = fetch_news_api_articles_sliced(query2, from_date2, to_date2, language=language2, sources=sources2, per_slice=8, slice_days=1)
        except Exception as e:
            print(f"NewsAPI error: {e}")
            na1, na2 = [], []

        # Always supplement with Google News RSS and de-duplicate
        rss1 = fetch_rss_articles(query1, from_date1, to_date1, max_items=60)
        rss2 = fetch_rss_articles(query2, from_date2, to_date2, max_items=60) if query2 else []

        # Add NYT and Guardian as additional editorial backbones
        nyt1 = fetch_nyt_articles(query1, from_date1, to_date1, max_items=60) if NYT_API_KEY else []
        nyt2 = fetch_nyt_articles(query2, from_date2, to_date2, max_items=60) if (query2 and NYT_API_KEY) else []
        gu1 = fetch_guardian_articles(query1, from_date1, to_date1, max_items=60) if GUARDIAN_API_KEY else []
        gu2 = fetch_guardian_articles(query2, from_date2, to_date2, max_items=60) if (query2 and GUARDIAN_API_KEY) else []

        # Build FULL combined sets for accurate analysis across the entire date range,
        # then trim for UI rendering so the page stays lightweight.
        merged1 = merge_articles_dedupe(na1, nyt1, max_total=1000)
        merged1 = merge_articles_dedupe(merged1, gu1, max_total=1000)
        full1 = sort_articles_desc(merge_articles_dedupe(merged1, rss1, max_total=1000))

        if query2:
            merged2 = merge_articles_dedupe(na2, nyt2, max_total=1000)
            merged2 = merge_articles_dedupe(merged2, gu2, max_total=1000)
            full2 = sort_articles_desc(merge_articles_dedupe(merged2, rss2, max_total=1000))
        else:
            full2 = []

        # Provider gating: if a specific provider is selected, use only that source
        try:
            if provider1:
                if provider1 == "nyt":
                    full1 = sort_articles_desc(nyt1)
                elif provider1 == "guardian":
                    full1 = sort_articles_desc(gu1)
                elif provider1 == "newsapi":
                    full1 = sort_articles_desc(na1)
                elif provider1 == "rss":
                    full1 = sort_articles_desc(rss1)
            if query2 and provider2:
                if provider2 == "nyt":
                    full2 = sort_articles_desc(nyt2)
                elif provider2 == "guardian":
                    full2 = sort_articles_desc(gu2)
                elif provider2 == "newsapi":
                    full2 = sort_articles_desc(na2)
                elif provider2 == "rss":
                    full2 = sort_articles_desc(rss2)
        except Exception:
            # If anything goes wrong, retain the default merged/full sets
            pass

        # Sector filtering (domain or keyword-based) before analysis
        try:
            if sector1:
                full1 = filter_by_sector(full1, sector1)
            if query2 and sector2:
                full2 = filter_by_sector(full2, sector2)
        except Exception:
            pass

        # UI cards (representative sample, still fairly rich)
        articles1 = full1[:120]
        articles2 = (full2[:120] if query2 else [])

        # Fallback path: if no results for the requested window, try recent coverage instead
        if not full1 and (not query2 or not full2):
            try:
                def _recent_30():
                    today = datetime.utcnow().date()
                    start = (today - timedelta(days=29)).isoformat()
                    end = today.isoformat()
                    return start, end

                ffrom1, fto1 = _recent_30()
                fb1 = []

                # If provider is explicitly pinned, try that first
                if provider1 == "nyt" and NYT_API_KEY:
                    fb1 = fetch_nyt_articles(query1, ffrom1, fto1, max_items=60)
                elif provider1 == "guardian" and GUARDIAN_API_KEY:
                    fb1 = fetch_guardian_articles(query1, ffrom1, fto1, max_items=60)
                elif provider1 == "newsapi" and NEWS_API_KEY:
                    fb1 = fetch_news_api_articles_sliced(query1, ffrom1, fto1, language=language1, sources=sources1, per_slice=8, slice_days=1)
                elif provider1 == "rss":
                    fb1 = fetch_rss_articles(query1, ffrom1, fto1, max_items=60)

                # If provider not pinned (All), broaden to other providers; otherwise stay restricted
                if not provider1:
                    if not fb1 and NEWS_API_KEY:
                        fb1 = fetch_news_api_articles_sliced(query1, ffrom1, fto1, language=language1, sources=sources1, per_slice=8, slice_days=1)
                    if not fb1 and NYT_API_KEY:
                        fb1 = fetch_nyt_articles(query1, ffrom1, fto1, max_items=60)
                    if not fb1 and GUARDIAN_API_KEY:
                        fb1 = fetch_guardian_articles(query1, ffrom1, fto1, max_items=60)
                    if not fb1:
                        fb1 = fetch_rss_articles(query1, ffrom1, fto1, max_items=60)

                # Provider-specific fallback for query1 when pinned and still empty
                if not fb1 and provider1 in ("nyt", "guardian", "newsapi"):
                    fb1 = fetch_rss_articles(query1, ffrom1, fto1, max_items=60)
                    fallback_reason1 = f"Requested provider '{provider1}' unavailable or returned no results; showing recent RSS coverage instead."

                fb2 = []
                if query2:
                    ffrom2, fto2 = _recent_30()
                    # Pinned provider first
                    if provider2 == "nyt" and NYT_API_KEY:
                        fb2 = fetch_nyt_articles(query2, ffrom2, fto2, max_items=60)
                    elif provider2 == "guardian" and GUARDIAN_API_KEY:
                        fb2 = fetch_guardian_articles(query2, ffrom2, fto2, max_items=60)
                    elif provider2 == "newsapi" and NEWS_API_KEY:
                        fb2 = fetch_news_api_articles_sliced(query2, ffrom2, fto2, language=language2, sources=sources2, per_slice=8, slice_days=1)
                    elif provider2 == "rss":
                        fb2 = fetch_rss_articles(query2, ffrom2, fto2, max_items=60)

                    # If provider not pinned (All), broaden to other providers; otherwise stay restricted
                    if not provider2:
                        if not fb2 and NEWS_API_KEY:
                            fb2 = fetch_news_api_articles_sliced(query2, ffrom2, fto2, language=language2, sources=sources2, per_slice=8, slice_days=1)
                        if not fb2 and NYT_API_KEY:
                            fb2 = fetch_nyt_articles(query2, ffrom2, fto2, max_items=60)
                        if not fb2 and GUARDIAN_API_KEY:
                            fb2 = fetch_guardian_articles(query2, ffrom2, fto2, max_items=60)
                        if not fb2:
                            fb2 = fetch_rss_articles(query2, ffrom2, fto2, max_items=60)

                # Provider-specific fallback for query2 when pinned and still empty
                if query2 and (not fb2) and provider2 in ("nyt", "guardian", "newsapi"):
                    fb2 = fetch_rss_articles(query2, ffrom2, fto2, max_items=60)
                    fallback_reason2 = f"Requested provider '{provider2}' unavailable or returned no results; showing recent RSS coverage instead."

                # Apply sector filter if present
                try:
                    if sector1:
                        fb1 = filter_by_sector(fb1, sector1)
                    if query2 and sector2:
                        fb2 = filter_by_sector(fb2, sector2)
                except Exception:
                    pass

                if fb1 or (query2 and fb2):
                    analysis1 = analyze_articles(sort_articles_desc(fb1), query1)
                    analysis2 = analyze_articles(sort_articles_desc(fb2), query2) if query2 else None

                    # Build info note and render results
                    _note_parts = [
                        "<p><strong>Note:</strong> No results were returned for the selected historical range. "
                        "Showing recent coverage from the last 30 days instead due to provider limitations.</p>"
                    ]
                    try:
                        if 'fallback_reason1' in locals() and fallback_reason1:
                            _note_parts.append("<p>" + html.escape(fallback_reason1) + "</p>")
                    except Exception:
                        pass
                    try:
                        if query2 and 'fallback_reason2' in locals() and fallback_reason2:
                            _note_parts.append("<p>" + html.escape(fallback_reason2) + "</p>")
                    except Exception:
                        pass
                    note = Markup("".join(_note_parts))

                    form_data = {
                        'language1': language1, 'language2': language2,
                        'source1': sources1, 'source2': sources2,
                        'provider1': provider1, 'provider2': provider2,
                        'sector1': sector1, 'sector2': sector2,
                        'from_date1': ffrom1, 'to_date1': fto1,
                        'from_date2': (ffrom2 if query2 else None), 'to_date2': (fto2 if query2 else None)
                    }

                    payload = {
                        "query1": query1, "query2": query2,
                        "enhanced_query1": {"enhanced_query": query1, "entity_type": "brand", "reasoning": "Fallback to recent 30 days"},
                        "enhanced_query2": ({"enhanced_query": query2, "entity_type": "brand", "reasoning": "Fallback to recent 30 days"} if query2 else None),
                        "textual_analysis": str(note),
                        "analysis1": analysis1, "analysis2": analysis2,
                        "articles1": fb1[:120], "articles2": (fb2[:120] if query2 else []),
                        "form_data": form_data
                    }
                    slug = uuid.uuid4().hex[:10]
                    try:
                        rec = SharedResult(slug=slug, payload=json.dumps(payload, default=str))
                        db.session.add(rec)
                        db.session.commit()
                    except Exception as e:
                        print(f"Error saving fallback share result to DB: {e}")
                    share_url = (request.url_root.rstrip('/') + f"/results/{slug}")
                    return redirect(share_url)
            except Exception as e:
                print("Fallback recent-30 error:", e)

            flash("No results found for the selected range and terms. Try broadening the date range or simplifying the query.")
            return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID, sectors=globals().get("SECTORS"))

        # Analyze and render on the FULL sets to ensure timeline and topics reflect the whole period,
        # not just the trimmed subset.
        analysis1 = analyze_articles(full1, query1)
        analysis2 = analyze_articles(full2, query2) if query2 else None

        # Persist sharable result with short slug
        form_data = {
            'language1': language1, 'language2': language2,
            'source1': sources1, 'source2': sources2,
            'provider1': provider1, 'provider2': provider2,
            'sector1': sector1, 'sector2': sector2,
            'from_date1': from_date1, 'to_date1': to_date1,
            'from_date2': from_date2, 'to_date2': to_date2
        }
        # Provide a neutral, quantitative summary so the template uses textual_analysis (no sentiment)
        try:
            a1_total = analysis1.get("total_articles", 0) if isinstance(analysis1, dict) else 0
            a2_total = (analysis2.get("total_articles", 0) if (query2 and isinstance(analysis2, dict)) else None)
            a1_topics = ", ".join([t.get("topic") for t in (analysis1.get("topics") or [])[:3]]) if isinstance(analysis1, dict) else ""
            a2_topics = ", ".join([t.get("topic") for t in ((analysis2.get("topics") or [])[:3] if (query2 and isinstance(analysis2, dict)) else [])])
            summary_html = "<p><strong>Coverage summary:</strong> {q1}: {n1} articles{vs}</p>".format(
                q1=html.escape(query1),
                n1=a1_total,
                vs=(f" vs {html.escape(query2)}: {a2_total} articles" if (query2 and a2_total is not None) else "")
            )
            if a1_topics:
                summary_html += f"<p><strong>Top topics for {html.escape(query1)}:</strong> {html.escape(a1_topics)}</p>"
            if query2 and a2_topics:
                summary_html += f"<p><strong>Top topics for {html.escape(query2)}:</strong> {html.escape(a2_topics)}</p>"
            info_html = Markup(summary_html)
        except Exception:
            info_html = Markup(f"<p><strong>Coverage summary:</strong> {html.escape(query1)}: {len(articles1)} articles"
                               + (f" vs {html.escape(query2)}: {len(articles2)} articles" if query2 else "")
                               + ".</p>")

        # Try JSON insights (strict structure) first, then fallback to HTML summary
        try:
            # Build insights payloads using FULL article sets for accurate outlet/PR stats
            scenario, payload_ins = build_insights_payload(query1, query2, full1, full2, analysis1, analysis2)
            ai_json = openai_insights_json({"scenario": scenario, **payload_ins})
            if isinstance(ai_json, dict):
                info_html = Markup(render_insights_html(ai_json))
        except Exception as e:
            print("OpenAI JSON insights error:", e)

        # Fallback to concise HTML summary via LLM if needed
        try:
            if not info_html:
                ai_html = generate_openai_summary(analysis1, analysis2, articles1, articles2, query1, query2)
                if ai_html:
                    info_html = Markup(ai_html)
        except Exception as e:
            print("OpenAI summary error:", e)

        payload = {
            "query1": query1, "query2": query2,
            "enhanced_query1": {"enhanced_query": query1, "entity_type": "brand", "reasoning": "Live fetch (NewsAPI + RSS supplement)"},
            "enhanced_query2": ({"enhanced_query": query2, "entity_type": "brand", "reasoning": "Live fetch (NewsAPI + RSS supplement)"} if query2 else None),
            "textual_analysis": str(info_html),
            "analysis1": analysis1, "analysis2": analysis2,
            "articles1": articles1, "articles2": (articles2 or []),
            "form_data": form_data
        }
        slug = uuid.uuid4().hex[:10]
        try:
            rec = SharedResult(slug=slug, payload=json.dumps(payload, default=str))
            db.session.add(rec)
            db.session.commit()
        except Exception as e:
            print(f"Error saving media share result to DB: {e}")
        share_url = (request.url_root.rstrip('/') + f"/results/{slug}")
        return redirect(share_url)

    # GET request renders the search form
    return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID, sectors=globals().get("SECTORS"))

@app.route("/results/<slug>")
def view_shared_result(slug):
    """Render a previously saved media analysis by slug."""
    rec = SharedResult.query.filter_by(slug=slug).first()
    if not rec:
        flash("Shared result not found or expired")
        return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID)
    try:
        data = json.loads(rec.payload)
    except Exception:
        flash("Unable to load shared result")
        return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID)
    # Build a fake request.form wrapper for template compatibility
    form_data = data.get("form_data") or {}
    req_proxy = type('obj', (object,), {'form': form_data})
    share_url = (request.url_root.rstrip('/') + f"/results/{slug}")
    # textual_analysis may be plain HTML string
    ta = data.get("textual_analysis")
    ta_markup = Markup(ta) if ta else None

    return render_template(
        "result.html",
        query1=data.get("query1"),
        query2=data.get("query2"),
        enhanced_query1=data.get("enhanced_query1"),
        enhanced_query2=data.get("enhanced_query2"),
        textual_analysis=ta_markup,
        analysis1=data.get("analysis1"),
        analysis2=data.get("analysis2"),
        articles1=data.get("articles1") or [],
        articles2=data.get("articles2") or [],
        request=req_proxy,
        ga_measurement_id=GA_MEASUREMENT_ID,
        share_url=share_url,
        slug=slug
    )

@app.route("/api/email_summary", methods=["POST"])
def email_summary():
    try:
        data = request.get_json(silent=True) or request.form or {}
        email = (data.get("email") or "").strip()
        slug = (data.get("slug") or "").strip()
        if not email or not slug:
            return jsonify({"ok": False, "error": "Missing email or slug"}), 400

        rec = SharedResult.query.filter_by(slug=slug).first()
        if not rec:
            return jsonify({"ok": False, "error": "Result not found"}), 404

        payload = {}
        try:
            payload = json.loads(rec.payload)
        except Exception:
            pass

        query1 = payload.get("query1") or "Analysis"
        query2 = payload.get("query2")
        a1 = payload.get("analysis1") or {}
        a2 = payload.get("analysis2") or {}
        total1 = a1.get("total_articles", 0) or 0
        total2 = a2.get("total_articles", 0) or 0

        share_url = request.url_root.rstrip('/') + f"/results/{slug}"
        summary_lines = [
            f'Media Analysis for "{query1}"' + (f' vs "{query2}"' if query2 else ""),
            "",
            f"Link: {share_url}",
            "",
            "Coverage Metrics:",
            f"- {query1}: {total1} articles",
        ]
        if query2:
            summary_lines.append(f"- {query2}: {total2} articles")

        topics = (a1.get("topics") or [])[:5]
        if topics:
            summary_lines.append("")
            summary_lines.append("Top Topics:")
            summary_lines.append(", ".join([t.get("topic") for t in topics if isinstance(t, dict) and t.get("topic")]))

        text_body = "\n".join(summary_lines)

        # Persist lead capture
        try:
            lead = LeadCapture(email=email, slug=slug, app_name="media_analyzer")
            db.session.add(lead)
            db.session.commit()
        except Exception as e:
            print(f"Lead save error: {e}")

        sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
        if not sg_key:
            return jsonify({"ok": True, "sent": False, "message": "SENDGRID_API_KEY not set; lead captured only"})

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            msg = Mail(
                from_email=("nstrauss@innatec3.com", "innate c3"),
                to_emails=[email],
                subject=f"Media Analysis: {query1}" + (f" vs {query2}" if query2 else ""),
                plain_text_content=text_body,
                html_content="<pre style='font-family:monospace'>" + html.escape(text_body) + "</pre>"
            )
            resp = _send_mail_object(msg)
            print("SendGrid response:", resp.status_code)
            return jsonify({"ok": True, "sent": True})
        except Exception as e:
            print("SendGrid error:", e)
            return jsonify({"ok": True, "sent": False, "message": "Email not sent; lead captured"}), 200
    except Exception as e:
        print("email_summary error:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

@app.route("/api/subscribe", methods=["POST"])
def api_subscribe():
    """
    Create or update a subscription for alerts tied to a results slug.
    Body: { email: str, slug: str, frequency: 'realtime'|'daily' }
    """
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip()
        slug = (data.get("slug") or "").strip()
        frequency = (data.get("frequency") or "").strip().lower()
        if not email or not slug or frequency not in ("realtime", "daily"):
            return jsonify({"ok": False, "error": "Missing/invalid email, slug, or frequency"}), 400

        # Load the referenced result to snapshot params (for future-proofing)
        rec = SharedResult.query.filter_by(slug=slug).first()
        params_json = "{}"
        if rec:
            params_json = rec.payload

        token = uuid.uuid4().hex
        # Upsert by (email, slug, frequency)
        sub = Subscription.query.filter_by(email=email, slug=slug, frequency=frequency).first()
        if sub:
            sub.active = True
            sub.params = params_json
            sub.last_checked_at = datetime.utcnow()
            if not sub.unsubscribe_token:
                sub.unsubscribe_token = token
        else:
            sub = Subscription(
                email=email,
                slug=slug,
                frequency=frequency,
                params=params_json,
                active=True,
                last_checked_at=datetime.utcnow(),
                unsubscribe_token=token
            )
            db.session.add(sub)
        db.session.commit()

        # Simple confirmation payload
        return jsonify({"ok": True, "token": sub.unsubscribe_token})
    except Exception as e:
        print("api_subscribe error:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

@app.route("/unsubscribe")
def unsubscribe():
    """
    Deactivate a subscription using its token.
    """
    try:
        token = (request.args.get("token") or "").strip()
        if not token:
            return "Missing token", 400
        sub = Subscription.query.filter_by(unsubscribe_token=token).first()
        if not sub:
            return "Subscription not found", 404
        sub.active = False
        db.session.commit()
        return "You have been unsubscribed from alerts for this analysis.", 200
    except Exception as e:
        print("unsubscribe error:", e)
        return "Server error", 500

@app.route("/api/lead", methods=["POST"])
def api_lead():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip()
        slug = (data.get("slug") or "").strip()
        action = (data.get("action") or "").strip()
        app_name = (data.get("app_name") or "media_analyzer").strip()
        extra_payload = {"action": action} if action else {}
        try:
            lead = LeadCapture(email=email, slug=slug, app_name=app_name, extra=(json.dumps(extra_payload) if extra_payload else None))
            db.session.add(lead)
            db.session.commit()
        except Exception as e:
            print(f"Lead save error (/api/lead): {e}")
        # Optional webhook forward to Google Sheets/Airtable bridge if configured
        webhook = os.environ.get("LEADS_WEBHOOK_URL")
        if webhook:
            try:
                requests.post(webhook, json={"email": email, "slug": slug, "action": action, "app": app_name}, timeout=5)
            except Exception as e:
                print(f"Webhook post error: {e}")
        return jsonify({"ok": True})
    except Exception as e:
        print("api_lead error:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500

@app.route("/og/<slug>.png")
def og_image(slug):
    try:
        rec = SharedResult.query.filter_by(slug=slug).first()
        data = json.loads(rec.payload) if rec else {}
        query1 = data.get("query1") or "Media Analysis"
        query2 = data.get("query2")
        a1 = data.get("analysis1") or {}
        total = a1.get("total_articles", 0) or 0
        dr = (a1.get("date_range") or {})
        date_start = dr.get("start") or ""
        date_end = dr.get("end") or ""
    except Exception:
        query1 = "Media Analysis"
        query2 = None
        total = 0
        date_start = ""
        date_end = ""

    title = f'{query1} vs {query2}' if query2 else query1

    # Create OG image 1200x630
    W, H = 1200, 630
    bg = (0, 94, 48)  # #005e30
    fg = (255, 255, 255)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default()
    font_med = ImageFont.load_default()
    font_small = ImageFont.load_default()

    # Header
    draw.text((60, 80), "innate c3 | media analysis", fill=fg, font=font_small)
    # Title
    draw.text((60, 130), title[:60], fill=fg, font=font_big)
    # Stats
    draw.text((60, 200), f"Articles: {total}", fill=fg, font=font_med)
    if date_start and date_end:
        draw.text((60, 240), f"{date_start} → {date_end}", fill=fg, font=font_med)
    # Footer
    draw.text((60, 560), "innatec3.com", fill=fg, font=font_small)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/examples")
def examples():
    """Public gallery of recent shared media analyses."""
    # Fetch latest 12 shared results
    try:
        recs = SharedResult.query.order_by(SharedResult.created_at.desc()).limit(12).all()
    except Exception as e:
        print(f"Error loading examples: {e}")
        recs = []

    cards = []
    for rec in recs:
        try:
            data = json.loads(rec.payload)
        except Exception:
            data = {}

        query1 = data.get("query1") or "Analysis"
        query2 = data.get("query2")
        title = f'{query1} vs {query2}' if query2 else query1

        a1 = data.get("analysis1") or {}
        total_articles = a1.get("total_articles", 0) or 0
        dr = (a1.get("date_range") or {})
        date_start = dr.get("start")
        date_end = dr.get("end")
        topics = (a1.get("topics") or [])[:3]
        topics_list = [t.get("topic") for t in topics if isinstance(t, dict) and t.get("topic")]

        share_url = (request.url_root.rstrip('/') + f"/results/{rec.slug}")

        cards.append({
            "slug": rec.slug,
            "title": title,
            "share_url": share_url,
            "total_articles": total_articles,
            "date_start": date_start,
            "date_end": date_end,
            "topics": topics_list,
            "created_at": rec.created_at.isoformat() if rec.created_at else None
        })

    return render_template("examples.html", cards=cards, ga_measurement_id=GA_MEASUREMENT_ID)


@app.route("/api/news", methods=["GET"])
def api_news():
    """
    Headlines endpoint backed by NewsAPI (if configured) with Google News RSS fallback.
    Query params:
      - query (required)
      - from (YYYY-MM-DD, optional)
      - to (YYYY-MM-DD, optional)
      - max (1..100, optional; default 50)
    """
    try:
        q = (request.args.get("query") or "").strip()
        from_date = request.args.get("from")
        to_date = request.args.get("to")
        try:
            max_items = int(request.args.get("max", "50"))
        except Exception:
            max_items = 50
        max_items = max(1, min(100, max_items))

        if not q:
            return jsonify({"ok": False, "error": "Missing query"}), 400

        na = []
        if NEWS_API_KEY:
            try:
                na = fetch_news_api_articles(q, from_date, to_date, language="en", sources=None, page_size=max_items)
            except Exception as e:
                print(f"/api/news NewsAPI error: {e}")
                na = []
        rss = fetch_rss_articles(q, from_date, to_date, max_items=max_items)
        combined = sort_articles_desc(merge_articles_dedupe(na, rss, max_total=max_items))
        return jsonify({"ok": True, "count": len(combined), "articles": combined})
    except Exception as e:
        print("api_news error:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500


@app.route("/api/news_batch", methods=["POST"])
def api_news_batch():
    """
    Batch headlines endpoint.
    Body JSON:
      {
        "queries": ["AI policy","Anthropic"],
        "from": "YYYY-MM-DD",      // optional
        "to": "YYYY-MM-DD",        // optional
        "max": 25                  // optional, 1..100
      }
    """
    try:
        data = request.get_json(silent=True) or {}
        queries = data.get("queries") or []
        from_date = data.get("from")
        to_date = data.get("to")
        try:
            max_items = int(data.get("max", 25))
        except Exception:
            max_items = 25
        max_items = max(1, min(100, max_items))

        out = {}
        for q in queries:
            if not isinstance(q, str):
                continue
            qn = q.strip()
            if not qn:
                continue

            na = []
            if NEWS_API_KEY:
                try:
                    na = fetch_news_api_articles(qn, from_date, to_date, language="en", sources=None, page_size=max_items)
                except Exception as e:
                    print(f"/api/news_batch NewsAPI error for '{qn}': {e}")
                    na = []
            rss = fetch_rss_articles(qn, from_date, to_date, max_items=max_items)
            combined = sort_articles_desc(merge_articles_dedupe(na, rss, max_total=max_items))
            out[qn] = combined

        return jsonify({"ok": True, "results": out})
    except Exception as e:
        print("api_news_batch error:", e)
        return jsonify({"ok": False, "error": "Server error"}), 500


# Background alert scheduler (simple, quota-aware)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None

def _parse_payload_queries(payload_json: str):
    try:
        obj = json.loads(payload_json or "{}")
        q1 = (obj.get("query1") or "").strip()
        q2 = (obj.get("query2") or "").strip() or None
        # ignore original exact dates for alerts; always roll to now-window
        return q1, q2
    except Exception:
        return "", None

def _fetch_windowed(q: str, start_dt: datetime, end_dt: datetime):
    # Use free/added sources for alerts to avoid NewsAPI 429s in background
    from_str = start_dt.strftime("%Y-%m-%d")
    to_str = end_dt.strftime("%Y-%m-%d")
    # Try NYT + Guardian + RSS
    res = []
    try:
        res = merge_articles_dedupe(fetch_nyt_articles(q, from_str, to_str, max_items=60), fetch_guardian_articles(q, from_str, to_str, max_items=60), max_total=200)
        res = merge_articles_dedupe(res, fetch_rss_articles(q, from_str, to_str, max_items=100), max_total=300)
    except Exception as e:
        print("alert fetch error:", e)
    return sort_articles_desc(res)

def _new_since(articles, since_dt: datetime):
    out = []
    for a in articles or []:
        try:
            dt = parse(a.get("publishedAt") or "")
            if not since_dt or dt.replace(tzinfo=None) > since_dt:
                out.append(a)
        except Exception:
            continue
    return out

def _send_alert_email(email: str, slug: str, freq: str, items: list):
    try:
        sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
        if not sg_key:
            print("SENDGRID_API_KEY not set; skipping alert email.")
            return False
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        share_url = request.url_root.rstrip('/') + f"/results/{slug}" if request else f"/results/{slug}"
        lines = [f"New coverage ({freq}) for your analysis:", "", share_url, ""]
        for a in items[:50]:
            date_str = (a.get("publishedAt") or "").split("T")[0]
            lines.append(f"- {a.get('title')} — {a.get('source',{}).get('name')} • {date_str}\n  {a.get('url')}")
        lines.append("")
        lines.append("Unsubscribe: " + (request.url_root.rstrip('/') + f"/unsubscribe?token="))  # token added by caller
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "innate c3"),
            to_emails=[email],
            subject="New coverage alert",
            plain_text_content="\n".join(lines),
            html_content="<pre style='font-family:monospace'>" + html.escape("\n".join(lines)) + "</pre>"
        )
        resp = _send_mail_object(msg)
        print("Alert email status:", resp.status_code)
        return True
    except Exception as e:
        print("send alert email error:", e)
        return False

def run_realtime_alerts():
    try:
        with app.app_context():
            now = datetime.utcnow()
            subs = Subscription.query.filter_by(active=True, frequency="realtime").all()
            for s in subs:
                q1, q2 = _parse_payload_queries(s.params or "{}")
                start = s.last_checked_at or (now - timedelta(minutes=30))
                items = _fetch_windowed(q1, start, now)
                if q2:
                    items += _fetch_windowed(q2, start, now)
                new_items = _new_since(items, s.last_seen_published_at or start)
                if new_items:
                    ok = _send_alert_email(s.email, s.slug, "real-time", new_items)
                    if ok:
                        # Update last_seen to newest item time
                        try:
                            newest = max(parse(a.get("publishedAt") or "") for a in new_items)
                            s.last_seen_published_at = newest.replace(tzinfo=None)
                        except Exception:
                            s.last_seen_published_at = now
                s.last_checked_at = now
            db.session.commit()
    except Exception as e:
        print("run_realtime_alerts error:", e)

def run_daily_alerts():
    try:
        with app.app_context():
            now = datetime.utcnow()
            start = now - timedelta(days=1)
            subs = Subscription.query.filter_by(active=True, frequency="daily").all()
            for s in subs:
                q1, q2 = _parse_payload_queries(s.params or "{}")
                items = _fetch_windowed(q1, start, now)
                if q2:
                    items += _fetch_windowed(q2, start, now)
                new_items = _new_since(items, s.last_seen_published_at or start)
                if new_items:
                    ok = _send_alert_email(s.email, s.slug, "daily", new_items)
                    if ok:
                        try:
                            newest = max(parse(a.get("publishedAt") or "") for a in new_items)
                            s.last_seen_published_at = newest.replace(tzinfo=None)
                        except Exception:
                            s.last_seen_published_at = now
                s.last_checked_at = now
            db.session.commit()
    except Exception as e:
        print("run_daily_alerts error:", e)

def run_outreach_digest():
    """Daily follow-up digest (cron 8:05 UTC). Thin wrapper so the scheduler,
    started just below, can register the job before the full impl is defined
    later in the file (resolved by late binding at run time)."""
    try:
        _run_outreach_digest()
    except Exception as e:
        print("run_outreach_digest error:", e)


def run_daily_traffic_digest():
    """Daily prospect-opens digest (cron 8:10 UTC). Thin wrapper; late-binds to
    the impl defined later in the file."""
    try:
        _run_daily_traffic_digest()
    except Exception as e:
        print("run_daily_traffic_digest error:", e)


def run_daily_inbound_digest():
    """Daily self-serve audit (inbound-lead) digest (cron 8:15 UTC). Thin wrapper;
    late-binds to the impl defined later in the file."""
    try:
        _run_daily_inbound_digest()
    except Exception as e:
        print("run_daily_inbound_digest error:", e)

# Start scheduler (guard against double-start under reloader)
if BackgroundScheduler:
    try:
        if not getattr(app, "_alerts_scheduler_started", False):
            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(run_realtime_alerts, 'interval', minutes=10)
            scheduler.add_job(run_daily_alerts, 'cron', hour=8, minute=0)
            scheduler.add_job(run_outreach_digest, 'cron', hour=8, minute=5)
            scheduler.add_job(run_daily_traffic_digest, 'cron', hour=8, minute=10)
            scheduler.add_job(run_daily_inbound_digest, 'cron', hour=8, minute=15)
            scheduler.start()
            app._alerts_scheduler_started = True
            print("Alert scheduler started.")
    except Exception as e:
        print("Scheduler start error:", e)

# =============================================================================
# PR Signal Finder — citation audit tool
# =============================================================================

CITATION_SUFFIX = """

You MUST include citations and sources in your response.
At the end of your response, add a section titled 'Sources and References:' listing at least 3 credible websites with full URLs.
Also add a section titled 'Recommended Resources:' listing websites users should visit for more information."""

CITATION_SYSTEM_PROMPT = "You are a helpful research assistant that always cites credible sources with full URLs. Every response must include specific publication names and URLs to support your recommendations."


def _today_label():
    """Returns e.g. 'May 2026' — used to keep LLM responses date-current."""
    return datetime.utcnow().strftime("%B %Y")


def _time_aware_note():
    """Appended to every prompt so LLMs default to recent (last-12-month) sources."""
    today = _today_label()
    return (
        f"\n\nToday's date is {today}. When the question asks about 'latest', "
        f"'recent', 'today', 'current', or otherwise time-sensitive information, "
        f"prioritize sources, articles, and developments from the past 12 months. "
        f"Do NOT default to older years like 2023 or 2024 when more recent content exists."
    )

URL_PATTERN = re.compile(r'https?://[^\s<>"\'`\)\]\},]+')
DOMAIN_BLACKLIST = {'example.com', 'placeholder.com', 'website.com', 'source.com', 'google.com', 'schema.org', 'w3.org', 'googleapis.com'}

INSTITUTIONAL_TLDS = ('.gov', '.gov.uk', '.gov.au', '.gov.ca', '.mil', '.edu', '.ac.uk', '.ac.jp', '.edu.au')

ANALYST_DOMAINS = {
    'gartner.com', 'forrester.com', 'idc.com',
    'omdia.com', '451research.com',
    'frost.com', 'frostandsullivan.com',
    'mckinsey.com', 'bain.com', 'bcg.com',
    'accenture.com', 'deloitte.com', 'pwc.com', 'ey.com', 'kpmg.com',
    'nucleusresearch.com', 'constellationr.com',
    'moorinsightsstrategy.com', 'moor-insights.com',
    'nelsonhall.com', 'everestgrp.com', 'isg-one.com',
    'canalys.com', 'gigaom.com', 'redmonk.com',
    'enterprisestrategygroup.com', 'esg-global.com',
    'futurumresearch.com', 'futuriom.com',
    'tbri.com', 'parkassociates.com',
}

EDITORIAL_ORG_ALLOWLIST = {
    'npr.org', 'propublica.org', 'pbs.org', 'bbc.org', 'reuters.org',
    'consumerreports.org', 'motherjones.org', 'theintercept.org',
    'texastribune.org', 'minnpost.org', 'voxmedia.org', 'theconversation.org',
    'cjr.org', 'niemanlab.org', 'poynter.org',
    'kff.org',
    'aljazeera.org',
}

# Organizations confirmed defunct or merged out of existence. Their domains
# may still serve static pages (so the URL validator passes them), but they
# cannot be partnered with. Filter them from any analyst/institutional output
# so Claude never sees them and can never recommend them.
#
# EXPAND THIS LIST as we discover others — every entry here is a recommendation
# we DIDN'T make to a paying client. When a report surfaces a real-world entity
# that no longer exists, add its domain(s) here.
KNOWN_DEFUNCT_ORGS = {
    'acte.org',                # Association of Corporate Travel Executives, dissolved 2018
}

NON_EDITORIAL_DOMAINS = {
    # User-generated / community
    'wikipedia.org', 'en.wikipedia.org', 'wikimedia.org',
    'reddit.com', 'quora.com', 'stackexchange.com', 'stackoverflow.com',
    'youtube.com', 'youtu.be',
    'linkedin.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
    'tiktok.com', 'pinterest.com', 'medium.com',
    # Marketplaces / retail / e-commerce
    'amazon.com', 'ebay.com', 'etsy.com', 'walmart.com', 'target.com',
    'aliexpress.com', 'shopify.com',
    # Software directories / review marketplaces (NOT editorial media)
    'g2.com', 'capterra.com', 'getapp.com', 'softwareadvice.com',
    'trustradius.com', 'peerspot.com', 'softwarereviews.com',
    'crozdesk.com', 'goodfirms.co', 'clutch.co', 'gartner.com/peer-insights',
    'flexera.com', 'producthunt.com',
    # Job boards
    'indeed.com', 'glassdoor.com', 'ziprecruiter.com',
    # Pure research / not pitchable as editorial
    'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'pmc.ncbi.nlm.nih.gov',
    'who.int', 'cdc.gov',
    'arxiv.org', 'researchgate.net', 'sciencedirect.com', 'springer.com',
    'link.springer.com', 'jstor.org', 'nature.com', 'science.org',
    # Academic publishers & preprint servers (LLMs lean on these heavily for
    # deep-tech categories, but they are not pitchable earned media).
    'mdpi.com', 'ieeexplore.ieee.org', 'wiley.com', 'onlinelibrary.wiley.com',
    'tandfonline.com', 'dl.acm.org', 'acm.org', 'semanticscholar.org',
    'iopscience.iop.org', 'frontiersin.org', 'preprints.org', 'biorxiv.org',
    'medrxiv.org', 'ssrn.com', 'iscience.org',
    # NOTE: keep technologyreview.com and spectrum.ieee.org as editorial — those
    # ARE pitchable. Only the academic-paper subdomains above are excluded.
    # Standards bodies & industry associations (partnership/speaking targets, not
    # pitchable earned media).
    '3gpp.org', 'gsma.com', 'bluetooth.com', 'ietf.org', 'etsi.org', 'itu.int',
    'o-ran.org', 'ieee802.org', 'iso.org', 'wi-fi.org',
    # Patent databases
    'uspto.gov', 'patents.uspto.gov', 'image-ppubs.uspto.gov', 'patents.google.com',
    # Hotel-industry SaaS + consortium + promotional sites (not pitchable editorial)
    'engine.com',                       # corporate booking SaaS
    'fcmtravel.com',                    # corporate travel SaaS
    'ccra.com',                         # travel-tech consortium
    'independentcollection.com',        # hotel marketing consortium
    'designhotels.com',                 # hotel marketing collective (curated listings, not editorial)
    'boutiquehotelsofcalifornia.com',   # regional hotel promo
    'travelplusstyle.com',              # hotel listings/affiliate, not editorial
    # MarTech / AdTech vendor blogs (publish content marketing on their own
    # domain — NOT pitchable as earned editorial media). LLMs cite their blog
    # posts because the content ranks well in search, but pitching them is
    # ineffective: there's no editor, no contributor program, and writing for
    # them is sales-aligned, not journalism.
    'wordstream.com',                   # paid-search SaaS owned by LOCALiQ/Gannett
    'improvado.io',                     # marketing data platform vendor
    'marketinginsidergroup.com',        # Michael Brenner's content-marketing agency
    'impactplus.com',                   # IMPACT content-marketing agency
    'cs-cart.com',                      # e-commerce platform vendor blog
    'canto.com',                        # DAM platform vendor blog
    'thecreativestable.com',            # personal blog / not editorial
    'generation.digital',               # boutique agency content
    'digitallinkage.com',               # agency content
    # User-generated tech publication (anyone can submit; not earned media)
    'towardsdatascience.com',           # Medium-hosted UGC publication
    'hackernoon.com',                   # Same model — UGC tech blog
    'dev.to',                           # UGC developer community
    # B2B-software vendor blogs + direct competitors surfaced in fintech / payroll /
    # AP-automation / ediscovery audits (Ramp, Gusto, Everlaw). Vendor content
    # marketing or a rival's own site — not pitchable earned media.
    'approvalmax.com', 'medius.com', 'nexusap.com',   # AP-automation vendor blogs
    'alaan.com', 'surepayroll.com',                   # direct competitors (spend / payroll)
    'workable.com',                                    # HR-vendor content (resources.workable.com)
    'adamsbrowncpa.com',                               # CPA-firm blog, not media
    'lighthouseglobal.com',                            # ediscovery services vendor
    'nbi-sems.com',                                    # legal CLE seminars vendor
    'expertinsights.com',                              # B2B software review listicle (G2-tier)
    'wifitalents.com',                                 # statistics content-farm
}

# B2B vendors, software marketplaces, and review platforms that LLMs sometimes cite
# but which are NOT pitchable media outlets. Used to suppress false-positive editorial classifications.
NON_EDITORIAL_VENDORS = {
    # Software review marketplaces
    'g2.com', 'capterra.com', 'softwareadvice.com', 'getapp.com', 'trustradius.com',
    'producthunt.com', 'trustpilot.com', 'sourceforge.net', 'alternativeto.net',
    # Major SaaS / enterprise vendors (their own .com is rarely editorial)
    'flexera.com', 'salesforce.com', 'hubspot.com', 'oracle.com', 'sap.com',
    'atlassian.com', 'slack.com', 'monday.com', 'asana.com', 'notion.so',
    'adobe.com', 'microsoft.com', 'aws.amazon.com', 'cloud.google.com',
    'shopify.com', 'wordpress.com', 'wix.com', 'squarespace.com',
    'zendesk.com', 'intercom.com', 'mailchimp.com', 'sendgrid.com',
    # DXP / CMS / martech vendors whose .com surfaces "vs" comparison content
    # for the enterprise marketing audits but isn't pitchable editorial.
    'dotcms.com', 'ingeniux.com', 'coredna.com', 'acquia.com', 'sitecore.com',
    'optimizely.com', 'contentful.com', 'contentstack.com', 'bloomreach.com',
    # Design / no-code / website-builder vendors (Canva-tier audits).
    'desygner.com', 'lucidchart.com', 'sketch.com', 'jimdo.com', 'vibe.us',
    # Boutique digital / design agencies that show up via their comparison blogs.
    'cleardigital.com', 'awesomic.com', 'branded.agency', 'brandedagency.com',
    # Startup / VC data + SaaS sites (not editorial media — surface in BCV /
    # VC-category audits with vendor-comparison or directory pages).
    'tracxn.com', 'aifundingtracker.com', 'foundershield.com',
    'switchpitch.com', 'growthlist.co', 'startupsavant.com', 'openvc.app',
    'qubit.capital', 'visible.vc', 'failory.com', 'everythingstartups.com',
    'rho.co', 'ellty.com', 'vcsheet.com', 'dealroom.net', 'seedtable.com',
    'earthianai.com', 'affinity.co', 'basetemplates.com', 'startupsavant.com',
    'firstparty.io',
    # E-commerce / OTA aggregators
    'expedia.com', 'booking.com', 'kayak.com', 'tripadvisor.com', 'yelp.com',
    'glassdoor.com', 'indeed.com',
    # SaaS / MarTech / productivity tool product sites that LLMs recommend as
    # "tools" — they get cited heavily in B2B audits but aren't pitchable media.
    # (Observed polluting Notion + Adobe dashboards as top "media targets".)
    'zapier.com', 'twilio.com', 'tealium.com', 'appsflyer.com', 'teamwork.com',
    'taskade.com', 'smartsuite.com', 'aprimo.com', 'coda.io', 'airtable.com',
    'clickup.com', 'trello.com', 'obsidian.md', 'taskade.io',
    'miro.com', 'evernote.com', 'roamresearch.com', 'github.com', 'getguru.com',
    'slab.com', 'roam.com', 'workspace.google.com', 'docs.google.com',
    # Software-comparison / lead-gen / B2B-events sites (not editorial)
    'saascompared.com', 'unboundb2b.com', 'thesmarketers.com', 'marcusevans.com',
    # Social-listening / CXM / marketing-agency vendor sites — observed
    # surfacing as "outlets" in martech/CXM audits (Adobe-class brands).
    # Sprinklr is a public CXM SaaS; Konnect Insights is a social listening
    # tool; 42DM and Prometheus Agency are B2B marketing agencies whose
    # comparison blogs get cited but aren't pitchable editorial.
    'sprinklr.com', 'konnectinsights.com', '42dm.net', 'prometheusagency.co',
    # CXM / martech vendor product sites + blogs (NICE is a major CXM vendor;
    # Pushwoosh push-notification SaaS; AnyRoad experiential-analytics vendor).
    'nice.com', 'pushwoosh.com', 'anyroad.com',
    # Syndicated market-research report mills — they sell research PDFs, you
    # can't pitch them an earned story. Observed as top "media targets" in
    # enterprise audits.
    'futuremarketinsights.com', 'custommarketinsights.com',
    'fortunebusinessinsights.com', 'marketsandmarkets.com',
    'grandviewresearch.com', 'mordorintelligence.com', 'precedenceresearch.com',
    # Black & White Zebra content-SEO network ("The * Lead/Manager/CMO" sites)
    # — affiliate/lead-gen publishers with no newsroom, not pitchable press.
    'thecxlead.com', 'thecmo.com', 'theecommmanager.com', 'thedigitalprojectmanager.com',
    'peoplemanagingpeople.com', 'theproductmanager.com', 'thectoclub.com',
    'managingeditor.com', 'indiecators.com',
    # Grounded-mode additions: native web search (ALL_GROUNDED) surfaces ~4x
    # more URLs, so SaaS tool / vendor / software-review .com/.co/.st sites the
    # productish-TLD rule can't catch leak into B2B targets. Observed polluting
    # grounded Notion + Adobe re-runs. (Consumer/beauty/outdoor audits stay
    # clean — this noise is B2B-specific.)
    'ringcentral.com', 'process.st', 'gitbook.com', 'larksuite.com',
    'trackingtime.co', 'waybook.com', 'cake.com', 'upflex.com', 'smartsuite.com',
    'selecthub.com', 'research.com', 'peoplemanagingpeople.com',
    'project-management.com', 'toools.design', 'softwarepundit.com',
}

# Top-level domains that, in the PR/marketing/consumer space, are ~entirely
# SaaS / AI vendor product sites — NOT editorial publications. Verified across
# all saved datasets: every .ai/.io/.so/.md/.cx/.dev/.app domain that surfaced
# was a product/tool site, zero were real publications. Domains on these TLDs
# are classified 'non_editorial' unless explicitly allowlisted
# (EDITORIAL_ORG_ALLOWLIST). NOTE: .co and .us are deliberately EXCLUDED — they
# carry legit publishers (thetrek.co, etc.) so a blanket rule would over-drop.
PRODUCTISH_TLDS = (
    '.ai', '.io', '.so', '.md', '.cx', '.dev', '.app',
    '.xyz', '.tech', '.live', '.build', '.sh',
)

# Certification / standards bodies. AI cites these when a brand's value prop
# involves a certification (B Corp, bluesign, OEKO-TEX, Fair Trade, etc.).
# They influence AI mindshare — but you EARN a certification or partner with
# them, you don't pitch them an earned-media story. So they belong in the
# institutional/partnership bucket, not the editorial media-target list.
# (Most cert bodies are .org and already route to institutional; this catches
# the .com/.net ones that were leaking into editorial — observed on Patagonia.)
CERTIFICATION_DOMAINS = {
    'bcorporation.net', 'bcorporation.com', 'bluesign.com', 'oeko-tex.com',
    'ecocert.com', 'fairtrade.net', 'fairtradecertified.org', 'gots.org',
    'cradletocradle.org', 'c2ccertified.org', 'climateneutral.org',
    'leapingbunny.org', 'crueltyfree.org', 'rainforest-alliance.org',
    'fsc.org', 'us.fsc.org', 'globalrecycled.org', 'responsiblewool.org',
    'usgbc.org', 'energystar.gov', 'gluten.org', 'usda.gov',
}


# Retailers, marketplaces, product databases, and shopping apps. LLMs cite
# these heavily in consumer-product audits (esp. beauty), but you cannot pitch
# a retailer an earned-media story — they sell product, they don't run an
# editorial desk. Polluting the media-target list with these destroys trust
# ("your #1 media target: sephora.com"). Matched on the registrable domain so
# subdomains (shop.sephora.com) are caught too.
RETAILER_DOMAINS = {
    # General + beauty retailers / marketplaces
    'sephora.com', 'ulta.com', 'nordstrom.com', 'macys.com', 'bloomingdales.com',
    'dermstore.com', 'cultbeauty.com', 'cultbeauty.co.uk', 'spacenk.com',
    'lookfantastic.com', 'feelunique.com', 'bluemercury.com', 'beautylish.com',
    'credobeauty.com', 'thedetoxmarket.com', 'thedetoxmarket.ca', 'follain.com',
    'revolve.com', 'asos.com', 'net-a-porter.com', 'goop.com', 'violetgrey.com',
    'kohls.com', 'cvs.com', 'walgreens.com', 'riteaid.com',
    # Ingredient / product databases + shopping/scanner apps (not editorial)
    'incidecoder.com', 'skinsort.com', 'thinkdirtyapp.com', 'ewg.org',
    'thingtesting.com', 'makeupalley.com', 'beautypedia.com', 'cosdna.com',
    'yuka.io', 'thegoodface.org',
}


def classify_citation_domain(domain):
    """Return 'analyst', 'institutional', 'non_editorial', 'retail', 'defunct',
    or 'editorial'.

    'defunct' / 'non_editorial' / 'retail' are terminal non-pitchable
    classifications — domains matching them are excluded from every downstream
    target list so the report never surfaces a retailer, vendor, or dead org
    as a media target.
    """
    d = (domain or "").lower().lstrip('.')
    d_stripped = d[4:] if d.startswith('www.') else d
    # Registrable-ish domain (last two labels) so subdomains route correctly:
    # go.forrester.com → forrester.com (analyst); shop.sephora.com → sephora.com.
    parts = d_stripped.split('.')
    reg = '.'.join(parts[-2:]) if len(parts) >= 2 else d_stripped

    # Defunct check first — wins over every other classification so a defunct
    # .org never leaks into the institutional list.
    if d_stripped in KNOWN_DEFUNCT_ORGS or d in KNOWN_DEFUNCT_ORGS:
        return 'defunct'
    if d in NON_EDITORIAL_DOMAINS or d_stripped in NON_EDITORIAL_DOMAINS or reg in NON_EDITORIAL_DOMAINS:
        return 'non_editorial'
    if d in NON_EDITORIAL_VENDORS or d_stripped in NON_EDITORIAL_VENDORS or reg in NON_EDITORIAL_VENDORS:
        return 'non_editorial'
    # Wire services / PR distribution (Business Wire, PR Newswire, …) — the AI is
    # citing a press release (often the brand's own), not pitchable editorial.
    if d in PR_DOMAINS or d_stripped in PR_DOMAINS or reg in PR_DOMAINS:
        return 'non_editorial'
    if d_stripped in RETAILER_DOMAINS or reg in RETAILER_DOMAINS:
        return 'retail'
    # Certification / standards bodies → institutional (partner/certify, don't
    # pitch). Checked before the editorial fallback so .com/.net cert bodies
    # don't leak into media targets.
    if d_stripped in CERTIFICATION_DOMAINS or reg in CERTIFICATION_DOMAINS:
        return 'institutional'
    # Analyst: match the registrable domain too so subdomains like
    # go.forrester.com / www2.gartner.com route to 'analyst', not 'editorial'.
    if d_stripped in ANALYST_DOMAINS or d in ANALYST_DOMAINS or reg in ANALYST_DOMAINS:
        return 'analyst'
    if d_stripped in EDITORIAL_ORG_ALLOWLIST:
        return 'editorial'
    if any(d.endswith(tld) for tld in INSTITUTIONAL_TLDS):
        return 'institutional'
    # .org and country-coded advocacy orgs (.org.uk, .org.au, …): patient /
    # disability / advocacy orgs are institutional partners, not media targets.
    if d.endswith('.org') or re.search(r'\.org\.[a-z]{2,3}$', d):
        return 'institutional'
    # Productish-TLD vendor sites (.ai/.io/.so/.md/.cx/.dev/.app...) — not
    # pitchable media. Checked AFTER the editorial allowlist so a deliberately
    # allowlisted exception still wins.
    if any(d.endswith(tld) for tld in PRODUCTISH_TLDS):
        return 'non_editorial'
    return 'editorial'


import unicodedata as _unicodedata

# Leading/structural words that should not, on their own, identify a brand's
# domain. Used when deriving competitor + brand domain stems.
_BRAND_STOPWORDS = {
    'the', 'and', 'a', 'an', 'of', 'co', 'inc', 'llc', 'ltd', 'corp',
    'corporation', 'company', 'group', 'holdings', 'global', 'international',
}


def _ascii_fold(s):
    """Strip accents/diacritics so 'Fjällräven' → 'fjallraven' and the domain
    fjallraven.com matches. NFKD-decompose then drop combining marks."""
    try:
        return ''.join(c for c in _unicodedata.normalize('NFKD', s)
                       if not _unicodedata.combining(c))
    except Exception:
        return s


def _competitor_domain_stems(competitor_counts):
    """Return {'words': set, 'concats': set} of domain stems for the brand's
    competitors, so we can drop competitor-owned sites from the editorial list.

    Two stem kinds, because a competitor's site can show up two ways:
      words   — the first significant word of the name (accent-folded, alnum,
                >= 3 chars, not a structural stopword). Matched against any
                dot-SEGMENT of a domain. Catches subdomains:
                  'REI Co-op'       → 'rei'   → newsroom.rei.com
                  'The North Face'  → 'north' → north.example.com
      concats — the full no-space concatenation of the name, AND the
                concatenation minus a leading stopword. Matched EXACTLY against
                a domain's registrable label. Catches multi-word brands whose
                site concatenates the words (the common real case):
                  'Guide Beauty'    → 'guidebeauty'   → guidebeauty.com
                  'Kohl Kreatives'  → 'kohlkreatives' → kohlkreatives.com
                  'The North Face'  → 'thenorthface' + 'northface' → thenorthface.com
                  'Fenty Beauty'    → 'fentybeauty'   → fentybeauty.com

    Exact-match only on both kinds (see _is_competitor_owned_domain) so generic
    words don't over-match legit publications.
    """
    words, concats = set(), set()
    for c in (competitor_counts or []):
        raw = (c.get('name') or '').strip()
        if not raw:
            continue
        name = _ascii_fold(raw.lower())
        parts = [re.sub(r'[^a-z0-9]', '', p) for p in name.split()]
        parts = [p for p in parts if p]
        if not parts:
            continue
        # First significant word (skip a leading stopword like "The").
        sig = [p for p in parts if p not in _BRAND_STOPWORDS]
        if sig and len(sig[0]) >= 3:
            words.add(sig[0])
        # Full concatenation + concatenation-without-leading-stopword.
        full = ''.join(parts)
        if len(full) >= 4:
            concats.add(full)
        if parts[0] in _BRAND_STOPWORDS and len(parts) > 1:
            trimmed = ''.join(parts[1:])
            if len(trimmed) >= 4:
                concats.add(trimmed)
    return {'words': words, 'concats': concats}


# SaaS/brand domains very commonly prefix the brand name with one of these
# ("getquantic.com", "trynotion.com", "usepylon.com", "joinhoney.com"). The
# competitor-owned-domain matcher compares the registrable label against the
# brand's no-space concat — so we ALSO emit the prefix-stripped form, or a
# competitor's own site leaks in as a "media target" (observed: getquantic.com
# surfacing as a pitch outlet in a SpotOn audit while Quantic is a competitor).
# Only the unambiguous prefixes; remainder must stay >= 4 chars.
_SAAS_DOMAIN_PREFIXES = ('get', 'try', 'use', 'join', 'meet', 'go')


def _registrable_label_candidates(segs):
    """Given a domain's alnum dot-segments, return the labels that could be the
    'registrable' name (the bit before the public-suffix-ish TLD). Handles both
    foo.com (label 'foo') and foo.co.uk (label 'foo', not 'co'), plus the
    SaaS-prefix-stripped form (getquantic → quantic)."""
    cands = set()
    if len(segs) >= 2:
        cands.add(segs[-2])
    if len(segs) >= 3:
        cands.add(segs[-3])  # foo.co.uk → also consider 3rd-to-last
    if segs:
        cands.add(segs[0])
    for c in list(cands):
        for p in _SAAS_DOMAIN_PREFIXES:
            if c.startswith(p) and len(c) - len(p) >= 4:
                cands.add(c[len(p):])
    return cands


def _is_competitor_owned_domain(domain, competitor_stems):
    """True if `domain` is a competitor's own site.

    competitor_stems is the dict from _competitor_domain_stems. Match rules
    (exact-equality only, never substring/startswith, to avoid nuking legit
    publications like 'notionalfinance.com' for competitor 'Notion'):
      - the domain's registrable label EQUALS a concat stem
        (guidebeauty.com → 'guidebeauty', thenorthface.com → 'thenorthface')
      - any dot-segment EQUALS a word stem
        (newsroom.rei.com → segment 'rei')

    Accepts the legacy plain-set shape too (treated as word stems) so any
    older caller still works.
    """
    if not domain or not competitor_stems:
        return False
    if isinstance(competitor_stems, (set, frozenset)):
        competitor_stems = {'words': set(competitor_stems), 'concats': set()}
    words = competitor_stems.get('words') or set()
    concats = competitor_stems.get('concats') or set()
    d = _ascii_fold((domain or '').lower())
    segs = [re.sub(r'[^a-z0-9]', '', p) for p in d.split('.')]
    segs = [s for s in segs if s]
    if not segs:
        return False
    for cand in _registrable_label_candidates(segs):
        if cand in concats:
            return True
    for s in segs:
        if s in words:
            return True
    return False


def _is_brand_own_domain(domain, brand):
    """Quick deterministic check: is this domain the brand's own property?

    Catches cases where the LLM cites the brand's own developer site,
    community forums, help center, business page, etc. These should never
    appear as "editorial media targets" — you can't pitch yourself. Examples:
      brand='Adobe' + domain='developer.adobe.com'   → True
      brand='Adobe' + domain='community.adobe.com'   → True
      brand='Adobe' + domain='helpx.adobe.com'       → True
      brand='Adobe' + domain='business.adobe.com'    → True
      brand='Adobe' + domain='adobe.com'             → True
      brand='Adobe' + domain='theverge.com'          → False

    Matches on the *root* domain (second-level), so subdomains are caught.
    Brand slugs under 3 chars are skipped to avoid noise (e.g. matching
    "ai" inside arbitrary domains).
    """
    if not brand or not domain:
        return False
    # Accent-fold first: "Estée Lauder" -> "estee lauder" -> "esteelauder" so it
    # matches esteelauder.com (without folding, the stripped 'é' yields
    # "estelauder", which misses the double-e domain and undercounts owned media).
    brand_slug = re.sub(r'[^a-z0-9]', '', _ascii_fold(brand.lower()))
    if len(brand_slug) < 3:
        return False
    d = (domain or '').lower().lstrip('.')
    if d.startswith('www.'):
        d = d[4:]
    # Strip path / port if any leaked in.
    d = d.split('/')[0].split(':')[0]
    parts = d.split('.')
    if len(parts) < 2:
        return False
    root = parts[-2].lower()
    # Bidirectional containment so multi-word brands ("HelloFresh" → hellofresh.com)
    # and single-word brands ("Adobe" → adobe.com / developer.adobe.com) both match.
    if brand_slug in root or root in brand_slug:
        return True
    return False


def verify_editorial_domains(editorial_domains, brand, category):
    """Filter the editorial list via Claude to drop B2B vendor/marketplace/non-media domains.

    Returns (verified_media, rejected) — both lists in original ranking order.
    On any failure, returns (editorial_domains, []) — safe to no-op.
    """
    if not editorial_domains:
        return [], []
    domains_to_check = editorial_domains[:30]
    domain_list = "\n".join(f"- {d['domain']}" for d in domains_to_check)
    # Brand-specific self-reference examples make the rejection rule concrete
    # for Claude. Build the most-common self-domain patterns from the brand
    # slug so the prompt can call them out by name.
    brand_for_prompt = (brand or '').strip() or '(unknown)'
    brand_slug = re.sub(r'[^a-z0-9]', '', brand_for_prompt.lower())
    self_examples = ""
    if len(brand_slug) >= 3:
        self_examples = (
            f"  e.g. for {brand_for_prompt}: {brand_slug}.com, www.{brand_slug}.com, "
            f"developer.{brand_slug}.com, community.{brand_slug}.com, helpx.{brand_slug}.com, "
            f"business.{brand_slug}.com, support.{brand_slug}.com, blog.{brand_slug}.com, "
            f"docs.{brand_slug}.com.\n"
        )
    try:
        resp = anthropic.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": (
                    "You are filtering a list of web domains to identify which are legitimate editorial "
                    "MEDIA outlets that a PR professional could pitch — vs. which are SaaS vendors, "
                    "software review marketplaces, e-commerce, community forums, the brand's own "
                    "properties, or corporate sites that aren't pitchable media.\n\n"
                    f"Context — brand: {brand_for_prompt}\nCategory: {category}\n\n"
                    "Definition of MEDIA: newspapers, magazines, trade press, B2B publications, "
                    "consumer publications, broadcast outlets, and digital-native news sites WITH "
                    "editorial staff who write articles.\n\n"
                    "Definition of NON-MEDIA (reject these aggressively):\n"
                    f"- THE BRAND'S OWN DOMAINS — including any subdomain of the brand's root domain.\n"
                    f"{self_examples}"
                    "  You cannot pitch yourself, and a brand's own developer/community/help/business "
                    "subdomains are self-promotional infrastructure, not editorial media.\n"
                    "- Vendor documentation / learning portals (e.g. learn.microsoft.com, docs.aws.amazon.com, "
                    "developer.nvidia.com, developer.apple.com, developer.mozilla.org). These are vendor docs, "
                    "not editorial media.\n"
                    "- AI / SaaS competitor product sites masquerading as 'media' (e.g. synthesia.io, "
                    "lumen5.com, datarobot.com, h2o.ai, anthropic.com, openai.com, runway.com). These are "
                    "product/marketing sites for vendors — sometimes they have blogs, but they're not "
                    "pitchable editorial.\n"
                    "- SaaS / MarTech / AdTech VENDOR BLOGS publishing content marketing on their own "
                    "domain. They rank well in search so LLMs cite them, but there is no editorial team, "
                    "no journalists, and pitching them as media doesn't work. Examples:\n"
                    "    wordstream.com (paid-search SaaS, owned by LOCALiQ/Gannett)\n"
                    "    improvado.io (marketing-data platform)\n"
                    "    canto.com (DAM platform)\n"
                    "    cs-cart.com (e-commerce platform)\n"
                    "    blog.hubspot.com (HubSpot's own content marketing)\n"
                    "    salesforce.com/blog (Salesforce's own content marketing)\n"
                    "  If a domain's primary business is selling SaaS / MarTech / AdTech software AND it "
                    "publishes a 'blog' / 'resources' / 'guides' section, it is a VENDOR BLOG, not editorial.\n"
                    "- CONTENT-MARKETING AGENCIES / personal thought-leadership sites masquerading as "
                    "media. Owned by individuals or small agencies who write self-promotional content. "
                    "There is no editor, no pitch process — they only publish their own work. Examples:\n"
                    "    marketinginsidergroup.com (Michael Brenner's agency)\n"
                    "    impactplus.com (IMPACT content marketing)\n"
                    "    thecreativestable.com\n"
                    "    digitallinkage.com\n"
                    "    generation.digital\n"
                    "  Red flag: site copy says 'we' / 'our agency' / 'work with us' on the homepage.\n"
                    "- USER-GENERATED PUBLICATION platforms where anyone can submit (Medium pubs, "
                    "Hacker Noon, Dev.to). Editorial structure is absent — there's no journalist to "
                    "pitch, just an open submission queue. Examples:\n"
                    "    towardsdatascience.com, hackernoon.com, dev.to, medium.com\n"
                    "- SaaS vendor sites (e.g. salesforce.com, atlassian.com, engine.com, fcmtravel.com, "
                    "sap.com, hubspot.com)\n"
                    "- SOFTWARE / APP / TOOL PRODUCT SITES that an AI recommends as a 'tool' or "
                    "'alternative'. This is the most common false positive in B2B and consumer-tech "
                    "audits: the LLM lists competing or adjacent products and cites their homepages. "
                    "These are product marketing sites, NOT pitchable media — there is no newsroom. "
                    "Examples: zapier.com, miro.com, evernote.com, obsidian.md, rock.so, taskade.com, "
                    "eesel.ai, lindy.ai, twilio.com, tealium.com, customer.io, heygen.com, stockimg.ai. "
                    "RULE OF THUMB: if the domain's primary business is selling/offering a software "
                    "product, app, or platform — even a well-known one — REJECT it. A real media outlet "
                    "publishes journalism about the category; a tool site sells the tool.\n"
                    "- RETAILERS / MARKETPLACES / SHOPPING or product-database sites (sephora.com, "
                    "ulta.com, nordstrom.com, credobeauty.com, incidecoder.com, skinsort.com, "
                    "thingtesting.com). You pitch journalists, not stores.\n"
                    "- Corporate-booking platforms or travel-tech consortia (e.g. ccra.com, gbta.org for "
                    "SaaS purposes, sabre.com, amadeus.com)\n"
                    "- Hotel-marketing collectives or member-curated listings (e.g. designhotels.com, "
                    "independentcollection.com, smallluxuryhotels.com, relaischateaux.com)\n"
                    "- Tourism boards' member-listing pages (e.g. visitseattle.org/members/..., "
                    "visitcalifornia.com/listings/...)\n"
                    "- Affiliate / booking aggregators (e.g. booking.com, expedia.com, hotels.com, "
                    "tripadvisor.com, kayak.com)\n"
                    "- Promotional regional consortia (e.g. boutiquehotelsofcalifornia.com)\n"
                    "- Listing sites that do NOT have editorial bylines or news coverage\n"
                    "- Review marketplaces (e.g. g2.com, capterra.com, trustradius.com, trustpilot.com)\n"
                    "- Corporate blogs, Wikipedia, Reddit/Stack Overflow communities, LinkedIn, Quora, "
                    "Amazon.\n\n"
                    f"Classify each domain below:\n{domain_list}\n\n"
                    "Respond with ONLY valid JSON, no prose:\n"
                    '{"media_domains": ["domain1.com", ...], "non_media_domains": ["domain2.com", ...]}'
                )
            }]
        )
        text = resp.content[0].text
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return editorial_domains, []
        data = json.loads(m.group())
        media_set = {(d or '').lower() for d in data.get('media_domains', [])}
        checked_set = {d['domain'].lower() for d in domains_to_check}
        # Keep: (a) anything beyond the first 30 we didn't check; (b) within the first 30, only verified-media.
        verified = [
            d for d in editorial_domains
            if d['domain'].lower() not in checked_set or d['domain'].lower() in media_set
        ]
        rejected = [d for d in domains_to_check if d['domain'].lower() not in media_set]
        return verified, rejected
    except Exception as e:
        print("verify_editorial_domains failed (continuing with unfiltered list):", e)
        return editorial_domains, []


def extract_urls(text):
    """Extract URLs from a block of text, clean trailing punctuation, filter blacklisted domains."""
    if not text:
        return []
    urls = URL_PATTERN.findall(text)
    out = []
    seen = set()
    for u in urls:
        u = u.rstrip('.,;:!?\'")]}')
        try:
            host = u.split('/')[2].lower()
        except Exception:
            continue
        host_no_www = host[4:] if host.startswith('www.') else host
        if host_no_www in DOMAIN_BLACKLIST:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append({"url": u, "domain": host_no_www})
    return out


def aggregate_citations(all_responses):
    """Count citation URLs by domain across responses, tracking LLMs and prompts.
    Ranking uses a diversity-weighted score so a domain cited by N LLMs ranks
    above one cited only by a single LLM at the same raw count — kills the
    'one LLM dominates and crowds out broader consensus' failure mode.
    """
    domain_data = {}
    for resp in all_responses:
        urls = resp.get('citations', [])
        for u in urls:
            domain = u['domain']
            if domain not in domain_data:
                domain_data[domain] = {
                    'domain': domain,
                    'urls': [],
                    'count': 0,
                    'llms': set(),
                    'prompts': set(),
                }
            domain_data[domain]['count'] += 1
            domain_data[domain]['urls'].append(u['url'])
            domain_data[domain]['llms'].add(resp['llm'])
            domain_data[domain]['prompts'].add(resp['prompt'])

    for d in domain_data.values():
        d['llms'] = list(d['llms'])
        d['prompts'] = list(d['prompts'])
        d['urls'] = list(set(d['urls']))
        # Article-only subset for `sample_urls` surfacing. Keeps the full
        # `urls` list intact (preserves raw citation data for JSON export +
        # downstream analysis), but exposes a curated list of URLs that look
        # like real articles — no homepages, no /tag/ index pages, no search
        # results. The analysis prompt prefers this list for sample_urls.
        d['specific_urls'] = [u for u in d['urls'] if _is_specific_article(u)]
        # diversity_score = count * (1 + 0.25 * (distinct LLMs - 1))
        # 1 LLM: ×1.00 · 2 LLMs: ×1.25 · 3 LLMs: ×1.50 · 4: ×1.75 · 5: ×2.00
        d['diversity_score'] = d['count'] * (1 + 0.25 * max(0, len(d['llms']) - 1))

    return sorted(domain_data.values(), key=lambda x: x['diversity_score'], reverse=True)


# ---------------------------------------------------------------------------
# Named-outlet detection — count outlets by NAME, not just by fetchable URL.
# ---------------------------------------------------------------------------
# Problem this solves: the citation universe (`aggregate_citations`) is built
# only from extracted URLs. But AI assistants — especially search-grounded ones
# (Gemini, Perplexity) — routinely NAME influential outlets in prose and in
# numbered reference lists while attaching URLs that don't resolve (hallucinated
# article paths). Topic-verify then prunes those URLs, so the outlet drops to a
# near-zero count or disappears entirely — even though three different models
# clearly cited it. Real, frequently-named publications (WWD, Fortune, Forbes,
# TechCrunch, The Verge, Harper's Bazaar…) were being badly under-counted or
# left off the report. The product's job is to identify the outlets that shape
# AI visibility, not to grade them on whether a link happened to survive.
#
# Fix: detect an outlet when its NAME (or bare domain string) appears in a
# response, independent of any working URL.
#
# PRECISION GUARANTEE: name-detection is gated to this curated ALLOWLIST of
# known editorial outlets. It can therefore never surface a competitor brand,
# retailer, or vendor product site by name — the registry *is* the safety
# mechanism. (That's the failure mode five cleanup cycles removed; this keeps
# it removed.)
#
# Per outlet:  domain -> {"name": display, "ci": [...], "abbr": [...]}
#   ci   — multiword names + coined single tokens that are NOT common English
#          words (e.g. "Beauty Independent", "TechCrunch", "Byrdie"). Matched
#          case-INsensitively on word boundaries — safe because they don't
#          collide with ordinary prose.
#   abbr — uppercase abbreviations (WWD, WSJ, NYT…). Matched case-SENSITIVELY so
#          "NYT" hits but "ap"/"ad"/"self" never do.
#   (no alias) — outlets whose only short name IS a common English word
#          (Fortune, Time, Self, Vogue, Allure, Outside, Wired…) carry no name
#          alias; they're still caught by their URL or bare-domain string, which
#          is unambiguous. This deliberately trades a little recall for zero
#          false positives on common nouns.
EDITORIAL_OUTLETS = {
    # ---- Telecom / 5G / IoT / robotics / deep-tech trade press ----
    "lightreading.com": {"name": "Light Reading", "ci": ["Light Reading"], "abbr": []},
    "rcrwireless.com": {"name": "RCR Wireless News", "ci": ["RCR Wireless"], "abbr": ["RCR"]},
    "fierce-network.com": {"name": "Fierce Network", "ci": ["Fierce Network", "FierceWireless", "Fierce Wireless", "FierceTelecom"], "abbr": []},
    "fiercewireless.com": {"name": "FierceWireless", "ci": ["FierceWireless", "Fierce Wireless"], "abbr": []},
    "telecoms.com": {"name": "Telecoms.com", "ci": ["Telecoms.com"], "abbr": []},
    "telecomtv.com": {"name": "TelecomTV", "ci": ["TelecomTV"], "abbr": []},
    "mobileworldlive.com": {"name": "Mobile World Live", "ci": ["Mobile World Live"], "abbr": ["MWL"]},
    "sdxcentral.com": {"name": "SDxCentral", "ci": ["SDxCentral"], "abbr": []},
    "iotbusinessnews.com": {"name": "IoT Business News", "ci": ["IoT Business News"], "abbr": []},
    "iotworldtoday.com": {"name": "IoT World Today", "ci": ["IoT World Today"], "abbr": []},
    "therobotreport.com": {"name": "The Robot Report", "ci": ["The Robot Report"], "abbr": []},
    "thenewstack.io": {"name": "The New Stack", "ci": ["The New Stack"], "abbr": []},
    "unmannedsystemstechnology.com": {"name": "Unmanned Systems Technology", "ci": ["Unmanned Systems Technology"], "abbr": []},
    "criticalcomms.com.au": {"name": "Critical Comms", "ci": ["Critical Comms"], "abbr": []},
    "spectrum.ieee.org": {"name": "IEEE Spectrum", "ci": ["IEEE Spectrum"], "abbr": []},
    # ---- Beauty / fashion ----
    "wwd.com": {"name": "WWD", "ci": ["Women's Wear Daily"], "abbr": ["WWD"]},
    "businessoffashion.com": {"name": "Business of Fashion", "ci": ["Business of Fashion"], "abbr": ["BoF"]},
    "glossy.co": {"name": "Glossy", "ci": ["Glossy"], "abbr": []},
    "harpersbazaar.com": {"name": "Harper's Bazaar", "ci": ["Harper's Bazaar"], "abbr": []},
    "allure.com": {"name": "Allure", "ci": [], "abbr": []},
    "vogue.com": {"name": "Vogue", "ci": ["Teen Vogue", "Vogue Business"], "abbr": []},
    "elle.com": {"name": "Elle", "ci": [], "abbr": []},
    "cosmopolitan.com": {"name": "Cosmopolitan", "ci": [], "abbr": []},
    "glamour.com": {"name": "Glamour", "ci": [], "abbr": []},
    "marieclaire.com": {"name": "Marie Claire", "ci": ["Marie Claire"], "abbr": []},
    "instyle.com": {"name": "InStyle", "ci": ["InStyle"], "abbr": []},
    "refinery29.com": {"name": "Refinery29", "ci": ["Refinery29", "Refinery 29"], "abbr": []},
    "popsugar.com": {"name": "PopSugar", "ci": ["PopSugar", "Pop Sugar"], "abbr": []},
    "whowhatwear.com": {"name": "Who What Wear", "ci": ["Who What Wear"], "abbr": []},
    "fashionista.com": {"name": "Fashionista", "ci": ["Fashionista"], "abbr": []},
    "byrdie.com": {"name": "Byrdie", "ci": ["Byrdie"], "abbr": []},
    "beautyindependent.com": {"name": "Beauty Independent", "ci": ["Beauty Independent"], "abbr": []},
    "newbeauty.com": {"name": "NewBeauty", "ci": ["NewBeauty", "New Beauty"], "abbr": []},
    "beautypackaging.com": {"name": "Beauty Packaging", "ci": ["Beauty Packaging"], "abbr": []},
    "cosmeticsandtoiletries.com": {"name": "Cosmetics & Toiletries", "ci": ["Cosmetics & Toiletries", "Cosmetics and Toiletries"], "abbr": []},
    "thezoereport.com": {"name": "The Zoe Report", "ci": ["The Zoe Report"], "abbr": []},
    "coveteur.com": {"name": "Coveteur", "ci": ["Coveteur"], "abbr": []},
    "nylon.com": {"name": "Nylon", "ci": [], "abbr": []},
    "dazeddigital.com": {"name": "Dazed", "ci": ["Dazed Digital"], "abbr": []},
    "hypebeast.com": {"name": "Hypebeast", "ci": ["Hypebeast"], "abbr": []},
    "hypebae.com": {"name": "Hypebae", "ci": ["Hypebae"], "abbr": []},
    "thecut.com": {"name": "The Cut", "ci": ["The Cut"], "abbr": []},
    "wmagazine.com": {"name": "W Magazine", "ci": ["W Magazine"], "abbr": []},
    "highsnobiety.com": {"name": "Highsnobiety", "ci": ["Highsnobiety"], "abbr": []},
    # ---- Health / wellness / lifestyle / home ----
    "themighty.com": {"name": "The Mighty", "ci": ["The Mighty"], "abbr": []},
    "healthline.com": {"name": "Healthline", "ci": ["Healthline"], "abbr": []},
    "verywellhealth.com": {"name": "Verywell Health", "ci": ["Verywell Health", "Verywell"], "abbr": []},
    "wellandgood.com": {"name": "Well+Good", "ci": ["Well+Good", "Well and Good"], "abbr": []},
    "self.com": {"name": "Self", "ci": [], "abbr": []},
    "womenshealthmag.com": {"name": "Women's Health", "ci": ["Women's Health"], "abbr": []},
    "menshealth.com": {"name": "Men's Health", "ci": ["Men's Health"], "abbr": []},
    "goodhousekeeping.com": {"name": "Good Housekeeping", "ci": ["Good Housekeeping"], "abbr": []},
    "realsimple.com": {"name": "Real Simple", "ci": ["Real Simple"], "abbr": []},
    "apartmenttherapy.com": {"name": "Apartment Therapy", "ci": ["Apartment Therapy"], "abbr": []},
    "architecturaldigest.com": {"name": "Architectural Digest", "ci": ["Architectural Digest"], "abbr": []},
    "thespruce.com": {"name": "The Spruce", "ci": ["The Spruce"], "abbr": []},
    # ---- General news / business ----
    "reuters.com": {"name": "Reuters", "ci": ["Reuters"], "abbr": []},
    "bloomberg.com": {"name": "Bloomberg", "ci": ["Bloomberg"], "abbr": []},
    "forbes.com": {"name": "Forbes", "ci": ["Forbes"], "abbr": []},
    "wsj.com": {"name": "The Wall Street Journal", "ci": ["Wall Street Journal"], "abbr": ["WSJ"]},
    "nytimes.com": {"name": "The New York Times", "ci": ["New York Times"], "abbr": ["NYT"]},
    "washingtonpost.com": {"name": "The Washington Post", "ci": ["Washington Post"], "abbr": []},
    "theguardian.com": {"name": "The Guardian", "ci": ["The Guardian"], "abbr": []},
    "cnbc.com": {"name": "CNBC", "ci": ["CNBC"], "abbr": []},
    "businessinsider.com": {"name": "Business Insider", "ci": ["Business Insider"], "abbr": []},
    "fastcompany.com": {"name": "Fast Company", "ci": ["Fast Company"], "abbr": []},
    "theatlantic.com": {"name": "The Atlantic", "ci": ["The Atlantic"], "abbr": []},
    "axios.com": {"name": "Axios", "ci": ["Axios"], "abbr": []},
    "ft.com": {"name": "Financial Times", "ci": ["Financial Times"], "abbr": []},
    "economist.com": {"name": "The Economist", "ci": ["The Economist"], "abbr": []},
    "fortune.com": {"name": "Fortune", "ci": [], "abbr": []},
    "inc.com": {"name": "Inc.", "ci": ["Inc. Magazine", "Inc Magazine"], "abbr": []},
    "entrepreneur.com": {"name": "Entrepreneur", "ci": [], "abbr": []},
    "hbr.org": {"name": "Harvard Business Review", "ci": ["Harvard Business Review"], "abbr": ["HBR"]},
    "npr.org": {"name": "NPR", "ci": [], "abbr": ["NPR"]},
    "bbc.com": {"name": "BBC", "ci": [], "abbr": ["BBC"]},
    "vox.com": {"name": "Vox", "ci": [], "abbr": []},
    "marketwatch.com": {"name": "MarketWatch", "ci": ["MarketWatch"], "abbr": []},
    "qz.com": {"name": "Quartz", "ci": [], "abbr": []},
    # ---- Tech ----
    "techcrunch.com": {"name": "TechCrunch", "ci": ["TechCrunch"], "abbr": []},
    "theverge.com": {"name": "The Verge", "ci": ["The Verge"], "abbr": []},
    "wired.com": {"name": "Wired", "ci": [], "abbr": []},
    "arstechnica.com": {"name": "Ars Technica", "ci": ["Ars Technica"], "abbr": []},
    "engadget.com": {"name": "Engadget", "ci": ["Engadget"], "abbr": []},
    "venturebeat.com": {"name": "VentureBeat", "ci": ["VentureBeat"], "abbr": []},
    "mashable.com": {"name": "Mashable", "ci": ["Mashable"], "abbr": []},
    "gizmodo.com": {"name": "Gizmodo", "ci": ["Gizmodo"], "abbr": []},
    "cnet.com": {"name": "CNET", "ci": ["CNET"], "abbr": []},
    "zdnet.com": {"name": "ZDNet", "ci": ["ZDNet"], "abbr": []},
    "technologyreview.com": {"name": "MIT Technology Review", "ci": ["MIT Technology Review"], "abbr": []},
    "theinformation.com": {"name": "The Information", "ci": ["The Information"], "abbr": []},
    "pcmag.com": {"name": "PCMag", "ci": ["PCMag"], "abbr": []},
    "pcworld.com": {"name": "PCWorld", "ci": ["PCWorld"], "abbr": []},
    "techradar.com": {"name": "TechRadar", "ci": ["TechRadar"], "abbr": []},
    "tomsguide.com": {"name": "Tom's Guide", "ci": ["Tom's Guide"], "abbr": []},
    "tomshardware.com": {"name": "Tom's Hardware", "ci": ["Tom's Hardware"], "abbr": []},
    "digitaltrends.com": {"name": "Digital Trends", "ci": ["Digital Trends"], "abbr": []},
    "thenextweb.com": {"name": "The Next Web", "ci": ["The Next Web"], "abbr": ["TNW"]},
    "9to5mac.com": {"name": "9to5Mac", "ci": ["9to5Mac"], "abbr": []},
    "macrumors.com": {"name": "MacRumors", "ci": ["MacRumors"], "abbr": []},
    "restofworld.org": {"name": "Rest of World", "ci": ["Rest of World"], "abbr": []},
    # ---- Marketing / adtech / B2B SaaS press ----
    "adage.com": {"name": "Ad Age", "ci": ["Ad Age", "AdAge"], "abbr": []},
    "adweek.com": {"name": "Adweek", "ci": ["Adweek"], "abbr": []},
    "digiday.com": {"name": "Digiday", "ci": ["Digiday"], "abbr": []},
    "thedrum.com": {"name": "The Drum", "ci": ["The Drum"], "abbr": []},
    "prweek.com": {"name": "PRWeek", "ci": ["PRWeek", "PR Week"], "abbr": []},
    "marketingdive.com": {"name": "Marketing Dive", "ci": ["Marketing Dive"], "abbr": []},
    "retaildive.com": {"name": "Retail Dive", "ci": ["Retail Dive"], "abbr": []},
    "modernretail.co": {"name": "Modern Retail", "ci": ["Modern Retail"], "abbr": []},
    "chiefmartec.com": {"name": "Chiefmartec", "ci": ["Chiefmartec", "Chief Martec"], "abbr": []},
    "searchenginejournal.com": {"name": "Search Engine Journal", "ci": ["Search Engine Journal"], "abbr": ["SEJ"]},
    "searchengineland.com": {"name": "Search Engine Land", "ci": ["Search Engine Land"], "abbr": []},
    "cmswire.com": {"name": "CMSWire", "ci": ["CMSWire"], "abbr": []},
    "techrepublic.com": {"name": "TechRepublic", "ci": ["TechRepublic"], "abbr": []},
    "infoworld.com": {"name": "InfoWorld", "ci": ["InfoWorld"], "abbr": []},
    "computerworld.com": {"name": "Computerworld", "ci": ["Computerworld"], "abbr": []},
    "thenewstack.io": {"name": "The New Stack", "ci": ["The New Stack"], "abbr": []},
    "thedigitalprojectmanager.com": {"name": "The Digital Project Manager", "ci": ["The Digital Project Manager"], "abbr": []},
    # ---- Outdoor / sustainability ----
    "outsideonline.com": {"name": "Outside", "ci": ["Outside Magazine", "Outside Online"], "abbr": []},
    "backpacker.com": {"name": "Backpacker", "ci": ["Backpacker"], "abbr": []},
    "gearpatrol.com": {"name": "Gear Patrol", "ci": ["Gear Patrol"], "abbr": []},
    "goodonyou.eco": {"name": "Good On You", "ci": ["Good On You"], "abbr": []},
    "treehugger.com": {"name": "Treehugger", "ci": ["Treehugger"], "abbr": []},
    "grist.org": {"name": "Grist", "ci": [], "abbr": []},
    # ---- Food / travel / hospitality ----
    "eater.com": {"name": "Eater", "ci": ["Eater"], "abbr": []},
    "bonappetit.com": {"name": "Bon Appétit", "ci": ["Bon Appetit", "Bon Appétit"], "abbr": []},
    "foodandwine.com": {"name": "Food & Wine", "ci": ["Food & Wine", "Food and Wine"], "abbr": []},
    "seriouseats.com": {"name": "Serious Eats", "ci": ["Serious Eats"], "abbr": []},
    "skift.com": {"name": "Skift", "ci": ["Skift"], "abbr": []},
    "travelandleisure.com": {"name": "Travel + Leisure", "ci": ["Travel + Leisure", "Travel and Leisure"], "abbr": []},
    "cntraveler.com": {"name": "Condé Nast Traveler", "ci": ["Conde Nast Traveler", "Condé Nast Traveler"], "abbr": []},
    "thepointsguy.com": {"name": "The Points Guy", "ci": ["The Points Guy"], "abbr": ["TPG"]},
    # ---- Personal finance ----
    "investopedia.com": {"name": "Investopedia", "ci": ["Investopedia"], "abbr": []},
    "nerdwallet.com": {"name": "NerdWallet", "ci": ["NerdWallet"], "abbr": []},
    "bankrate.com": {"name": "Bankrate", "ci": ["Bankrate"], "abbr": []},
    "kiplinger.com": {"name": "Kiplinger", "ci": ["Kiplinger"], "abbr": []},
    "barrons.com": {"name": "Barron's", "ci": ["Barron's"], "abbr": []},
}


def _norm_apostrophes(t):
    """Fold curly quotes to straight so 'Harper's' patterns match LLM prose."""
    return (t or '').replace('’', "'").replace('‘', "'")


_OUTLET_PATTERNS = None


def _outlet_patterns():
    """Lazily compile the registry into match patterns (compiled once)."""
    global _OUTLET_PATTERNS
    if _OUTLET_PATTERNS is None:
        m = {}
        for dom, a in EDITORIAL_OUTLETS.items():
            m[dom] = {
                'name': a.get('name') or dom,
                'ci': [re.compile(r'\b' + re.escape(_norm_apostrophes(x)) + r'\b', re.IGNORECASE)
                       for x in a.get('ci', [])],
                'abbr': [re.compile(r'\b' + re.escape(x) + r'\b') for x in a.get('abbr', [])],
                'dom': re.compile(r'\b' + re.escape(dom) + r'\b', re.IGNORECASE),
            }
        _OUTLET_PATTERNS = m
    return _OUTLET_PATTERNS


def _responses_citing_outlet_idx(domain, all_responses, pats=None):
    """Indices of the responses that cite `domain` — by a real URL OR by name.

    Counts a response when ANY of:
      - it has an extracted citation URL on that domain (existing signal), OR
      - the bare domain string appears in the text (e.g. 'wwd.com' with no
        protocol — unambiguous), OR
      - a registry name/abbreviation alias appears (allowlist-gated, so safe).
    """
    p = (pats or _outlet_patterns()).get(domain)
    idx = []
    for i, r in enumerate(all_responses):
        if any((c.get('domain') == domain) for c in (r.get('citations') or [])):
            idx.append(i)
            continue
        if not p:
            continue
        t = _norm_apostrophes(r.get('response', '') or '')
        if p['dom'].search(t) \
           or any(pp.search(t) for pp in p['ci']) \
           or any(pp.search(t) for pp in p['abbr']):
            idx.append(i)
    return idx


def _augment_citations_with_named_outlets(ranked_domains, all_responses):
    """Make named-but-unlinked outlets first-class citizens of the citation
    universe. For every allowlisted outlet that is cited by name or bare domain,
    set its count to the number of responses that name it (if higher than the
    URL-only count) and attach `_citing_idx` so share-of-voice stays consistent.

    Monotonic: never LOWERS an existing count — it only promotes known-good
    outlets the URL counter under-credited. Returns a re-sorted list.
    """
    if not all_responses:
        return ranked_domains
    pats = _outlet_patterns()
    by_dom = {d.get('domain'): d for d in ranked_domains}
    new_entries = []
    for dom in EDITORIAL_OUTLETS:
        idx = _responses_citing_outlet_idx(dom, all_responses, pats)
        if not idx:
            continue
        rc = len(idx)
        llms = sorted({all_responses[i].get('llm') for i in idx if all_responses[i].get('llm')})
        prompts = sorted({all_responses[i].get('prompt') for i in idx if all_responses[i].get('prompt')})
        entry = by_dom.get(dom)
        if entry is None:
            entry = {
                'domain': dom, 'urls': [], 'specific_urls': [],
                'count': rc, 'llms': llms, 'prompts': prompts,
                'via_name_mention': True,
            }
            by_dom[dom] = entry
            new_entries.append(entry)
        else:
            if rc > entry.get('count', 0):
                entry['count'] = rc
                entry['via_name_mention'] = True
            entry['llms'] = sorted(set(entry.get('llms') or []) | set(llms))
            entry['prompts'] = sorted(set(entry.get('prompts') or []) | set(prompts))
        entry['_citing_idx'] = idx
        entry['outlet_name'] = pats[dom]['name']
        entry['diversity_score'] = entry['count'] * (1 + 0.25 * max(0, len(entry['llms']) - 1))
    merged = list(ranked_domains) + new_entries
    merged.sort(key=lambda x: x.get('diversity_score', 0), reverse=True)
    return merged


def _count_brand_mentions(brand, all_responses, is_primary_brand=False):
    """Deterministic, case-insensitive, word-boundary count of how many responses
    mention the brand. Replaces the previous LLM-estimated count which was
    biased low because the analysis prompt only saw 500-char excerpts of each
    response (so mentions past char 500 were silently dropped).

    Multi-form heuristic WITH FALSE-POSITIVE GUARDRAIL:
    For multi-word brand names where the first word is >= 5 chars, we want to
    ALSO count responses that mention just the first word (e.g. "ACME Corp"
    being shortened to "ACME"). But this breaks badly when the first word is
    a common category word — "Beauty Blender" → matching standalone "Beauty"
    finds 29/29 in a beauty audit, not 1.

    Guardrail: only OR-combine the first-word matches when the first-word
    count is at most 2× the full-form count (or both counts are very small).
    If the first word matches far more often than the full brand name, it's
    almost certainly the category noun and we should ignore the fallback.

    Examples (full / first / final):
      - "Patagonia"            → 5 / —  / 5      (single-word; no fallback)
      - "ACME Corporation"     → 5 / 8  / OR=8   (8 ≤ 2×5=10 → use fallback)
      - "Beauty Blender"       → 1 / 29 / 1      (29 > 2×1=2 → drop fallback)
      - "The Honest Company"   → 4 / —  / 4      ('The' too short; no fallback)
      - "Iron Mountain"        → 2 / 0  / 2      ('Iron' alone never appears)
    """
    if not brand:
        return 0
    full_pat = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
    full_count = sum(
        1 for r in all_responses
        if full_pat.search(r.get('response', '') or '')
    )
    parts = brand.split()

    # SINGLE-WORD COMMON-WORD GUARDRAIL — symmetric to the multi-word case.
    # Short single-word brands ("On", "Apple", "Notion", "Tilt") collide with
    # common English words ("on the market", "apple pie", "the notion that").
    # The case-insensitive \bOn\b regex matches everywhere — "On" the brand at
    # 47/50 vs the real On Running mention count of ~17. Fix: also compute the
    # CASE-SENSITIVE count; if the lowercase form dominates the count, that's
    # the English-word collision and we should fall back to case-sensitive
    # (which matches the brand's proper-noun form only).
    #
    # Examples (CI / CS / final):
    #   - "On"      → 47 / 17 / 17   (CI > 2×CS=34 → collision; use CS)
    #   - "Tilt"    → e.g. 6 / 6 / 6 (CI == CS → no collision; CI)
    #   - "Patagonia" → 24 / 24 / 24 (always capitalized → no change)
    if len(parts) == 1 and len(brand) <= 6:
        cs_pat = re.compile(r'\b' + re.escape(brand) + r'\b')
        cs_count = sum(
            1 for r in all_responses
            if cs_pat.search(r.get('response', '') or '')
        )
        if full_count > max(cs_count * 2, 3):
            return cs_count
        # COMPETITOR-ONLY tighter guard for a short token whose case-insensitive
        # count is inflated by a generic lowercase word ("BILL" <- "bill payment",
        # "Mesh" <- "mesh network"). The 2x guard above can NEVER fire once the
        # proper-noun (CS) count is large — 34*2=68 exceeds the 50-response corpus —
        # so BILL read 43 (incl. generic "bill") and wrongly beat the audited brand.
        # When the lowercase-only surplus is material, fall back to the proper-noun
        # count. Kept OFF the audited brand (is_primary_brand) so its tuned behavior
        # is unchanged; only competitor rankings are affected.
        if (not is_primary_brand and cs_count >= 2
                and (full_count - cs_count) >= max(3, 0.15 * full_count)):
            return cs_count

    # The aggressive short-form handling (4-char first words + the proper-noun
    # override) applies ONLY to the AUDITED BRAND (is_primary_brand). Competitors
    # stay conservative — first word must be >=6 chars and pass the dominance
    # guard — so a shared parent token like "Apple" can't inflate a competitor
    # "Apple Arcade" with "Apple Music"/"Apple TV+" mentions.
    min_first = 4 if is_primary_brand else 6
    if len(parts) > 1 and len(parts[0]) >= min_first:
        first_pat = re.compile(r'\b' + re.escape(parts[0]) + r'\b', re.IGNORECASE)
        first_count = sum(
            1 for r in all_responses
            if first_pat.search(r.get('response', '') or '')
        )
        # (a) Original guard: accept the fallback when the first word doesn't
        # dominate (<= max(2×full, 3)).
        accept = first_count <= max(full_count * 2, 3)
        # (b) BRAND ONLY: accept a DOMINATING first word when it's proper-noun-
        # like — almost always Capitalized — i.e. a distinctive short form ("Hims"
        # for "Hims & Hers", 90% capitalized) not a common word used generically
        # ("Beauty" appears lowercase all over a beauty audit). Same CI-vs-CS
        # signal as the single-word guard. Capped at 4× so a standalone entity
        # can't run away. This stops "Hims & Hers" reading 10/50 when AI calls it
        # just "Hims" in 29/50.
        if is_primary_brand and not accept:
            first_cs_count = sum(
                1 for r in all_responses
                if re.compile(r'\b' + re.escape(parts[0]) + r'\b').search(r.get('response', '') or '')
            )
            proper_nounish = first_cs_count >= first_count * 0.7
            accept = proper_nounish and full_count >= 2 and first_count <= full_count * 4
        if accept:
            combined = sum(
                1 for r in all_responses
                if full_pat.search(r.get('response', '') or '') or
                   first_pat.search(r.get('response', '') or '')
            )
            return combined
    return full_count


# Generic words that show up inside brand strings but are far too common to use
# as evidence that a surfaced competitor is the brand's own alias/parent.
_GENERIC_BRAND_TOKENS = {
    'ai', 'the', 'inc', 'llc', 'ltd', 'co', 'corp', 'corporation', 'group',
    'labs', 'lab', 'app', 'apps', 'platform', 'platforms', 'software', 'solution',
    'solutions', 'technologies', 'technology', 'tech', 'systems', 'system', 'io',
    'com', 'net', 'ultimate', 'global', 'digital', 'cloud', 'company', 'agent',
    'agents', 'assistant', 'team', 'studio', 'works', 'data', 'health',
}


def _split_self_referential_competitors(brand, competitors):
    """Partition competitors into (real_competitors, related_brands). A 'related
    brand' is a surfaced competitor that the brand string LITERALLY NAMES — i.e.
    the brand's own alias / parent / acquirer (e.g. a competitor 'Zendesk' when
    the brand entered is 'Ultimate (Zendesk Ultimate)').

    WHY: a brand must never be framed as 'losing to' its own parent. Observed on a
    👎'd audit — Ultimate.ai (acquired by Zendesk) scored 0% and the report said it
    trailed 'competitor' Zendesk 0-to-66, which reads as nonsense to anyone who
    knows the brand. Pulling the parent out of the competitive set fixes the
    head-to-head everywhere (mindshare framing, SoV, headline) at once.

    MATCH RULE (deliberately strict): the competitor's FULL name must appear as a
    word-boundary substring of the brand string, be >= 4 chars, and not be a
    generic token. Word-OVERLAP is NOT enough — otherwise peer orgs that merely
    share a common word get wrongly pulled (e.g. 'Black Girls Code' / 'Women Who
    Code' vs the brand 'Girls Who Code', which share 'girls'/'code'/'who').
    """
    if not brand or not competitors:
        return list(competitors or []), []
    bl = re.sub(r'\s+', ' ', brand.lower())
    real, related = [], []
    for c in competitors:
        nm = re.sub(r'\s+', ' ', (c.get('name') or '').lower().strip())
        if (len(nm) >= 4 and nm not in _GENERIC_BRAND_TOKENS
                and re.search(r'(?<![a-z0-9])' + re.escape(nm) + r'(?![a-z0-9])', bl)):
            related.append(c)
        else:
            real.append(c)
    return real, related


def _dedupe_competitors(competitor_counts, all_responses=None):
    """Merge competitor entries that are the SAME entity expressed two ways —
    e.g. 'JPMorgan' and 'J.P. Morgan Asset Management' (observed on a BlackRock
    audit, listed as two rivals). Two names merge when one's alphanumeric-
    normalized form contains the other's (min length 4, to avoid merging on a
    tiny shared fragment). The longer / higher-count name is kept as canonical;
    the merged mention_count is recounted as responses matching EITHER stored
    form (so we never double-count a response that used both). Order preserved
    by descending count."""
    if not competitor_counts:
        return competitor_counts
    def norm(n):
        return re.sub(r'[^a-z0-9]', '', (n or '').lower())
    items = list(competitor_counts)
    used = [False] * len(items)
    out = []
    for i in range(len(items)):
        if used[i]:
            continue
        ni = norm(items[i].get('name'))
        group = [items[i]]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            nj = norm(items[j].get('name'))
            if ni and nj and min(len(ni), len(nj)) >= 4 and (ni in nj or nj in ni):
                group.append(items[j])
                used[j] = True
        if len(group) == 1:
            out.append(items[i])
            continue
        canon = max(group, key=lambda m: ((m.get('mention_count') or 0), len(m.get('name') or '')))
        merged = dict(canon)
        if all_responses:
            pats = [re.compile(r'\b' + re.escape(m.get('name') or '') + r'\b', re.IGNORECASE)
                    for m in group if m.get('name')]
            merged['mention_count'] = sum(
                1 for r in all_responses
                if any(p.search(r.get('response', '') or '') for p in pats))
        else:
            merged['mention_count'] = max((m.get('mention_count') or 0) for m in group)
        names = ', '.join(m.get('name') for m in group if m.get('name'))
        print(f"_dedupe_competitors: merged [{names}] -> '{merged.get('name')}' ({merged['mention_count']})")
        out.append(merged)
    out.sort(key=lambda c: c.get('mention_count') or 0, reverse=True)
    return out


# Component / chip / hardware SUPPLIERS that routinely get extracted as top
# "competitors" for a data-center / cloud / AI-infrastructure OPERATOR — but the
# operator BUYS from them, it doesn't compete with them (observed: NVIDIA #2 and
# AMD top-10 "competitors" to IREN, an AI data-center company). Matched
# case-insensitively against the exact competitor name.
_COMPONENT_SUPPLIERS = {
    'nvidia', 'amd', 'intel', 'tsmc', 'arm', 'arm holdings', 'broadcom',
    'micron', 'sk hynix', 'supermicro', 'super micro', 'super micro computer',
}


def _drop_supplier_noncompetitors(brand, category, competitor_counts):
    """Remove component/chip suppliers from the competitive set of an infra/cloud/
    data-center operator — they're suppliers, not competitors. Kept UNLESS the
    audited brand is itself a chip/hardware maker (then they ARE peers), and only
    applied in infrastructure-style categories where the confusion happens."""
    if not competitor_counts:
        return competitor_counts
    cat = (category or '').lower()
    brand_l = (brand or '').strip().lower()
    # Brand is itself a hardware/chip maker -> suppliers are legitimate peers.
    if brand_l in _COMPONENT_SUPPLIERS or any(
            w in cat for w in ('chip', 'semiconductor', 'processor', 'silicon',
                               'hardware maker', 'gpu maker', 'chipmaker')):
        return competitor_counts
    # Only prune in infra / cloud / data-center / compute contexts.
    if not any(w in cat for w in ('data center', 'data centre', 'data-center', 'datacenter',
                                  'cloud', 'infrastructure', 'compute', 'hosting',
                                  'colocation', 'colo', 'gpu')):
        return competitor_counts
    kept, dropped = [], []
    for c in competitor_counts:
        (dropped if (c.get('name') or '').strip().lower() in _COMPONENT_SUPPLIERS else kept).append(c)
    if dropped:
        print(f"_drop_supplier_noncompetitors: removed {[c.get('name') for c in dropped]} "
              f"(supplier, not competitor; brand={brand})")
    return kept


def _classify_competitor_types(brand, category, competitor_counts):
    """Classify each extracted competitor as a brand peer, retailer, or
    marketplace so the report treats them differently.

    Why: when an audit's category mixes brands and channels (e.g. "athletic
    retailer for men's running"), the AI surfaces BOTH brand peers (Nike,
    Tracksmith) and multi-brand retailers (Running Warehouse, REI). Framing a
    single-brand company like Lululemon as "trailing Running Warehouse" is
    apples-to-oranges — Running Warehouse SELLS Lululemon and Nike alongside
    each other, it doesn't compete with them. PR competitive analysis needs
    only the brand peers.

    Adds a 'type' field to each competitor dict:
      'brand_peer'  — direct brand/DTC competitor (Nike, Tracksmith)
      'retailer'    — multi-brand retail channel (Running Warehouse, REI)
      'marketplace' — general marketplace (Amazon, eBay)
    On any failure, defaults all to 'brand_peer' (fail-safe — keeps current
    behaviour of treating everyone as competitors).
    """
    if not competitor_counts:
        return competitor_counts
    names = [c.get('name') for c in competitor_counts if c.get('name')]
    if not names:
        return competitor_counts
    try:
        prompt = (
            f'You are classifying competitors for a PR competitive audit of "{brand}" '
            f'in the category: "{category or "general"}".\n\n'
            f'For each name, classify as exactly one of:\n'
            f'- brand_peer: a direct BRAND competitor that makes or sells its own products '
            f'in the same category (e.g. for Lululemon: Nike, Adidas, Tracksmith, Patagonia). '
            f'Use this for brands and DTC vendors.\n'
            f'- retailer: a multi-brand retail channel that distributes other companies\' '
            f'products (e.g. Running Warehouse, REI, Dick\'s Sporting Goods, Sephora, '
            f'Fleet Feet, Road Runner Sports, Best Buy, Nordstrom). They sell brands, '
            f'they don\'t compete with them.\n'
            f'- marketplace: a general marketplace (Amazon, eBay, Etsy, Walmart).\n\n'
            f'Names:\n' + '\n'.join(f'- {n}' for n in names) + '\n\n'
            f'Return ONLY a JSON object mapping name -> type. Use the EXACT names from '
            f'the list. No prose, no markdown, just the JSON.'
        )
        resp = anthropic.messages.create(
            model=CLAUDE_SONNET, max_tokens=600, timeout=30.0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.content[0].text or "").strip()
        m = re.search(r'\{.*\}', txt, re.DOTALL)
        types = json.loads(m.group(0) if m else txt)
        n_retail = n_market = n_peer = 0
        for c in competitor_counts:
            t = (types.get(c.get('name')) or 'brand_peer').strip().lower()
            if t not in ('brand_peer', 'retailer', 'marketplace'):
                t = 'brand_peer'
            c['type'] = t
            if t == 'retailer': n_retail += 1
            elif t == 'marketplace': n_market += 1
            else: n_peer += 1
        print(f"competitor type classification: peers={n_peer} retailers={n_retail} marketplaces={n_market}")
    except Exception as e:
        print("competitor type classification failed (defaulting all to brand_peer):", str(e)[:160])
        for c in competitor_counts:
            c.setdefault('type', 'brand_peer')
    return competitor_counts


def _extract_competitor_candidates(brand, category, all_responses, max_candidates=15):
    """Ask Claude to extract a CANDIDATE LIST of competitor brand names from
    the actual response text. Returns just names — counts come from
    _count_brand_mentions (deterministic, authoritative).

    This decouples competitor *discovery* (LLM, judgement call) from competitor
    *frequency* (deterministic substring match over full response text). Before
    this split, Claude was asked to estimate counts from 500-char excerpts in
    the analysis prompt, which produced numbers a consultant could trivially
    falsify by grepping the raw responses for "Spanx" etc. Now the counts always
    match what a regex over the full text would produce.
    """
    if not all_responses:
        return []
    excerpts = ""
    for i, r in enumerate(all_responses[:30]):  # cap context size; we have ≤30 responses for free, more for paid
        text = (r.get('response', '') or '')[:2000]
        excerpts += f"\n--- Response {i+1} [{r.get('llm','?')}] ---\n{text}\n"

    try:
        resp = anthropic.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": f"""Extract the brand/product/company names that appear as COMPETITORS or PEER OPTIONS to "{brand}" in the category "{category}" across the AI responses below.

Rules:
- Return ONLY actual brand/product names — not generic categories (e.g. "wireless headphones" is not a brand).
- Exclude "{brand}" itself.
- Exclude reviewer/journalist names, publication names, and platform names (Amazon, Reddit, etc.).
- Maximum {max_candidates} candidates. Prefer the most-mentioned distinct brands.
- Keep names in their canonical form (e.g. "Sony WH-1000XM5" → "Sony", "Bose QuietComfort" → "Bose"). For brand families, use the umbrella brand, not the SKU.
- If no clear competitor brands appear, return an empty list.

Respond with ONLY a JSON array of strings: ["Brand A", "Brand B", ...]

RESPONSES:
{excerpts}"""
            }]
        )
        txt = resp.content[0].text
        match = re.search(r'\[.*?\]', txt, re.DOTALL)
        if not match:
            return []
        names = json.loads(match.group())
        if not isinstance(names, list):
            return []
        # Sanitize: strings only, dedupe case-insensitively, drop the brand itself.
        seen = set()
        out = []
        brand_lower = (brand or '').strip().lower()
        for n in names:
            if not isinstance(n, str):
                continue
            n = n.strip()
            if not n or n.lower() == brand_lower:
                continue
            key = n.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
            if len(out) >= max_candidates:
                break
        return out
    except Exception as e:
        print("competitor candidate extraction failed (continuing without):", e)
        return []


def _compute_outlet_share_of_voice(brand, competitor_counts, all_responses, editorial_domains, max_outlets=10, brand_aliases=None):
    """For each top EDITORIAL MEDIA TARGET, compute the brand's and each
    competitor's mention rate WITHIN the subset of responses that cite that
    outlet, and compare to each brand's overall mention rate across all
    responses. Identifies outlets where the brand over-indexes (STRENGTH —
    defend/amplify) vs where competitors over-index (OPPORTUNITY — pitch).

    IMPORTANT — caller must pass the EDITORIAL-ONLY list, not all ranked
    domains. PR pitching opportunities + strengths only apply to editorial
    publications (Allure, The Mighty, Beauty Independent, etc.) — they don't
    apply to .gov pages, .edu pages, the brand's own domain, competitor own-
    domains, trade-association sites, B2B vendor portals, etc. Pass
    `editorial_domains` (the post-classify + post-self-domain + post-
    verify_editorial_domains list) so the SoV section stays aligned with
    the media_targets list in the rest of the report.

    Returns a list of dicts, one per outlet (capped at `max_outlets`), each:
      {
        domain: 'forbes.com',
        responses_citing: 7,         # number of responses that cited this outlet
        brand_mention_count: 12,     # brand mentioned in N responses overall
        brand_overall_sov: 0.40,     # 12 of 30 overall responses mention brand
        brand_mentions_at_outlet: 5, # brand mentioned in 5 of the 7 outlet-citing responses
        brand_sov_at_outlet: 0.71,   # 5 / 7
        brand_sov_differential: 0.31,# at_outlet - overall (positive = over-indexes)
        verdict: 'strength' | 'opportunity' | 'neutral',
        verdict_label: 'Strength — brand over-indexes...',
        top_competitor_at_outlet: {
            name, mentions_at_outlet, sov_at_outlet, overall_sov, differential
        } | None,
      }

    Sorted: strengths first (highest differential), then opportunities (most
    negative differential / strongest competitor lead), then neutrals.
    Strength threshold: brand SoV at outlet exceeds overall by >= 0.15 AND
      brand mention rate at outlet >= top competitor rate at outlet.
    Opportunity threshold: top competitor SoV at outlet exceeds brand SoV at
      outlet by >= 0.15, OR (brand absent at outlet AND competitor present
      with >= 2 responses).
    """
    total_responses = len(all_responses)
    if total_responses == 0 or not editorial_domains:
        return []

    # Pre-build mention-detection patterns for brand + each competitor.
    # Match the same multi-form heuristic + first-word-dominance guardrail
    # as _count_brand_mentions (see that function for examples). For each
    # name we return EITHER [full_pat] only, or [full_pat, first_pat]
    # depending on whether the first-word fallback survives the guardrail
    # against the response corpus. This is computed once per name so SoV
    # per-outlet counts don't keep re-evaluating the guardrail.
    def _patterns_for(name, is_primary=False):
        if not name:
            return []
        full_pat = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
        parts = name.split()

        # Single-word common-word guardrail (same logic as _count_brand_mentions):
        # short single-word brands ("On", "Apple", "Notion") collide with the
        # English word. If the case-insensitive match count dominates the
        # case-sensitive one, swap to the proper-noun-only case-sensitive pattern.
        if len(parts) == 1 and len(name) <= 6:
            cs_pat = re.compile(r'\b' + re.escape(name) + r'\b')
            ci_count = sum(
                1 for r in all_responses
                if full_pat.search(r.get('response', '') or '')
            )
            cs_count = sum(
                1 for r in all_responses
                if cs_pat.search(r.get('response', '') or '')
            )
            if ci_count > max(cs_count * 2, 3):
                return [cs_pat]

        # BRAND ONLY gets the 4-char first-word handling; competitors stay >=6 +
        # dominance-only, so a shared parent token ("Apple" in "Apple Arcade")
        # can't inflate a competitor. Mirrors _count_brand_mentions.
        min_first = 4 if is_primary else 6
        if not (len(parts) > 1 and len(parts[0]) >= min_first):
            return [full_pat]
        first_pat = re.compile(r'\b' + re.escape(parts[0]) + r'\b', re.IGNORECASE)
        full_count = sum(
            1 for r in all_responses
            if full_pat.search(r.get('response', '') or '')
        )
        first_count = sum(
            1 for r in all_responses
            if first_pat.search(r.get('response', '') or '')
        )
        if first_count > max(full_count * 2, 3):
            # First word dominates. Drop it — UNLESS this is the audited brand and
            # the first word is a distinctive proper-noun short form (mostly
            # Capitalized, like "Hims"), capped at 4× so a standalone entity can't
            # run. Keeping it stops per-outlet SoV (and the headline) undercounting
            # the brand.
            if not is_primary:
                return [full_pat]
            first_cs_count = sum(
                1 for r in all_responses
                if re.compile(r'\b' + re.escape(parts[0]) + r'\b').search(r.get('response', '') or '')
            )
            proper_nounish = first_cs_count >= first_count * 0.7
            if not (proper_nounish and full_count >= 2 and first_count <= full_count * 4):
                return [full_pat]
        return [full_pat, first_pat]

    brand_patterns = _patterns_for(brand or '', is_primary=True)
    for _al in (brand_aliases or []):     # fold sub-brand aliases (e.g. iShares) into the brand's SoV
        brand_patterns = brand_patterns + _patterns_for(_al, is_primary=True)
    competitor_pat_map = {
        c['name']: _patterns_for(c['name'])
        for c in (competitor_counts or [])
    }
    competitor_overall_sov = {
        c['name']: (c['mention_count'] / total_responses) if total_responses else 0.0
        for c in (competitor_counts or [])
    }

    brand_overall_count = sum(
        1 for r in all_responses
        if any(p.search(r.get('response', '') or '') for p in brand_patterns)
    ) if brand_patterns else 0
    brand_overall_sov = (brand_overall_count / total_responses) if total_responses else 0.0

    # Skip outlets that are a competitor's own site. There's no meaningful
    # "opportunity" to pitch IBM's coverage to ibm.com — the LLM is just
    # citing the competitor's own product page. Uses the shared module-level
    # stem logic (multi-word + accent-aware) so this stays in sync with the
    # editorial-list filter in run_citation_audit.
    competitor_domain_stems = _competitor_domain_stems(competitor_counts)

    def _domain_owned_by_competitor(domain):
        return _is_competitor_owned_domain(domain, competitor_domain_stems)

    # For each cited domain, find the responses that mentioned it (any URL on
    # that domain in `r['citations']`), then count brand + competitor hits in
    # that subset.
    # Input is already filtered to editorial-only by the caller, so we walk
    # the list as-is. Cap at max_outlets to bound the SoV table size.
    out = []
    for d in editorial_domains[:max_outlets * 2]:  # pull 2x then filter competitor-owned
        domain = d.get('domain')
        if not domain:
            continue
        if _domain_owned_by_competitor(domain):
            continue  # see comment above; not a pitchable outlet
        # Prefer the augmented citing-response set (URL OR name) attached by
        # _augment_citations_with_named_outlets, so SoV agrees with the count
        # shown on the card. Fall back to URL-only match for domains the
        # registry doesn't know (unchanged behaviour).
        citing_idx = d.get('_citing_idx')
        if citing_idx is not None:
            citing_responses = [all_responses[i] for i in citing_idx]
        else:
            citing_responses = [
                r for r in all_responses
                if any((c.get('domain') == domain) for c in (r.get('citations') or []))
            ]
        n_at_outlet = len(citing_responses)
        if n_at_outlet < 1:
            continue  # outlet appears in no responses (shouldn't happen given upstream filter)

        brand_at = sum(
            1 for r in citing_responses
            if any(p.search(r.get('response', '') or '') for p in brand_patterns)
        ) if brand_patterns else 0

        # Scan competitors first so both n=1 and n>=2 paths can surface the
        # same all_competitors_at_outlet list. UI uses this to render a bar
        # per competitor below the brand bar.
        emerging_comp_rows = []
        for cname, pats in competitor_pat_map.items():
            if not pats:
                continue
            cnt = sum(
                1 for r in citing_responses
                if any(p.search(r.get('response', '') or '') for p in pats)
            )
            if cnt > 0:
                emerging_comp_rows.append({
                    "name": cname,
                    "mentions_at_outlet": cnt,
                    "sov_at_outlet": round(cnt / n_at_outlet, 3),
                    "overall_sov": round(competitor_overall_sov.get(cname, 0.0), 3),
                    "differential": round((cnt / n_at_outlet) - competitor_overall_sov.get(cname, 0.0), 3),
                })
        emerging_comp_rows.sort(key=lambda x: x["sov_at_outlet"], reverse=True)

        # n=1 case: emit an 'emerging' row but skip the strength/opportunity
        # verdict math (degenerate at this sample size). Still surface which
        # competitors were mentioned alongside this single citation — that's
        # actionable intel even at n=1.
        if n_at_outlet == 1:
            also_mentioned = [c["name"] for c in emerging_comp_rows]
            if brand_at:
                core = f"{brand} appeared in that response"
            else:
                core = f"{brand} did not appear in that response"
            if also_mentioned:
                also_phrase = ", ".join(also_mentioned[:5])
                core += f"; mentioned alongside: {also_phrase}."
            else:
                core += "."
            label = (
                f"Emerging mention — this outlet was cited just once. {core} "
                f"Too thin for share-of-voice analysis; worth investigating as a relationship to develop."
            )
            out.append({
                "domain": domain,
                "responses_citing": n_at_outlet,
                "brand_overall_sov": round(brand_overall_sov, 3),
                "brand_mentions_at_outlet": brand_at,
                "brand_sov_at_outlet": float(brand_at),  # 0.0 or 1.0
                "brand_sov_differential": 0.0,
                "verdict": "emerging",
                "verdict_label": label,
                "top_competitor_at_outlet": (emerging_comp_rows[0] if emerging_comp_rows else None),
                "all_competitors_at_outlet": emerging_comp_rows[:5],
            })
            continue

        brand_sov_at = brand_at / n_at_outlet
        brand_diff = brand_sov_at - brand_overall_sov

        # Build per-competitor SoV at this outlet. comp_rows ends up sorted
        # by SoV desc (== mentions desc, since n_at_outlet is constant).
        comp_rows = []
        for cname, pats in competitor_pat_map.items():
            if not pats:
                continue
            cnt = sum(
                1 for r in citing_responses
                if any(p.search(r.get('response', '') or '') for p in pats)
            )
            if cnt > 0:
                comp_rows.append({
                    "name": cname,
                    "mentions_at_outlet": cnt,
                    "sov_at_outlet": round(cnt / n_at_outlet, 3),
                    "overall_sov": round(competitor_overall_sov.get(cname, 0.0), 3),
                    "differential": round((cnt / n_at_outlet) - competitor_overall_sov.get(cname, 0.0), 3),
                })
        comp_rows.sort(key=lambda x: x["sov_at_outlet"], reverse=True)

        # Verdict detection uses the HIGHEST-SoV competitor (the leader/threat)
        # to decide whether brand is being out-cited at this outlet. Display
        # uses the CLOSEST-SoV competitor (smallest absolute distance from
        # brand's own SoV at this outlet) which produces the most actionable
        # head-to-head readout:
        #   - opportunity card → the leader you need to displace usually IS
        #     also the closest by SoV (since they're winning by definition)
        #   - strength card → the nearest challenger from below
        #   - tied / neutral case → whoever is at the same level
        # When two competitors are equidistant, prefer the one with higher SoV
        # (the more present competitor).
        leader_comp = comp_rows[0] if comp_rows else None  # used for verdict math
        if comp_rows:
            top_comp = min(
                comp_rows,
                key=lambda c: (abs(c["sov_at_outlet"] - brand_sov_at), -c["sov_at_outlet"]),
            )
        else:
            top_comp = None

        # Verdict logic — switches between two modes based on brand's overall
        # share of voice across the response set. The relative-delta thresholds
        # (which work fine for "normal" brands at 20-60% baseline) collapse on
        # dominant brands (Patagonia at 97% overall has no headroom for +15pp
        # strength and no competitor can plausibly catch up to opportunity-trigger
        # +15pp ahead). For those brands we switch to absolute floors so the
        # section still surfaces meaningful insight.
        STRENGTH_PP = 0.15
        OPPORTUNITY_PP = 0.15
        # Defend-class lead: leading the competitive field at an outlet cited in
        # at least this many responses is a position to DEFEND even without a big
        # over-index vs the brand's own baseline (see standard-mode block below).
        LEAD_MIN_RESPONSES = 5
        DOMINANT_BRAND_THRESHOLD = 0.70  # overall SoV at which we switch modes
        DOMINANT_STRENGTH_FLOOR = 0.95   # in dominant mode, strength = brand at >= 95%
        DOMINANT_OPPORTUNITY_DELTA = 0.10  # in dominant mode, opportunity = brand drops 10pp below baseline
        # The 10pp opportunity floor for dominant brands is intentionally looser
        # than the 15pp delta for normal-mode brands. At a 97% baseline, even a
        # 10pp slip (e.g. 83% at an outlet) is signal — the brand is essentially
        # ubiquitous everywhere except a few specific places, and those places
        # are where targeted earned media moves the needle the most.
        # Contested co-lead: an outlet where the brand is a strong named voice
        # (>= floor of its answers) and over-indexes vs its own baseline, but a
        # rival narrowly edges it. Relative-only logic buries this as 'neutral';
        # it's really a strength to DEFEND while closing a small gap.
        CONTESTED_ABS_FLOOR = 0.65     # named in >= 65% of the outlet's answers
        CONTESTED_OVERINDEX_PP = 0.05  # and at least 5pp above the brand's own baseline
        CONTESTED_GAP_PP = 0.15        # and within 15pp of the leader
        verdict = "neutral"
        verdict_label = (
            f"Aligned — {brand} performs in line with its overall AI mindshare here. "
            f"Steady coverage; no urgent action needed."
        )

        is_dominant_brand = brand_patterns and brand_overall_sov >= DOMINANT_BRAND_THRESHOLD

        if is_dominant_brand:
            # Saturation mode: brand already wins almost everywhere. Strength =
            # totally saturating this outlet (everyone citing it mentions the
            # brand). Opportunity = brand's mention rate slips noticeably below
            # its overall baseline, i.e. this outlet is one of the few where it
            # under-performs.
            if brand_sov_at >= DOMINANT_STRENGTH_FLOOR:
                verdict = "strength"
                verdict_label = (
                    f"{brand} dominates this outlet — defend the relationship and pitch "
                    f"follow-up coverage to extend the lead."
                )
            elif (brand_overall_sov - brand_sov_at) >= DOMINANT_OPPORTUNITY_DELTA:
                verdict = "opportunity"
                # For opportunity show the leader (whom to displace), not
                # the closest-distance competitor.
                if leader_comp:
                    top_comp = leader_comp
                verdict_label = (
                    f"{brand} under-cited here vs overall mindshare. Pitch to close the gap."
                )
        else:
            # Standard mode: relative-delta verdicts. Use leader_comp (highest
            # SoV) for the math so opportunity-detection responds to the actual
            # leader, not whichever competitor happens to be closest by SoV.
            comp_sov = leader_comp["sov_at_outlet"] if leader_comp else 0.0
            comp_at = leader_comp["mentions_at_outlet"] if leader_comp else 0
            if brand_patterns and brand_diff >= STRENGTH_PP and (not leader_comp or brand_sov_at >= comp_sov):
                verdict = "strength"
                verdict_label = (
                    f"{brand} over-indexes here. Defend the relationship; "
                    f"pitch follow-up coverage."
                )
            # DEFEND-CLASS LEAD: the brand is the outright most-cited brand at a
            # materially-cited outlet, even without a big over-index vs its OWN
            # baseline. Leading the field at a top outlet is the textbook "punch
            # above your weight" signal — a position to DEFEND, not the
            # "neutral / no action" the relative-delta test would bury it as.
            # (Fixes: a brand ahead of the category leader at the single
            # most-cited outlet rendering as "steady; no urgent action".)
            elif (brand_patterns and brand_at >= 2 and n_at_outlet >= LEAD_MIN_RESPONSES
                  and brand_at > comp_at):
                verdict = "strength"
                if leader_comp and (brand_sov_at - comp_sov) < 0.10:
                    # Narrow lead — honest about how close the threat is.
                    verdict_label = (
                        f"{brand} is the most-cited brand here ({brand_at} of {n_at_outlet} "
                        f"responses) but {leader_comp['name']} is right behind ({comp_at}) — "
                        f"a contested lead at a top outlet. Defend it before it flips."
                    )
                elif leader_comp:
                    verdict_label = (
                        f"{brand} leads the field here ({brand_at} of {n_at_outlet} responses, "
                        f"ahead of {leader_comp['name']} at {comp_at}) — defend and extend this "
                        f"lead at a top outlet."
                    )
                else:
                    verdict_label = (
                        f"{brand} owns this outlet ({brand_at} of {n_at_outlet} responses) with "
                        f"no competitor present — lock it in with follow-up coverage."
                    )
            # CONTESTED CO-LEAD: strong, over-indexing presence at an influential
            # outlet where a rival is only narrowly ahead — a strength to defend,
            # not the 'neutral' the lead-only test would assign. (e.g. brand named
            # in 80% of an outlet's answers, +16pp over its baseline, leader at 93%.)
            elif (brand_patterns and n_at_outlet >= LEAD_MIN_RESPONSES
                  and brand_sov_at >= CONTESTED_ABS_FLOOR
                  and brand_diff >= CONTESTED_OVERINDEX_PP
                  and leader_comp and (comp_sov - brand_sov_at) <= CONTESTED_GAP_PP):
                verdict = "strength"
                verdict_label = (
                    f"{brand} is a dominant named voice here ({brand_at} of {n_at_outlet} "
                    f"responses, {brand_sov_at:.0%}), over-indexing vs its overall mindshare — "
                    f"{leader_comp['name']} is only narrowly ahead. Defend this relationship "
                    f"while you close the gap."
                )
            elif leader_comp:
                comp_lead = comp_sov - brand_sov_at
                if comp_lead >= OPPORTUNITY_PP or (brand_at == 0 and leader_comp["mentions_at_outlet"] >= 2):
                    verdict = "opportunity"
                    # For opportunity display the LEADER (the one to displace),
                    # not the closest-distance competitor.
                    top_comp = leader_comp
                    if brand_at == 0:
                        verdict_label = (
                            f"{leader_comp['name']} owns this outlet; {brand} is absent. "
                            f"Pitch a story angle that displaces them."
                        )
                    else:
                        verdict_label = (
                            f"{leader_comp['name']} out-cites {brand} here. "
                            f"Close the gap with earned coverage."
                        )

            # Neck-and-neck at a TOP outlet — still 'neutral' (neither side
            # clearly leads), but "steady; no urgent action" badly undersells a
            # dead heat with a rival at one of the most-cited outlets. Relabel so
            # the reader sees it's contested and winnable. Reference top_comp —
            # the same head-to-head rival the card displays — so the label and
            # the card agree. (Observed: Palisociety 7-7 with its nearest rival
            # at Travel+Leisure, n=21, read as "no action".)
            if (verdict == "neutral" and brand_patterns and brand_at > 0
                    and top_comp and n_at_outlet >= LEAD_MIN_RESPONSES
                    and abs(brand_sov_at - (top_comp.get('sov_at_outlet') or 0)) <= 0.06):
                _tc_at = top_comp.get('mentions_at_outlet') or 0
                if brand_at >= _tc_at:
                    verdict_label = (
                        f"{brand} is neck-and-neck with {top_comp['name']} here "
                        f"({brand_at} vs {_tc_at} of {n_at_outlet} responses) at a top outlet "
                        f"— no one owns it yet; a sharper story can tip it your way."
                    )
                else:
                    verdict_label = (
                        f"{brand} sits just behind {top_comp['name']} here "
                        f"({brand_at} vs {_tc_at} of {n_at_outlet} responses) at a top outlet "
                        f"— close and winnable; a sharper story can pull you ahead."
                    )

        out.append({
            "domain": domain,
            "responses_citing": n_at_outlet,
            "brand_overall_sov": round(brand_overall_sov, 3),
            "brand_mentions_at_outlet": brand_at,
            "brand_sov_at_outlet": round(brand_sov_at, 3),
            "brand_sov_differential": round(brand_diff, 3),
            "verdict": verdict,
            "verdict_label": verdict_label,
            "top_competitor_at_outlet": top_comp,
            "all_competitors_at_outlet": comp_rows[:5],
        })

    # Sort: strengths (largest positive differential first), opportunities
    # (largest competitor lead first — i.e. most-negative brand differential),
    # neutrals last.
    def _sort_key(row):
        v = row["verdict"]
        if v == "strength":
            return (0, -row["brand_sov_differential"])
        if v == "opportunity":
            return (1, row["brand_sov_differential"])
        return (2, -row["responses_citing"])
    out.sort(key=_sort_key)
    # Cap at the originally-requested max_outlets after filtering.
    return out[:max_outlets]


# Marquee outlets a PR pro prioritizes as NAMED targets in a multi-outlet "Build"
# move — established publications with real newsrooms. Used ONLY to rank the named
# expansion targets (major trades ahead of niche newsletters / blogs); it does not
# gate which outlets become cards. Citation volume breaks ties within this set.
MAJOR_TRADE_OUTLETS = {
    # deep-tech / telecom / 5G / robotics trade press
    'lightreading.com', 'rcrwireless.com', 'fierce-network.com', 'fiercewireless.com',
    'telecoms.com', 'telecomtv.com', 'mobileworldlive.com', 'sdxcentral.com',
    'therobotreport.com', 'spectrum.ieee.org', 'thenewstack.io',
    # mainstream tech / business / general press
    'techradar.com', 'technologyreview.com', 'techcrunch.com', 'theverge.com',
    'wired.com', 'arstechnica.com', 'zdnet.com', 'venturebeat.com', 'engadget.com',
    'cnet.com', 'reuters.com', 'bloomberg.com', 'wsj.com', 'nytimes.com',
    'forbes.com', 'fortune.com', 'cnbc.com', 'axios.com', 'ft.com',
    'businessinsider.com', 'theinformation.com',
}


def _compute_headline_move(brand, outlet_sov, media_targets=None):
    """Pick THE single highest-priority action from the outlet share-of-voice
    data and return a {verb, outlet, text} dict (or None).

    Every dashboard — even an all-strength 'you dominate everywhere' one —
    needs ONE unmistakable next step so the reader isn't left asking 'so what
    do I do?'. Priority order:
      1. OPPORTUNITY with the biggest competitor lead → pitch to close the gap.
      2. (no opportunities) the most-CONTESTED strength — your strongest
         position where a competitor is closest → defend this first.
      3. (uncontested strengths only) your single strongest outlet → lock it in.
      4. (all emerging / thin) the most-cited outlet you're not established at
         → cultivate it early.
    Deterministic, no API call — works on the rerender path too.

    When `media_targets` is supplied, the candidate pool is constrained to
    outlets that actually appear as cards on the page. Without this filter the
    headline can point at an outlet that got capped out of media_targets (or
    one that's vendor noise lurking in outlet_sov), leaving the reader with a
    "Pitch X" action and no corresponding card to learn more about X.
    """
    sov = outlet_sov or []
    if not sov:
        return None
    if media_targets:
        _target_doms = {(t.get('domain') or '').lower() for t in media_targets if t.get('domain')}
        if _target_doms:
            sov = [r for r in sov if (r.get('domain') or '').lower() in _target_doms]
            if not sov:
                return None
    b = brand or "Your brand"

    def _n(r):
        return r.get('responses_citing') or 0

    def _brand_at(r):
        return r.get('brand_mentions_at_outlet') or 0

    opportunities = [r for r in sov if r.get('verdict') == 'opportunity']
    strengths = [r for r in sov if r.get('verdict') == 'strength']
    emerging = [r for r in sov if r.get('verdict') == 'emerging']

    # THE #1 MOVE is at the outlet where AI visibility is most at stake AND there
    # is a clear action: the most-cited OPPORTUNITY (whitespace to pitch) or
    # CONTESTED STRENGTH (a lead a competitor is actively closing). A SECURE
    # strength (wide lead / no rival) isn't urgent, so it never preempts a pitch
    # — it only becomes the move when there's nothing to pitch or defend under
    # threat (handled in the fallback below). Ranked by CITATION VOLUME first:
    # the most-cited outlet is where a move shifts AI mindshare most, and at n<=2
    # a one-mention competitor "lead" is noise. (Earlier versions ALWAYS led with
    # an opportunity, which buried "you lead the single most-cited outlet but a
    # rival is one mention behind" — frequently the bigger story.)
    def _comp_sov(r):
        return (r.get('top_competitor_at_outlet') or {}).get('sov_at_outlet') or 0

    def _is_contested_strength(r):
        tc = r.get('top_competitor_at_outlet') or {}
        return bool(tc.get('name')) and ((r.get('brand_sov_at_outlet') or 0) - _comp_sov(r)) < 0.12

    def _is_crowded_tie(r):
        """A 'best X' vendor listicle where AI just names the whole field — 2+
        competitors tie or beat the brand at the same outlet. Co-leading such an
        outlet isn't a defensible position: there's nothing distinctive to defend
        (everyone appears), so it must not drive the headline 'move'. This is what
        stops 'Defend MarkTechPost' when ServiceNow, Microsoft, Google and UiPath
        are all 9-of-10 in the same roundup."""
        ba = _brand_at(r)
        if ba <= 0:
            return False
        allc = r.get('all_competitors_at_outlet') or []
        return sum(1 for c in allc if (c.get('mentions_at_outlet') or 0) >= ba) >= 2

    contested_strengths = [r for r in strengths
                           if _is_contested_strength(r) and not _is_crowded_tie(r)]

    # MAINTAIN — a broadly-present brand with NO material lever (no opportunity
    # worth pitching, no genuine lead worth defending) and NO competitor running
    # away with the category is best served by an honest "hold", not a forced
    # thin "Pitch X" off a 2-mention gap in 5 responses. (Observed: ServiceNow,
    # already top-tier and neck-and-neck with the field, was told to pitch IT Pro
    # on a 4-vs-2 gap in 5 responses once its only "strengths" were whole-field
    # listicle ties. A material lever = an opportunity with a real gap (>=3) at a
    # real-volume outlet (n>=8), or a genuine non-listicle lead to defend at n>=8.)
    def _gap(r):
        tc = r.get('top_competitor_at_outlet') or {}
        return (tc.get('mentions_at_outlet') or 0) - _brand_at(r)
    _material_move = any(
        (r.get('verdict') == 'opportunity' and _n(r) >= 8 and _gap(r) >= 3)
        or (r.get('verdict') == 'strength' and _n(r) >= 8 and not _is_crowded_tie(r))
        for r in sov)
    _brand_overall = max((r.get('brand_overall_sov') or 0) for r in sov)
    _comp_overall = max((c.get('overall_sov') or 0
                         for r in sov for c in (r.get('all_competitors_at_outlet') or [])),
                        default=0)
    _clear_leader = (_comp_overall - _brand_overall) >= 0.15  # a rival to chase
    # "Maintain" requires an ACTUAL verified lead/tie somewhere — i.e. at least
    # one surviving 'strength' verdict. A brand that leads NOWHERE (every outlet
    # an opportunity/neutral) is thin-but-present, not "winning, hold." The old
    # `or brand_overall_sov >= 0.30` escape hatch let a 0-strength brand qualify
    # (observed: Okta — top-mentioned at 30% but co-mention-only, guard demoted
    # its one lead to opportunity, yet Maintain fired and falsely claimed it
    # "leads at the outlets AI cites most"). Require a real lead instead.
    _present = any(r.get('verdict') == 'strength' for r in sov)
    # "Maintain" claims the brand is top-tier — so it must actually lead (or sit
    # within a hair of) the field by OVERALL mindshare, not merely lead its few
    # sampled outlets. A brand that leads where it's cited but trails badly
    # overall (e.g. ZaiNar 14% vs Ericsson 24%) is broadly-discoverable-but-
    # behind: the honest move is BUILD (grow reach), not "hold, nothing urgent."
    _overall_leader = _brand_overall >= (_comp_overall - 0.03)
    if (not _material_move) and (not _clear_leader) and _present:
        _won = sorted((r for r in sov if _brand_at(r) > 0),
                      key=lambda r: (_n(r), _brand_at(r)), reverse=True)
        _top = _won[0] if _won else None
        _topdom = _top.get('domain') if _top else None
        if _overall_leader:
            # NARROW overall lead (top rival within ~6pts) — a fragile 1-3 response
            # edge in a tightly clustered field is NOT "hold, nothing urgent." If a
            # rival out-cites the brand at a meaningful-volume outlet, recommend
            # pulling ahead THERE (convert the dead heat into a durable lead) rather
            # than coasting on Maintain. Comfortable leaders (>=6pt margin) still
            # Maintain. (Observed: GE HealthCare 74% edging Philips 72% / Siemens
            # 68% was told to "Maintain" a 2-pt lead — the honest lever is to pull
            # ahead where Philips out-cites it, not to coast.)
            _margin = _brand_overall - _comp_overall
            _contested = sorted((r for r in sov if _n(r) >= 4),
                                key=lambda r: (_gap(r), _n(r)), reverse=True)
            if 0 <= _margin < 0.06 and _contested and _gap(_contested[0]) >= 1:
                _cd = _contested[0].get('domain')
                _mpp = max(1, round(_margin * 100))
                text = (f"Extend the lead. {b} is out front by only ~{_mpp} point"
                        f"{'' if _mpp == 1 else 's'} in a tightly clustered field — a fragile edge, "
                        f"not a safe margin. Pull ahead where a rival is closest: press for coverage "
                        f"at {_cd}, where a competitor currently out-cites {b}, to turn a dead heat "
                        f"into a durable lead. Re-audit next quarter to confirm the gap widens.")
                return {"verb": "Extend", "outlet": _cd, "text": text}
            _where = (f" — keep your cadence at {_topdom} and the other outlets where "
                      f"you already lead" if _topdom else "")
            text = (f"Maintain. {b} already holds top-tier AI visibility in this category — it leads "
                    f"or ties the field at the outlets AI cites most, and no competitor is pulling "
                    f"away. There's no urgent earned-media lever right now{_where}; re-audit next "
                    f"quarter to catch any shift early.")
            return {"verb": "Maintain", "outlet": _topdom, "text": text}
        # Broadly discoverable but trailing overall — grow reach, don't "hold."
        # Build is inherently MULTI-OUTLET: name the lead to anchor + the next
        # most-cited pitchable outlets to expand into (the lead excluded), so the
        # move reads as "widen across the category press", not a single pitch.
        _gap_pp = max(1, round((_comp_overall - _brand_overall) * 100))
        # Rank named expansion targets: established trades (MAJOR_TRADE_OUTLETS)
        # ahead of niche newsletters/blogs, THEN by citation volume. A PR pro
        # would rather pitch Light Reading than a robotics newsletter that AI
        # happened to cite more.
        _others = sorted(
            (r for r in sov if (r.get('domain') or '') != _topdom and _n(r) >= 1),
            key=lambda r: (1 if (r.get('domain') or '').lower() in MAJOR_TRADE_OUTLETS else 0,
                           _n(r), _brand_at(r)),
            reverse=True)
        _targets = [r.get('domain') for r in _others[:3] if r.get('domain')]

        def _join(names):
            names = [n for n in names if n]
            if len(names) <= 1:
                return names[0] if names else ""
            return ", ".join(names[:-1]) + " and " + names[-1]

        if _topdom and _targets:
            text = (f"Build. {b} leads at {_topdom} but appears in only a minority of category "
                    f"answers — trailing the leader by ~{_gap_pp} points. Expand earned coverage "
                    f"into the other outlets where AI names competitors instead: {_join(_targets)}. "
                    f"Re-audit next quarter to track the climb.")
            return {"verb": "Build", "outlet": _topdom, "outlets": [_topdom] + _targets, "text": text}
        if _topdom:
            text = (f"Build. {b} is broadly discoverable but appears in only a minority of answers, "
                    f"trailing the leader by ~{_gap_pp} points. Hold your cadence at {_topdom} where "
                    f"you already lead, then widen the aperture into the category outlets where AI "
                    f"names competitors instead; re-audit next quarter to track the climb.")
            return {"verb": "Build", "outlet": _topdom, "text": text}
        text = (f"Build. {b} is broadly discoverable but appears in only a minority of answers, "
                f"trailing the leader by ~{_gap_pp} points. Expand earned coverage into the category "
                f"outlets where AI names competitors instead; re-audit next quarter to track the climb.")
        return {"verb": "Build", "outlet": None, "text": text}

    move_pool = opportunities + contested_strengths
    if move_pool:
        # Volume first; ties broken toward an active gap (opportunity outranks a
        # contested strength at equal volume) and the larger competitor margin.
        def _move_key(r):
            tc = r.get('top_competitor_at_outlet') or {}
            gap = (tc.get('mentions_at_outlet') or 0) - _brand_at(r)
            return (_n(r), 1 if r.get('verdict') == 'opportunity' else 0, gap)
        r = max(move_pool, key=_move_key)
        tc = r.get('top_competitor_at_outlet') or {}
        dom, n, ba = r.get('domain'), _n(r), _brand_at(r)
        cm = tc.get('mentions_at_outlet') or 0
        if r.get('verdict') == 'opportunity':
            if r.get('_pre_guard_verdict') == 'strength':
                # Coverage-guard downgrade: AI names the brand when it cites this
                # outlet, but the pages it actually cites don't feature the brand,
                # so that visibility isn't yet ANCHORED in the source. The move is
                # to earn coverage at this outlet (coverage -> AI presence), NOT to
                # "convert a co-mention" (wrong direction) — and we only assert
                # absence for pages we actually fetched.
                if r.get('_guard_reason') == 'mention':
                    gap_clause = (f"but the cited pages we checked mention {b} only in "
                                  f"passing, so that visibility isn't anchored in the source.")
                else:
                    gap_clause = (f"but the cited pages we checked don't mention {b}, so that "
                                  f"visibility isn't anchored in the source.")
                text = (f"Pitch {dom}. AI names {b} in {ba} of the {n} answers that cite it, "
                        f"{gap_clause} It's an outlet AI leans on for your category — earning "
                        f"coverage there is a high-leverage way to show up in more AI answers.")
            elif tc.get('name') and cm > ba:
                text = (f"Pitch {dom}. {tc['name']} shows up in {cm} of the {n} AI responses "
                        f"citing it, {b} in {ba} — closing this gap is your highest-leverage "
                        f"earned-media move.")
            else:
                text = (f"Pitch {dom}. {b} appears in only {ba} of the {n} AI responses citing it, "
                        f"below its overall visibility — the clearest place to grow.")
            return {"verb": "Pitch", "outlet": dom, "text": text}
        # Contested strength → defend the lead at the most-cited at-risk outlet.
        if ba > cm:
            text = (f"Defend {dom}. {b} leads there ({ba} of {n} responses) but {tc['name']} "
                    f"is right behind ({cm}) — protect this lead before it flips; it's your "
                    f"most-cited outlet at risk.")
        else:
            text = (f"Defend {dom}. {b} is tied with {tc['name']} at the top here ({ba} each "
                    f"of {n} responses) — your most-cited outlet, and a dead heat; pull ahead "
                    f"with a fresh story before they do.")
        return {"verb": "Defend", "outlet": dom, "text": text}

    # 2/3. No opportunities — defend. Prefer the most-CITED strength where a
    # competitor also appears (most contested + most important), else the
    # most-cited uncontested strength.
    if strengths:
        # Prefer a genuinely differentiated lead; fall back to all strengths only
        # if every one is a crowded listicle (never crash / always return a move).
        pool = [r for r in strengths if not _is_crowded_tie(r)] or strengths
        contested = [r for r in pool if (r.get('top_competitor_at_outlet') or {}).get('name')]
        if contested:
            r = max(contested, key=lambda x: (_n(x), (x.get('top_competitor_at_outlet') or {}).get('mentions_at_outlet') or 0))
            tc = r.get('top_competitor_at_outlet') or {}
            dom, n, ba = r.get('domain'), _n(r), _brand_at(r)
            cm = tc.get('mentions_at_outlet') or 0
            text = (f"Defend {dom}. {b} leads there ({ba} of {n} responses), but {tc['name']} "
                    f"is also present ({cm}) — protect this relationship first so the lead holds.")
            return {"verb": "Defend", "outlet": dom, "text": text}
        r = max(pool, key=lambda x: (_n(x), _brand_at(x)))
        dom, n, ba = r.get('domain'), _n(r), _brand_at(r)
        text = (f"Defend {dom}. {b} owns the AI conversation there ({ba} of {n} responses) "
                f"with no competitor present — lock it in with a follow-up story.")
        return {"verb": "Defend", "outlet": dom, "text": text}

    # 4. Thin / all-emerging — cultivate the most-cited outlet.
    if emerging:
        r = max(emerging, key=_n)
        dom = r.get('domain')
        text = (f"Cultivate {dom}. It's among the most-cited outlets in your category "
                f"where {b} isn't yet established — build the relationship early, before "
                f"competitors lock it in.")
        return {"verb": "Cultivate", "outlet": dom, "text": text}
    return None


# ── Owned Signal Finder ──────────────────────────────────────────────────────
# A re-lens of the SAME grounded audit data, focused on OWNED media: the brand's
# and competitors' own sites/blogs/docs that AI cites, plus the topics where a
# competitor's owned content is what AI pulls and the brand has nothing to cite.
# Fully deterministic (no LLM calls) — runs on the same 50-response batch as the
# earned report (PR Signal Finder). Reuses the editorial/owned domain classifiers.

def _url_host(u):
    """Bare host of a URL (www-stripped), '' on failure."""
    try:
        from urllib.parse import urlparse
        h = (urlparse(u or '').netloc or '').lower()
        return h[4:] if h.startswith('www.') else h
    except Exception:
        return ''


_BRAND_DOMAINS_CACHE = {}


def _registrable_root(dom):
    """Second-level label of a domain ('investors.elcompanies.com' -> 'elcompanies')."""
    parts = [p for p in (dom or '').lower().lstrip('.').split('/')[0].split(':')[0].split('.') if p]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else '')


_BRAND_ALIASES_CACHE = {}


def _resolve_brand_aliases(brand):
    """Best-effort SUB-BRAND / product-brand names that AI uses interchangeably with the
    brand (e.g. 'iShares' for BlackRock, 'SPDR' for State Street, 'Google' for Alphabet)
    via one cheap Haiku call — so the brand isn't undercounted when an answer names only
    the sub-brand. Returns a list of distinct sub-brand strings; [] on any failure.
    Cached per-process by brand."""
    if not brand:
        return []
    key = brand.strip().lower()
    if key in _BRAND_ALIASES_CACHE:
        return _BRAND_ALIASES_CACHE[key]
    out = []
    try:
        resp = anthropic.messages.create(
            model=CLAUDE_HAIKU, max_tokens=200,
            messages=[{"role": "user", "content": (
                f'The company/brand "{brand}" may have well-known SUB-BRANDS or product-line brands '
                f'that people and AI assistants use to refer to its products INSTEAD of the company '
                f'name — e.g. BlackRock\'s ETF brand is "iShares", State Street\'s is "SPDR", '
                f'Alphabet\'s is "Google". List ONLY such distinct sub-brand names for "{brand}" that '
                f'an AI answer might use without naming "{brand}" itself. Be conservative — most '
                f'brands have none. Reply ONLY as a JSON array of strings, e.g. ["iShares"]; if none, [].'
            )}])
        txt = (resp.content[0].text or "").strip()
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        if m:
            for a in json.loads(m.group()):
                if isinstance(a, str) and a.strip() and a.strip().lower() != key and a.strip() not in out:
                    out.append(a.strip())
    except Exception as e:
        print("brand-alias resolve failed (continuing):", str(e)[:120])
    out = out[:6]
    _BRAND_ALIASES_CACHE[key] = out
    return out


def _resolve_brand_domains(brand, category=None):
    """Best-effort official domain(s) for the brand via one cheap Haiku call, so the
    owned analysis can credit corporate/abbreviation domains that name-matching can't
    catch (e.g. elcompanies.com for 'Estée Lauder', whose label shares no letters with
    the brand name). Returns a list of bare registrable domains; [] on any failure, in
    which case the owned analysis falls back to pure name-matching (no regression).

    Passing the brand's CATEGORY disambiguates same-name companies: 'IREN' the
    AI-data-center firm resolves to iren.com, not the Italian utility Iren S.p.A.
    at iren.it. Without it, Haiku picks the better-known namesake and poisons the
    owned attribution. Cached per (brand, category) per process."""
    if not brand:
        return []
    key = (brand.strip().lower(), (category or '').strip().lower())
    if key in _BRAND_DOMAINS_CACHE:
        return _BRAND_DOMAINS_CACHE[key]
    out = []
    try:
        resp = anthropic.messages.create(
            model=CLAUDE_HAIKU, max_tokens=200,
            messages=[{"role": "user", "content": (
                f'What are the official website domain(s) owned by the company/brand "{brand}"? '
                f'{("It operates in: " + category + ". If this name is shared by unrelated companies, use that category to identify the RIGHT one. ") if category else ""}'
                f'Include its main consumer site and its corporate/investor site if they differ. '
                f'Reply ONLY with a JSON array of bare registrable domains (no paths, no "www"), '
                f'e.g. ["esteelauder.com","elcompanies.com"]. If you are not confident, reply [].'
            )}])
        txt = (resp.content[0].text or "").strip()
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        if m:
            for d in json.loads(m.group()):
                if not isinstance(d, str):
                    continue
                dd = re.sub(r'^https?://', '', d.strip().lower()).lstrip('.').split('/')[0]
                dd = re.sub(r'^www\.', '', dd)
                if dd and '.' in dd and dd not in out:
                    out.append(dd)
    except Exception as e:
        print("brand-domain resolve failed (continuing without hint):", str(e)[:120])
    out = out[:6]
    _BRAND_DOMAINS_CACHE[key] = out
    return out


_DOMAIN_ALIVE_CACHE = {}


def _domain_is_dead(dm):
    """True ONLY if the domain's root definitively does not host a real site
    (HTTP 404/410). FAIL-OPEN: returns False on 200/3xx/403/timeouts/DNS/SSL
    errors, so a real-but-slow or bot-blocking domain is never dropped. Used to
    catch an LLM-hallucinated same-name domain that name-matches the brand but
    is a dead/parking page — observed: IREN resolved to iren.ai (404) instead of
    the real iren.com (200). Cached per-process."""
    if not dm:
        return False
    if dm in _DOMAIN_ALIVE_CACHE:
        return _DOMAIN_ALIVE_CACHE[dm]
    import urllib.request
    import urllib.error
    dead = False
    try:
        req = urllib.request.Request(
            'https://' + dm + '/', method='GET',
            headers={'User-Agent': 'Mozilla/5.0 (compatible; SignalFinder/1.0)'})
        urllib.request.urlopen(req, timeout=4).close()  # any response -> alive
    except urllib.error.HTTPError as e:
        dead = e.code in (404, 410)                      # real-but-dead root page
    except Exception:
        dead = False                                     # DNS/timeout/SSL -> keep
    _DOMAIN_ALIVE_CACHE[dm] = dead
    return dead


def _verify_brand_domains(brand, aliases, domains, all_responses):
    """Drop a brand_domain that doesn't plausibly belong to the brand: its
    registrable label must match the brand/alias name, OR the bare domain string
    must actually appear somewhere in the raw citations/response text. Catches a
    hallucinated domain tied to a hallucinated alias (BMS's fake alias "Eliquat"
    -> a fabricated eliquat.com) and same-name-company mixups (a same-named but
    unrelated company's domain) that a name-only check alone would miss. Applied
    on EVERY render (not just fresh) so a stale stored domain from before this
    guard existed gets dropped on the next refresh instead of reused blindly."""
    if not domains:
        return domains
    bl = re.sub(r'[^a-z0-9]', '', (brand or '').lower())
    alias_labels = [re.sub(r'[^a-z0-9]', '', (a or '').lower()) for a in (aliases or [])]
    out = []
    for dm in domains:
        label = re.sub(r'[^a-z0-9]', '', (dm or '').split('.')[0].lower())
        plausible_name = bool(label) and (
            label in bl or (bl and (bl.startswith(label) or bl.endswith(label)))
            or any(label and al and (label in al or al in label) for al in alias_labels))
        seen = any(
            (dm or '').lower() in ((c.get('url') or '') + (c.get('domain') or '')).lower()
            for r in (all_responses or []) for c in (r.get('citations') or []))
        if not seen:
            seen = any((dm or '').lower() in (_response_text(r) or '').lower() for r in (all_responses or []))
        if seen:
            out.append(dm)              # grounded in a real citation -> always trust
            continue
        if plausible_name and not _domain_is_dead(dm):
            # Name-matches the brand but never cited: keep it UNLESS the domain is
            # verifiably dead (a same-name hallucination on a parking/404 domain,
            # e.g. IREN -> iren.ai (404) instead of the real iren.com).
            out.append(dm)
    return out


# Curated sub-brand -> parent map for de-duplicating the competitor list. Hand-maintained
# on purpose: an LLM resolver hallucinated false parents (e.g. PIMCO -> Capital Group) and
# missed real ones, which would corrupt client-facing competitor counts. High-confidence
# entries only — the key is matched case-insensitively against each competitor name.
_COMPETITOR_SUBBRAND_PARENTS = {
    'spdr': 'State Street',
    'spdrs': 'State Street',
    'ishares': 'BlackRock',
    # Pharma flagship drugs. Without these, a brand credited for its OWN flagship
    # drug via brand_aliases (e.g. BMS <- Opdivo) looks like it "leads" competitors
    # who are only ever counted by company name — inflating the lead (observed:
    # BMS "+10 over Merck" collapsed to a ~3pt lead once Merck got credit for
    # Keytruda too). Keep to well-known, unambiguous flagship drugs only.
    'keytruda': 'Merck',
    'opdivo': 'Bristol Myers Squibb',
    'yervoy': 'Bristol Myers Squibb',
    'eliquis': 'Bristol Myers Squibb',
    'revlimid': 'Bristol Myers Squibb',
    'orencia': 'Bristol Myers Squibb',
    'tecentriq': 'Roche',
    'avastin': 'Roche',
    'herceptin': 'Roche',
    'ocrevus': 'Roche',
    'imfinzi': 'AstraZeneca',
    'tagrisso': 'AstraZeneca',
    'ibrance': 'Pfizer',
    'enhertu': 'Daiichi Sankyo',
    'repatha': 'Amgen',
    'prolia': 'Amgen',
    'enbrel': 'Amgen',
}


def _merge_competitor_subbrands(competitor_counts, all_responses):
    """Fold a sub-brand competitor into its parent so AI's asset/ETF answers don't list
    the same firm twice (e.g. 'SPDR' is State Street's ETF brand). Idempotent across
    re-renders: a parent's count is ALWAYS recounted as the union of the parent + its
    known sub-brands over the raw answers — whether or not the sub-brand is still a
    separate row — so repeatedly re-rendering doesn't quietly drop the merge once the
    sub-brand entry has been removed. Curated map (above), not an LLM, to stay
    deterministic. (Symmetry: the parent house is credited parent+sub-brand just as the
    audited brand is, e.g. State Street+SPDR == BlackRock+iShares.)"""
    if not competitor_counts or len(competitor_counts) < 2:
        return competitor_counts
    # Reverse the curated map: parent (lowercased) -> set of known sub-brand tokens.
    parent_to_subs = {}
    for sub, parent in _COMPETITOR_SUBBRAND_PARENTS.items():
        parent_to_subs.setdefault(parent.lower(), set()).add(sub)
    present = {(c.get('name') or '').lower() for c in competitor_counts}

    def _subs_for_parent(name_lower):
        for pl, subs in parent_to_subs.items():
            if name_lower == pl or name_lower.startswith(pl + ' '):   # "State Street Global Advisors"
                return subs
        return None

    def _parent_present(parent_lower):
        return any(nl == parent_lower or nl.startswith(parent_lower + ' ') for nl in present)

    drop = []
    for c in competitor_counts:
        nm = c.get('name') or ''
        nml = nm.lower()
        # Parent row: recount union(parent + its known sub-brands) every time.
        subs = _subs_for_parent(nml)
        if subs and all_responses:
            forms = _brand_match_forms(nm)
            for s in subs:
                forms += _brand_match_forms(s)
            union = sum(1 for r in all_responses if _brand_present_in_text(forms, _response_text(r)))
            c['mention_count'] = max(c.get('mention_count') or 0, union)
        # Sub-brand row: drop it, but only when its parent is present to fold into.
        parent_of = _COMPETITOR_SUBBRAND_PARENTS.get(nml)
        if parent_of and _parent_present(parent_of.lower()):
            drop.append(nm)
    if drop:
        print(f"[merge] folded sub-brand competitors {sorted(drop)} into parents")
    return [c for c in competitor_counts if c.get('name') not in drop]


def _owned_peer_competitors(competitors):
    """Owned analysis compares against brand PEERS only — retailers/marketplaces
    sell many brands' goods, so their owned sites aren't a like-for-like content
    rival (mirrors the earned SoV peer rule)."""
    return [c for c in (competitors or []) if (c.get('type') or 'brand_peer') == 'brand_peer']


def _compute_owned_share_of_voice(brand, competitors, related_brands, ranked_domains,
                                  brand_domain_hints=None):
    """Owned share-of-voice: citations to the brand's OWN domains vs each
    competitor's own domains. A parent/acquirer named in the brand string
    (related_brands — e.g. Zendesk for an Ultimate audit) is folded into the
    brand's owned side, since the brand's content lives there. brand_domain_hints
    (resolved official domains) catch corporate/abbreviation sites name-matching
    misses. Returns the leaderboard + the brand's share of the owned-citation pie.
    Deterministic."""
    peers = _owned_peer_competitors(competitors)
    related_stems = _competitor_domain_stems(related_brands or [])
    comp_stem_map = {c['name']: _competitor_domain_stems([c]) for c in peers if c.get('name')}
    hint_roots = {_registrable_root(d) for d in (brand_domain_hints or []) if d}
    brand_owned, brand_domains = 0, []
    comp_owned, comp_domains = {}, {}
    for d in (ranked_domains or []):
        dom, cnt = (d.get('domain') or ''), (d.get('count') or 0)
        if not dom:
            continue
        if (_is_brand_own_domain(dom, brand) or _is_competitor_owned_domain(dom, related_stems)
                or _registrable_root(dom) in hint_roots):
            brand_owned += cnt
            brand_domains.append({'domain': dom, 'count': cnt})
            continue
        for name, stems in comp_stem_map.items():
            if _is_competitor_owned_domain(dom, stems):
                comp_owned[name] = comp_owned.get(name, 0) + cnt
                comp_domains.setdefault(name, []).append({'domain': dom, 'count': cnt})
                break
    owned_total = brand_owned + sum(comp_owned.values())
    rows = [{
        'name': brand, 'is_brand': True, 'count': brand_owned,
        'share': round(brand_owned / owned_total, 3) if owned_total else 0.0,
        'domains': sorted(brand_domains, key=lambda x: -x['count'])[:5],
    }]
    for name, cnt in sorted(comp_owned.items(), key=lambda x: -x[1]):
        rows.append({
            'name': name, 'is_brand': False, 'count': cnt,
            'share': round(cnt / owned_total, 3) if owned_total else 0.0,
            'domains': sorted(comp_domains[name], key=lambda x: -x['count'])[:3],
        })
    return {
        'brand_owned': brand_owned, 'owned_total': owned_total,
        'brand_share': round(brand_owned / owned_total, 3) if owned_total else 0.0,
        'rows': rows,
        'parent_folded': [c.get('name') for c in (related_brands or []) if c.get('name')],
    }


def _compute_earned_owned_mix(brand, competitors, related_brands, ranked_domains,
                              brand_domain_hints=None):
    """How citation-heavy the category is on EARNED (editorial) vs OWNED (brand +
    competitor sites) — tells the brand which game to play. Deterministic."""
    peers = _owned_peer_competitors(competitors)
    related_stems = _competitor_domain_stems(related_brands or [])
    comp_stems = _competitor_domain_stems(peers)
    hint_roots = {_registrable_root(d) for d in (brand_domain_hints or []) if d}
    editorial = owned = total = 0
    for d in (ranked_domains or []):
        dom, cnt = (d.get('domain') or ''), (d.get('count') or 0)
        total += cnt
        if classify_citation_domain(dom) == 'editorial':
            editorial += cnt
        if (_is_brand_own_domain(dom, brand)
                or _is_competitor_owned_domain(dom, related_stems)
                or _is_competitor_owned_domain(dom, comp_stems)
                or _registrable_root(dom) in hint_roots):
            owned += cnt
    return {'editorial': editorial, 'owned': owned, 'total': total}


def _compute_brand_citation_mix(brand, competitors, related_brands, all_responses,
                                brand_domain_hints=None, brand_aliases=None):
    """Brand-level counterpart to the category mix: across the answers that
    MENTION the brand, how does the citation diet split — the brand's OWN pages
    vs editorial vs competitors' own sites? Answers 'how much of your own AI
    story do you author?' (e.g. Citi: 2 of ~100 citations behind Citi-mentioning
    answers are Citi's own content). Response-level attribution using the same
    mention matcher as the rest of the report. Deterministic, no API."""
    peers = _owned_peer_competitors(competitors)
    related_stems = _competitor_domain_stems(related_brands or [])
    comp_stems = _competitor_domain_stems(peers)
    hint_roots = {_registrable_root(d) for d in (brand_domain_hints or []) if d}
    bforms = _brand_match_forms(brand, brand_aliases)
    own = editorial = competitor = other = 0
    for r in (all_responses or []):
        if not _brand_present_in_text(bforms, _response_text(r)):
            continue
        for c in (r.get('citations') or []):
            dom = _url_host(c.get('url') or '') or (c.get('domain') or '')
            if not dom:
                continue
            if (_is_brand_own_domain(dom, brand)
                    or _is_competitor_owned_domain(dom, related_stems)
                    or _registrable_root(dom) in hint_roots):
                own += 1
            elif _is_competitor_owned_domain(dom, comp_stems):
                competitor += 1
            elif classify_citation_domain(dom) == 'editorial':
                editorial += 1
            else:
                other += 1
    total = own + editorial + competitor + other
    return {'own': own, 'editorial': editorial, 'competitor': competitor,
            'other': other, 'total': total}


def _compute_content_gaps(brand, competitors, related_brands, all_responses, max_gaps=6,
                          brand_domain_hints=None):
    """Per-topic 'where to publish': prompts where AI cited a competitor's OWN
    content and the brand had none, ranked by competitor-citation volume. The
    actionable core of Owned Signal Finder. Deterministic."""
    peers = _owned_peer_competitors(competitors)
    related_stems = _competitor_domain_stems(related_brands or [])
    comp_stem_map = {c['name']: _competitor_domain_stems([c]) for c in peers if c.get('name')}
    hint_roots = {_registrable_root(d) for d in (brand_domain_hints or []) if d}
    by_prompt = {}
    for r in (all_responses or []):
        p = r.get('prompt') or ''
        if not p:
            continue
        slot = by_prompt.setdefault(p, {'brand': 0, 'comp': {}})
        for c in (r.get('citations') or []):
            dom = _url_host(c.get('url') or '')
            if not dom:
                continue
            if (_is_brand_own_domain(dom, brand) or _is_competitor_owned_domain(dom, related_stems)
                    or _registrable_root(dom) in hint_roots):
                slot['brand'] += 1
                continue
            for name, stems in comp_stem_map.items():
                if _is_competitor_owned_domain(dom, stems):
                    slot['comp'][name] = slot['comp'].get(name, 0) + 1
                    break
    gaps = []
    for p, v in by_prompt.items():
        if v['brand'] == 0 and v['comp']:
            winners = sorted(v['comp'].items(), key=lambda x: -x[1])
            gaps.append({
                'topic': p, 'competitor_volume': sum(v['comp'].values()),
                'winners': [{'name': n, 'count': c} for n, c in winners[:3]],
            })
    gaps.sort(key=lambda g: -g['competitor_volume'])
    return gaps[:max_gaps]


def _norm_owned_url(url):
    """Display-normalize a citation URL for grouping: drop scheme, leading www,
    query string (utm tags etc.) and fragment, and any trailing slash. Preserves
    path case so the link still resolves. Bare homepage collapses to just the host."""
    u = (url or '').strip()
    if not u:
        return ''
    u = re.sub(r'#.*$', '', u)
    u = re.sub(r'\?.*$', '', u)
    u = re.sub(r'^https?://', '', u, flags=re.I)
    u = re.sub(r'^www\.', '', u, flags=re.I)
    return u.rstrip('/')


# Portfolio benchmark for the owned/editorial citation mix — SNAPSHOT fallback,
# computed 2026-07 across the 47 audits then on file (owned share of all
# citations: p25 9.5%, median 12.6%, p75 16.9%). The live value comes from
# _current_owned_mix_bench() below, which re-derives these quartiles from the
# stored reports every 12h so the "N categories we've audited" count grows with
# each new audit; this constant only serves when the DB query fails or the
# portfolio is thin.
_OWNED_MIX_BENCH = {'n': 47, 'median': 0.126, 'p25': 0.095, 'p75': 0.169, 'asof': '2026-07'}

_MIX_BENCH_CACHE = {'t': 0.0, 'bench': None}


def _current_owned_mix_bench():
    """Live portfolio benchmark for the owned/editorial mix line: quartiles of
    (owned / total citations) across every stored report that has an owned mix,
    recomputed at most every 12h. The JSON fields are extracted in SQL (Postgres
    CAST + #>>, SQLite json_extract), so no multi-MB payload parsing in Python.
    Falls back to the _OWNED_MIX_BENCH snapshot when the query fails or fewer
    than 20 reports qualify — the report line always renders."""
    import time as _time
    now = _time.time()
    if _MIX_BENCH_CACHE['bench'] and (now - _MIX_BENCH_CACHE['t']) < 12 * 3600:
        return _MIX_BENCH_CACHE['bench']
    pairs = []
    try:
        from sqlalchemy import text as _sqltext
        if db.engine.dialect.name == 'postgresql':
            rows = db.session.execute(_sqltext(
                "SELECT CAST(payload AS json) #>> '{owned,mix,owned}', "
                "       CAST(payload AS json) #>> '{owned,mix,total}' "
                "FROM shared_results WHERE payload LIKE '%\"owned\"%'")).fetchall()
        else:
            rows = db.session.execute(_sqltext(
                "SELECT json_extract(payload, '$.owned.mix.owned'), "
                "       json_extract(payload, '$.owned.mix.total') "
                "FROM shared_results")).fetchall()
        pairs = [(float(o), float(t)) for o, t in rows
                 if o is not None and t not in (None, 0, '0', '')]
    except Exception as e:
        db.session.rollback()   # a failed query would poison the PG transaction
        print("[mix-bench] live recompute failed, using snapshot:", str(e)[:120])
    if len(pairs) >= 20:
        ratios = sorted(o / t for o, t in pairs if t)
        n = len(ratios)
        bench = {'n': n, 'median': ratios[n // 2], 'p25': ratios[n // 4],
                 'p75': ratios[(3 * n) // 4], 'asof': 'live'}
    else:
        bench = dict(_OWNED_MIX_BENCH)
    _MIX_BENCH_CACHE.update(t=now, bench=bench)
    return bench


def _owned_url_kind(disp_url):
    """Coarse content-type of an owned URL: is AI citing the brand's HOMEPAGE
    (name recognition, nothing deeper to pull) or DEEP content (blog/newsroom/
    product pages — publishing doing real work)? Homepage detection strips
    locale prefixes (/en-us) and index/home/default filenames, so
    lumen.com/en-us/home.html counts as a homepage, not deep content."""
    host, _, path = disp_url.partition('/')
    p = '/' + path.lower() if path else '/'
    p = re.sub(r'^/(?:[a-z]{2}(?:-[a-z]{2})?)(?=/|$)', '', p)      # locale prefix
    p = re.sub(r'/(?:index|home|default)\.\w+$', '', p).rstrip('/')
    if not p:
        return 'homepage'
    h = host.lower()
    if h.startswith('ir.') or 'investor' in p:
        return 'investor relations'
    if h.startswith(('news.', 'press.')) or re.search(
            r'newsroom|/news(?:/|$)|/press|/media(?:/|$)|/stories(?:/|$)', p):
        return 'newsroom'
    if re.search(r'/blog|/insights|/articles|/research|/perspectives|/learn(?:/|$)'
                 r'|/resources|/guides|/report', p):
        return 'blog / insights'
    return 'site page'


def _compute_top_owned_urls(brand, related_brands, all_responses, brand_domain_hints=None,
                            max_urls=8):
    """The brand's OWN pages that AI actually cites, ranked by citation count.
    Same brand-owned classification as the owned SoV, but grouped by URL (path)
    instead of domain — answers 'which of our pages does AI pull from?'. Each URL
    is tagged with its kind (homepage vs blog/newsroom/etc.), and the return
    bundles a homepage-vs-deep citation split over the FULL set (not just the
    displayed top rows) — the 'AI knows your name vs AI reads your publishing'
    distinction. Returns {'urls': [...], 'homepage_citations': h,
    'total_citations': t}. Deterministic, no API."""
    related_stems = _competitor_domain_stems(related_brands or [])
    hint_roots = {_registrable_root(d) for d in (brand_domain_hints or []) if d}
    seen = {}  # lowercased url -> [display_url, count]
    for r in (all_responses or []):
        for c in (r.get('citations') or []):
            url = (c.get('url') or '').strip()
            if not url:
                continue
            dom = _url_host(url)
            if not dom:
                continue
            if not (_is_brand_own_domain(dom, brand)
                    or _is_competitor_owned_domain(dom, related_stems)
                    or _registrable_root(dom) in hint_roots):
                continue
            disp = _norm_owned_url(url)
            if not disp:
                continue
            key = disp.lower()
            if key in seen:
                seen[key][1] += 1
            else:
                seen[key] = [disp, 1]
    ranked = sorted(seen.values(), key=lambda x: (-x[1], x[0]))
    home = sum(n for d, n in seen.values() if _owned_url_kind(d) == 'homepage')
    total = sum(n for _, n in seen.values())
    return {
        'urls': [{'url': d, 'count': n, 'kind': _owned_url_kind(d)} for d, n in ranked[:max_urls]],
        'homepage_citations': home,
        'total_citations': total,
    }


def _compute_owned_headline_move(brand, owned_sov, content_gaps):
    """The single #1 OWNED move — publish on the biggest content gap, or (no gaps,
    brand already cited) extend the lead. Deterministic, no LLM."""
    b = brand or 'Your brand'
    if content_gaps:
        g = content_gaps[0]
        wn = ((g.get('winners') or [{}])[0]).get('name') or 'Competitors'
        topic = g['topic'] if len(g['topic']) <= 90 else g['topic'][:88] + '…'
        return {
            'verb': 'Publish',
            'text': (f'Publish on "{topic}". {wn}\'s own content is what AI cites for this query, '
                     f'and {b} has nothing AI can pull — owned content here is your fastest path '
                     f'to getting cited.'),
        }
    if (owned_sov or {}).get('brand_owned', 0) > 0:
        return {
            'verb': 'Extend',
            'text': (f"{b}'s own content already gets cited ({owned_sov['brand_owned']} times) and no "
                     f"single topic shows a competitor's content beating you outright — keep "
                     f"publishing on the queries you already win to widen the lead."),
        }
    return None


def _compute_owned_analysis(brand, competitors, related_brands, ranked_domains, all_responses,
                            brand_domain_hints=None, brand_aliases=None):
    """Bundle the owned-media lens into one dict for the payload (`owned` key).
    Pure-Python; same data as the earned report. brand_domain_hints (resolved
    official domains) credit corporate/abbreviation sites name-matching misses.
    Safe no-op on thin inputs."""
    sov = _compute_owned_share_of_voice(brand, competitors, related_brands, ranked_domains,
                                        brand_domain_hints=brand_domain_hints)
    mix = _compute_earned_owned_mix(brand, competitors, related_brands, ranked_domains,
                                    brand_domain_hints=brand_domain_hints)
    gaps = _compute_content_gaps(brand, competitors, related_brands, all_responses,
                                 brand_domain_hints=brand_domain_hints)
    top = _compute_top_owned_urls(brand, related_brands, all_responses,
                                  brand_domain_hints=brand_domain_hints)
    return {
        'owned_sov': sov,
        'mix': mix,
        'brand_mix': _compute_brand_citation_mix(brand, competitors, related_brands,
                                                 all_responses,
                                                 brand_domain_hints=brand_domain_hints,
                                                 brand_aliases=brand_aliases),
        'content_gaps': gaps,
        'top_owned_urls': top['urls'],
        'owned_page_mix': {'homepage': top['homepage_citations'],
                           'deep': top['total_citations'] - top['homepage_citations']},
        'headline_move': _compute_owned_headline_move(brand, sov, gaps),
    }


def _enrich_owned_recommendations(brand, category, owned):
    """Turn each owned content-gap *query* into an actionable publishing angle via
    one Claude call — so the report says "publish an explainer on how index funds
    differ from actively managed funds" instead of echoing the raw question. Mutates
    `owned` in place: sets gap['recommendation'] and rewrites the Publish headline to
    lead with the top angle. No-op on any failure (the raw query phrasing remains)."""
    gaps = (owned or {}).get('content_gaps') or []
    if not gaps:
        return owned
    listing = "\n".join(f'{i+1}. "{g.get("topic", "")}"' for i, g in enumerate(gaps))
    b = brand or 'the brand'
    try:
        resp = anthropic.messages.create(
            model=CLAUDE_HAIKU, max_tokens=700,
            messages=[{"role": "user", "content": (
                f'Brand: {b}\nCategory: {category or ""}\n\n'
                f'Below are AI-search queries where {b} has NO owned content that AI cites, but a '
                f'competitor does. For EACH query, write a concise, specific CONTENT ANGLE {b} should '
                f'publish to start winning that query. Extrapolate the editorial angle — do NOT echo '
                f'the question. ~8-16 words, a noun phrase/brief, do NOT begin with "Publish" or '
                f'"Create". Begin with a LOWERCASE letter unless the first word is a proper noun or '
                f'an acronym (AI, ESG, ETF). Tie it to {b} where natural.\n\n'
                f'Style examples:\n'
                f'  query "how do I choose between index funds and actively managed funds for my '
                f'portfolio" -> "an informational explainer on how index funds differ from actively '
                f'managed funds"\n'
                f'  query "which cosmetics companies have the strongest sustainability and ESG '
                f'reputations" -> "sustainability and ESG as a reputation differentiator for {b}"\n\n'
                f'Queries:\n{listing}\n\n'
                f'Return ONLY a JSON array of exactly {len(gaps)} strings, in the same order.'
            )}])
        txt = (resp.content[0].text or "").strip()
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        recs = json.loads(m.group()) if m else []
        for g, rec in zip(gaps, recs):
            if isinstance(rec, str) and rec.strip():
                g['recommendation'] = rec.strip()
    except Exception as e:
        print("owned recommendation enrich failed (continuing):", str(e)[:120])
    # Lead the Publish headline with the top gap's angle (deterministic fallback kept).
    hm = (owned or {}).get('headline_move') or {}
    rec0 = (gaps[0].get('recommendation') or '').strip()
    if rec0 and hm.get('verb') == 'Publish':
        wn = ((gaps[0].get('winners') or [{}])[0]).get('name') or 'A competitor'
        hm['text'] = (f"Publish {rec0}. {wn}'s own content is what AI cites for this query today, and "
                      f"{b} has nothing AI can pull — owned content here is your fastest path to "
                      f"getting cited.")
    return owned


def _wins_article_label(url):
    """A readable article label from a cited URL's slug (the citation data carries
    no title). Returns '' for homepages/section roots so they're skipped."""
    import urllib.parse
    try:
        p = urllib.parse.urlparse(url or '')
    except Exception:
        return ''
    segs = [s for s in p.path.split('/') if s and s.lower() not in ('index.html', 'amp')]
    if not segs:
        return ''
    slug = re.sub(r'\.(html?|php|aspx)$', '', segs[-1])
    slug = re.sub(r'[?#].*$', '', slug)
    slug = re.sub(r'[-_]+', ' ', slug)
    slug = re.sub(r'\b\d{4,}\b', '', slug)          # strip trailing numeric IDs from slugs
    slug = re.sub(r'\s+', ' ', slug).strip()
    if len(slug) < 4:
        return ''
    if len(slug) > 56:
        slug = slug[:56].rsplit(' ', 1)[0] + '…'
    return slug[:1].upper() + slug[1:]


def _outlet_url_recount(outlet, brand, rival, all_responses, brand_aliases=None):
    """Reproducible per-outlet counts for the wins claim: among answers that cite
    `outlet` via a real URL, how many NAME the brand vs the top rival (per-answer
    presence, using the same matcher as the rest of the report). The stored
    outlet_sov counts ALSO credit answers that merely name the outlet with no link —
    right for share-of-voice, but a 'what AI pulls' claim must survive a recipient
    counting the citation links, so the wins use this conservative URL count.

    Uses distinctive-token presence matching (NOT the corpus-gated _count_brand_mentions,
    which undercounts a short-form brand on a per-answer basis — e.g. an answer saying
    'JetBlue', not 'JetBlue Airways')."""
    bforms = _brand_match_forms(brand, brand_aliases)
    rforms = _brand_match_forms(rival) if rival else []
    n = bp = cp = 0
    for r in (all_responses or []):
        hosts = [(c.get('domain') or '').lower() for c in (r.get('citations') or [])]
        if not any(h == outlet or h.endswith('.' + outlet) for h in hosts):
            continue
        n += 1
        txt = _response_text(r)
        if _brand_present_in_text(bforms, txt):
            bp += 1
        if rforms and _brand_present_in_text(rforms, txt):
            cp += 1
    return n, bp, cp


def _compute_earned_wins(brand, outlet_sov, all_responses, max_wins=8, max_articles=6, brand_aliases=None):
    """The flattering inverse of media-targets: outlets where the brand's earned
    coverage is *working* — either it out-cites rivals in AI's answers (leading) or
    its coverage is verified on the pages AI pulled (page-confirmed present) — so
    non-dominant brands still show real wins. Ranked verified-first, with the
    specific articles AI is pulling. Powers
    the report's 'what's working' lead section so a prospect sees validation before
    growth areas. Deterministic, no API. Returns up to max_articles candidate
    articles per outlet (with URLs) so _attach_wins_recency can re-select toward
    recent placements; callers/templates display the first ~2."""
    wins = []
    for r in (outlet_sov or []):
        dom = (r.get('domain') or '').lower()
        # Skip outlets the stored (broader 'cited or named') count already says are
        # tiny, then RECOUNT the survivors from real URL citations only — so the
        # displayed 'in N answers / named X' survives a recipient counting the links.
        if not dom or (r.get('responses_citing') or 0) < 2:
            continue
        rival_name = (r.get('top_competitor_at_outlet') or {}).get('name') or ''
        n, b, cmax = _outlet_url_recount(dom, brand, rival_name, all_responses, brand_aliases=brand_aliases)
        pe = r.get('page_evidence') or {}
        bp = pe.get('brand_pages') or 0
        leads = b > cmax              # a STRICT lead — a tie is not a lead
        confirmed = bp > 0
        # A 'strength to defend': AI names the brand in most of the answers citing this
        # outlet (>=60%) AND no rival clearly out-cites it there (the brand is within
        # 10% of the top rival, or ahead). Reproducible from the URL-citation recount,
        # so it survives a reader counting links. A verified-but-clearly-trailing outlet
        # (Upgraded Points, where United leads 14 to 11) is a pitch target, not a
        # defend. The n>=3 / b>=2 floor drops thin one-off citations.
        if n < 3 or b < 2:
            continue
        if not (b >= 0.6 * n and b >= 0.9 * cmax):
            continue
        wins.append({
            'outlet': r.get('domain'), 'answers': n, 'brand_mentions': b,
            'rival_mentions': cmax, 'page_confirmed': bp,
            'pages_checked': pe.get('pages_ok') or 0,
            'leads': leads, 'tier': 'leading' if leads else 'present', 'articles': [],
        })
    # Defend list: lead with the most influential relationship (most AI answers cite
    # it), with on-page-verified outlets winning the tiebreak.
    wins.sort(key=lambda w: (-w['answers'], -(1 if w['page_confirmed'] else 0)))
    wins = wins[:max_wins]
    hosts = {w['outlet']: w for w in wins}
    seen = {h: {} for h in hosts}   # host -> {label: {'llms': set, 'url': url}}
    for r in (all_responses or []):
        llm = r.get('llm')
        for c in (r.get('citations') or []):
            h = (c.get('domain') or '').lower()
            if h in hosts:
                u = c.get('url') or ''
                label = _wins_article_label(u)
                if label:
                    slot = seen[h].setdefault(label, {'llms': set(), 'url': u})
                    slot['llms'].add(llm)
    brand_tokens = [f.lower() for f in _brand_match_forms(brand, brand_aliases) if ' ' not in f]  # URL-matchable forms
    for w in wins:
        ranked = sorted(seen[w['outlet']].items(), key=lambda x: -len(x[1]['llms']))[:max_articles]
        w['articles'] = [{'label': lbl, 'url': v['url'], 'llms': sorted(x for x in v['llms'] if x),
                          # a brand-specific article (URL names the brand, e.g.
                          # .../jetblue-domestic-first-class-cabin) is far better
                          # sample coverage than a generic category listicle
                          'brand_relevant': any(t in (v['url'] or '').lower() for t in brand_tokens)}
                         for lbl, v in ranked]
    return {
        'placements': len(wins),
        'total_answers': sum(w['answers'] for w in wins),
        'page_confirmed': sum(1 for w in wins if w['page_confirmed']),
        'wins': wins,
    }


def _attach_wins_recency(earned_wins, recent_days=120):
    """Enrich win articles with publication date + real title from the CitedPage
    cache, then re-select the 2 strongest per outlet biased toward RECENT placements
    (a piece from the last few months lands harder than a 2-year-old evergreen) and
    surface the date. Graceful: undated articles keep their slug label and sort after
    dated ones; on any lookup failure each outlet's candidates are simply trimmed to
    2 by citation breadth. Requires a DB/app context (CitedPage)."""
    import hashlib
    wins = (earned_wins or {}).get('wins') or []
    urls = [a.get('url') for w in wins for a in (w.get('articles') or []) if a.get('url')]
    meta = {}
    if urls:
        try:
            hl = {hashlib.sha1(u.encode()).hexdigest(): u for u in urls}
            for r in CitedPage.query.filter(CitedPage.url_hash.in_(list(hl.keys()))).all():
                meta[r.url_hash] = {'title': (r.title or '').strip(), 'published': (r.published or '').strip()}
        except Exception as e:
            print("wins recency lookup failed (continuing):", str(e)[:120])
    now = datetime.utcnow()
    recent_total = 0
    for w in wins:
        cands = w.get('articles') or []
        for a in cands:
            h = hashlib.sha1((a.get('url') or '').encode()).hexdigest()
            m = meta.get(h) or {}
            t = m.get('title') or ''
            if t:
                t = html.unescape(t)                 # decode &#039; etc. (esc() re-escapes on render)
                # strip a trailing " | Site" / " - Site" / " — Site" suffix (outlet shown already)
                t = re.sub(r'\s*[|–—\-]\s*[^|–—\-]{1,30}$', '', t).strip() or t
            if 4 < len(t) <= 110:
                a['label'] = t                       # real headline beats the URL slug
            age = None
            pub = m.get('published') or ''
            if pub:
                try:
                    dt = parse(pub, fuzzy=True, ignoretz=True)
                    if 1990 < dt.year <= now.year + 1:
                        age = (now - dt).days
                        a['date_label'] = dt.strftime('%b %Y')
                except Exception:
                    pass
            a['age_days'] = age
            a['recent'] = (age is not None and 0 <= age <= recent_days)

        def _score(a):
            base = len(a.get('llms') or []) * 2      # citation breadth = relevance signal
            if a.get('recent'):
                base += 4                            # strong recency bias, but breadth still counts
            elif a.get('age_days') is not None and a['age_days'] <= 365:
                base += 1
            return base
        # Brand-specific articles (URL names the brand — a dedicated piece about YOUR
        # brand/launch) are the best sample coverage, so they rank ahead of even a
        # heavily-cited generic category listicle; then by recency/breadth score.
        cands.sort(key=lambda a: (0 if a.get('brand_relevant') else 1, -_score(a),
                                  a.get('age_days') if a.get('age_days') is not None else 99999))
        w['articles'] = cands[:2]
        if any(a.get('recent') for a in w['articles']):
            recent_total += 1
    earned_wins['recent_placements'] = recent_total
    return earned_wins


def _compute_launch_landing(brand, product_name, all_responses, competitors=None, brand_aliases=None):
    """Launch-landing metrics for the Announcement-Anchored mode: did the brand and
    its SPECIFIC product surface organically in the category, how AI frames it (new vs
    absent), who leads the field, and a negative-narrative flag for operator review.
    Deterministic (reuses the distinctive-token matcher). Returns None on no responses;
    the 'launch not yet detected' empty case is handled by the caller/template."""
    resp = [r for r in (all_responses or []) if _response_text(r)]
    total = len(resp)
    if not total:
        return None
    bforms = _brand_match_forms(brand, brand_aliases)
    pforms = _brand_match_forms(product_name) if (product_name or '').strip() else []
    NEW_RE = re.compile(
        r"\b(new|newly|launch\w*|debut\w*|introduc\w+|unveil\w*|upcoming|coming soon|"
        r"rolling out|roll out|set to (?:launch|debut|offer|introduce)|just announced|"
        r"recently announced|will (?:offer|launch|introduce|debut|add)|plans to)\b", re.I)
    NEG_RE = re.compile(
        r"\b(cut\w*|reduc\w+|shrink\w*|smaller|downgrad\w*|criticism|backlash|complaint\w*|"
        r"controvers\w+|disappoint\w*|slash\w*|removed|loses?|losing|worse)\b", re.I)
    EXC_RE = re.compile(
        r"\b(doesn'?t|does not|don'?t|do not|no longer|lacks?|without|unlike|"
        r"not (?:offer|have|available|yet))\b", re.I)
    from collections import defaultdict
    brand_ans = product_ans = framed_new = neg_ans = exc_ans = 0
    brand_by_llm = defaultdict(lambda: [0, 0])
    product_llms = set()
    neg_snippets = []
    for r in resp:
        txt = _response_text(r)
        llm = r.get('llm') or '?'
        brand_by_llm[llm][1] += 1
        b = _brand_present_in_text(bforms, txt)
        p = bool(pforms) and _brand_present_in_text(pforms, txt)
        if b:
            brand_ans += 1
            brand_by_llm[llm][0] += 1
        if p:
            product_ans += 1
            product_llms.add(llm)
        # New / negative / exception framing only counts when the term sits NEAR a
        # brand or product mention (<=110 chars), so a competitor's complaints in the
        # same answer aren't miscounted as the brand's.
        bpos = [m.start() for f in (bforms + pforms)
                for m in re.finditer(r'\b' + re.escape(f) + r'\b', txt, re.I)]
        if not bpos:
            continue

        def _near(rx, _txt=txt, _bpos=bpos):
            for m in rx.finditer(_txt):
                if any(abs(m.start() - bp) <= 110 for bp in _bpos):
                    return m
            return None
        if _near(NEW_RE):
            framed_new += 1
        nm = _near(NEG_RE)
        if nm:
            neg_ans += 1
            if len(neg_snippets) < 3:
                s, e = max(0, nm.start() - 70), min(len(txt), nm.end() + 70)
                neg_snippets.append('…' + re.sub(r'\s+', ' ', txt[s:e]).strip() + '…')
        if b and _near(EXC_RE):
            exc_ans += 1
    leaders = sorted([(c.get('name'), c.get('mention_count') or 0)
                      for c in (competitors or []) if c.get('name')], key=lambda x: -x[1])[:5]
    field = sorted(leaders + [(brand, brand_ans)], key=lambda x: -x[1])
    brand_rank = next((i + 1 for i, (n, _) in enumerate(field) if n == brand), None)
    return {
        'product_name': (product_name or '').strip(),
        'landed_answers': brand_ans, 'total': total,
        'product_answers': product_ans, 'product_llms': sorted(product_llms),
        'brand_by_llm': {l: v[0] for l, v in brand_by_llm.items()},
        'framed_as_new': framed_new, 'as_exception': exc_ans,
        'negative_flag': neg_ans > 0, 'negative_answers': neg_ans, 'negative_snippets': neg_snippets,
        'leaders': [{'name': n, 'mentions': m} for n, m in leaders],
        'brand_rank': brand_rank, 'field_size': len(field),
    }


# Assistants whose answers are grounded in live web search (so a brand can
# surface there from RECENT press / its own site even if the base model has
# never "heard of" it). The others answer mostly from parametric knowledge, so
# a mention there means the brand has reached real cultural saturation. The
# gap between the two is a core strategic signal for challenger brands.
SEARCH_GROUNDED_LLMS = {'Gemini', 'Perplexity'}

# ALL_GROUNDED mode — when enabled, Claude, ChatGPT, and Grok ALSO run live web
# search (Claude web_search tool, ChatGPT gpt-4o-search-preview, Grok via
# OpenRouter ":online"), so all 5 assistants answer in "search mode." This makes
# the cross-model comparison apples-to-apples (no model penalised for being
# parametric) and reflects the most recent web coverage. Each response carries a
# per-call `grounded` flag (set False when a grounded call failed and fell back
# to a parametric retry), so the report can still show which models actually
# retrieved. Default OFF — flip ALL_GROUNDED=1 on Render after profiling memory/
# latency (the earlier native-search attempt OOM'd on the 512MB Starter box;
# we're on Standard/2GB now, with bounded searches + no double-call fallback).
ALL_GROUNDED = os.environ.get("ALL_GROUNDED", "").strip().lower() in ("1", "true", "yes", "on")


_BRAND_GENERIC_TOKENS = {
    'the', 'and', 'inc', 'corp', 'co', 'llc', 'ltd', 'plc', 'group', 'holding', 'holdings',
    'company', 'companies', 'airways', 'airlines', 'technologies', 'technology', 'systems',
    'international', 'global', 'brands', 'labs', 'studio', 'studios', 'media', 'corporation',
    'healthcare', 'health', 'medical', 'sciences', 'science', 'solutions', 'services',
    'pharmaceuticals', 'pharma', 'partners', 'ventures', 'capital', 'financial',
}


def _brand_match_forms(brand, aliases=None):
    """Distinctive forms for matching the primary brand in answer text: the full name
    plus its most distinctive token — a CamelCase/ALLCAPS token like 'JetBlue' or
    'LEGO', else the longest non-generic token — so 'JetBlue Airways' is found via
    'JetBlue' and 'The LEGO Group' via 'LEGO'. Plus any resolved SUB-BRAND aliases (e.g.
    'iShares' for BlackRock) and their distinctive tokens, so the brand isn't
    undercounted when an answer names only the sub-brand. Needed because the corpus-
    gated _count_brand_mentions fallback misses these on small per-LLM subsets."""
    brand = (brand or '').strip()
    if not brand:
        return []
    forms = [brand]
    toks = re.findall(r"[A-Za-z][\w&'’-]*", brand)
    # A token can LOOK distinctive (CamelCase / ALLCAPS) while still being a
    # generic corporate-suffix word — "HealthCare" in "GE HealthCare" matches
    # the CamelCase shape but is just the word "healthcare", which appears in
    # almost every response about a healthcare company (inflated GE 37/50 to a
    # false 49/50). Apply the same generic-token filter here as the fallback
    # branch below already used.
    distinctive = [t for t in toks
                  if (re.search(r'[a-z][A-Z]', t) or (t.isupper() and len(t) >= 3))
                  and t.lower() not in _BRAND_GENERIC_TOKENS]
    if distinctive:
        forms += distinctive
    else:
        sig = [t for t in toks if len(t) >= 4 and t.lower() not in _BRAND_GENERIC_TOKENS]
        if sig:
            forms.append(max(sig, key=len))
    out = []
    for i, f in enumerate(forms):
        if not f:
            continue
        if i > 0 and len(f) < 3:        # keep the full name regardless; derived tokens need >=3
            continue
        if f.lower() not in {x.lower() for x in out}:
            out.append(f)
    for al in (aliases or []):          # fold in sub-brand aliases + their distinctive tokens
        # An alias is matched WHOLESALE against the answer text (not decomposed
        # like the brand name above), so it needs its own length/generic guard —
        # otherwise a short/common alias (e.g. "GE") bypasses the >=3-char guard
        # entirely, since as forms[0] of its own recursive call it's never i>0.
        if len(al) < 3 or al.lower() in _GENERIC_ALIAS_WORDS:
            continue
        for f in _brand_match_forms(al):
            if f and f.lower() not in {x.lower() for x in out}:
                out.append(f)
    return out


def _response_text(r):
    return r.get('response') or r.get('text') or r.get('answer') or ''


def _brand_present_in_text(forms, text):
    """True if any distinctive brand form appears (word-boundary, case-insensitive)."""
    return any(re.search(r'\b' + re.escape(f) + r'\b', text, re.IGNORECASE) for f in (forms or []))


# Common English words that recur as brand-alias / product-line names but are too
# generic to safely count without false positives (GE HealthCare's alias "Discovery"
# collides with generic usage of the word; "Innova" collides with "innovation").
# Small and high-confidence on purpose — extend only on a confirmed false positive.
_GENERIC_ALIAS_WORDS = {
    'discovery', 'signature', 'precision', 'vitality', 'freedom', 'horizon',
    'access', 'edge', 'core', 'pulse', 'vision', 'summit', 'origin', 'spark', 'signa',
    'innova', 'innovation', 'connect', 'plus', 'pro', 'max', 'one', 'go', 'live',
}


def _verify_brand_aliases(aliases, all_responses):
    """Drop hallucinated / too-generic aliases BEFORE they're used for counting.
    An alias must (1) be >=3 chars, (2) not be a generic English word, (3) appear
    at least once in the raw response text (word-boundary), and (4) not be
    lowercase-dominated (a proxy for "this is being used as an ordinary word, not
    the brand"). Catches two real failure modes seen in production: hallucinated
    aliases that match NOTHING (BMS's "Cellgene"/"Eliquat" — garbled/wrong drug
    names invented by the resolver) and generic aliases that match too MUCH (GE
    HealthCare's "GE"/"Innova"/"Discovery" inflated 37/50 to a false 49/50).
    Deterministic — no LLM call, safe to run on every audit including reruns of
    older payloads whose stored aliases predate this guard."""
    out = []
    for a in (aliases or []):
        a = (a or '').strip()
        if len(a) < 3 or a.lower() in _GENERIC_ALIAS_WORDS:
            continue
        pat_ci = re.compile(r'(?<![A-Za-z0-9])' + re.escape(a) + r'(?![A-Za-z0-9])', re.IGNORECASE)
        ci = sum(1 for r in (all_responses or []) if pat_ci.search(_response_text(r)))
        if ci == 0:
            continue
        pat_cs = re.compile(r'(?<![A-Za-z0-9])' + re.escape(a) + r'(?![A-Za-z0-9])')
        cs = sum(1 for r in (all_responses or []) if pat_cs.search(_response_text(r)))
        if (ci - cs) >= 3 and cs < ci * 0.5:
            continue
        out.append(a)
    return out


def _compute_per_llm_visibility(brand, all_responses, aliases=None):
    """Per-assistant brand visibility: for each LLM, in how many of ITS responses
    the brand appears. A headline "20% mindshare" can hide the real story — e.g. a
    brand cited in 9/10 Gemini responses but 0/10 on ChatGPT/Claude/Grok is
    concentrated in one (search-grounded) assistant, not broadly embedded.

    Counts answer-PRESENCE directly from the raw response text via distinctive-token
    matching (NOT the corpus-gated _count_brand_mentions, which undercounts a brand
    referred to by short form on a per-LLM subset — e.g. Grok/Perplexity saying
    'JetBlue', not 'JetBlue Airways'). Returns {llm, mentions, total, rate, grounded}
    sorted by rate desc.
    """
    if not brand or not all_responses:
        return []
    forms = _brand_match_forms(brand, aliases)
    order = []
    seen = set()
    for r in all_responses:
        l = r.get('llm')
        if l and l not in seen:
            seen.add(l)
            order.append(l)
    out = []
    for l in order:
        subset = [r for r in all_responses if r.get('llm') == l]
        n = len(subset)
        m = sum(1 for r in subset if _brand_present_in_text(forms, _response_text(r)))
        # Prefer the per-response `grounded` flag (present on audits run after
        # the grounded-retrieval change): an assistant is "grounded" here if a
        # majority of its responses actually retrieved. Fall back to the static
        # SEARCH_GROUNDED_LLMS set for older audits that lack the flag.
        flagged = [r for r in subset if 'grounded' in r]
        if flagged:
            grounded = sum(1 for r in flagged if r.get('grounded')) >= (len(flagged) / 2)
        else:
            grounded = l in SEARCH_GROUNDED_LLMS
        out.append({
            'llm': l,
            'mentions': m,
            'total': n,
            'rate': round(m / n, 3) if n else 0.0,
            'grounded': grounded,
        })
    out.sort(key=lambda x: x['rate'], reverse=True)
    return out


def _llm_visibility_read(brand, per_llm):
    """One-line strategic read of the per-assistant visibility spread."""
    if not per_llm:
        return None
    b = brand or 'Your brand'
    total = len(per_llm)
    present = [x for x in per_llm if (x.get('mentions') or 0) > 0]
    absent = [x['llm'] for x in per_llm if (x.get('mentions') or 0) == 0]
    present_names = [x['llm'] for x in present]

    # DEPTH metrics — being named by every assistant is NOT the same as being
    # named OFTEN by each. Without this guard a brand at 3/10 on each assistant
    # reads as "broad, embedded visibility" when it's really named in only a
    # minority of answers. mean_rate = avg mention rate across assistants;
    # leader/weakest = strongest/thinnest assistant. Used below to gate the
    # strong "embedded / consistent" language honestly.
    def _rate(x):
        t = x.get('total') or 0
        return (x.get('mentions') or 0) / t if t else 0.0
    def _frac(x):
        return f"{x.get('mentions')}/{x.get('total')}"
    _tot_resp = sum(x.get('total') or 0 for x in per_llm)
    mean_rate = (sum(x.get('mentions') or 0 for x in per_llm) / _tot_resp) if _tot_resp else 0.0
    leader = max(per_llm, key=lambda x: x.get('mentions') or 0)
    weakest = min(per_llm, key=lambda x: x.get('mentions') or 0)
    # "Embedded/consistent" only when the brand is named in a real share of EACH
    # assistant's answers — a majority on average AND a solid plurality on the
    # weakest. "Uneven" when one assistant carries the brand far more than
    # another (>=30pp gap). Otherwise it's present-but-thin.
    is_deep = mean_rate >= 0.5 and _rate(weakest) >= 0.3
    is_uneven = (_rate(leader) - _rate(weakest)) >= 0.3

    # GROUNDED MODE (ALL_GROUNDED): every assistant answered with live web
    # search, so the spread reflects DISCOVERABILITY — what AI finds when it
    # searches the current web — NOT what's embedded in the model's memory. The
    # "search-grounded vs parametric" framing below doesn't apply when all 5
    # retrieve, so reframe around live-search reach.
    if all(x.get('grounded') for x in per_llm):
        if not present:
            return (f"{b} is absent from all {total} assistants even in live-search mode — "
                    f"no discoverable web presence in this category yet.")
        if len(present) == total:
            # "All 5 surfacing" doesn't mean broad+deep. Three honest reads:
            # DEEP (named in most answers everywhere) → strong, consistent;
            # UNEVEN (one assistant carries it, e.g. Lululemon ranged 1-5) →
            # name strongest + weakest; THIN (present everywhere but only a
            # minority of answers each) → broadly discoverable, not the default.
            if is_deep:
                return (f"{b} surfaces in most answers across all {total} assistants in live-search "
                        f"mode — strong, consistent discoverability when AI searches your category.")
            if is_uneven:
                return (f"{b} surfaces across all {total} assistants but unevenly — strongest on "
                        f"{leader.get('llm')} ({_frac(leader)}), thinnest on "
                        f"{weakest.get('llm')} ({_frac(weakest)}). Broad reach, uneven depth.")
            return (f"{b} is named by all {total} assistants in live-search mode but only in a "
                    f"minority of answers (~{round(mean_rate*100)}% each) — broadly discoverable, "
                    f"not yet the default recommendation.")
        if len(present) >= max(2, total - 2):
            return (f"{b} surfaces on {len(present)} of {total} assistants in live-search mode; "
                    f"thinner on {', '.join(absent)}. Solid discoverability with room to broaden.")
        return (f"{b} surfaces on only {len(present)} of {total} assistants "
                f"({', '.join(present_names)}) even with live search — limited discoverability; "
                f"the underlying web coverage AI can find is thin.")

    if not present:
        return (f"{b} doesn't surface on any of the {total} assistants for unbranded "
                f"category queries — effectively invisible in AI today.")
    if len(present) == total:
        # Present on every assistant — but gate the "embedded" claim on DEPTH.
        # 3/10 on each assistant is recognition across the board, NOT broadly
        # embedded; saying so over-claims and erodes trust on the first read.
        if is_deep:
            return (f"{b} surfaces in most answers across all {total} assistants — "
                    f"broad, embedded AI visibility.")
        if is_uneven:
            return (f"{b} is named by all {total} assistants but unevenly — strongest on "
                    f"{leader.get('llm')} ({_frac(leader)}), thinnest on "
                    f"{weakest.get('llm')} ({_frac(weakest)}).")
        return (f"{b} is named by all {total} assistants but only in a minority of answers "
                f"(~{round(mean_rate*100)}% each) — recognized across the board, not yet the "
                f"default pick.")
    # Concentrated only in the search-grounded assistants, absent from the
    # parametric ones = recent-press visibility, not yet model-embedded.
    if set(present_names) <= SEARCH_GROUNDED_LLMS and absent:
        return (f"{b} surfaces mainly on search-grounded assistants "
                f"({', '.join(present_names)}) but is absent from {', '.join(absent)} — "
                f"visibility is search-driven (recent press / your own site), not yet "
                f"embedded in the models people use most.")
    return (f"{b} surfaces on {len(present)} of {total} assistants "
            f"({', '.join(present_names)}); absent from {', '.join(absent)}.")


def _ml_brand_patterns(brand):
    """Word-boundary, case-insensitive brand matcher for the coarse media-
    landscape presence check (multi-form, no corpus guardrail)."""
    if not brand:
        return []
    pats = [re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)]
    parts = brand.split()
    if len(parts) > 1 and len(parts[0]) > 5:
        pats.append(re.compile(r'\b' + re.escape(parts[0]) + r'\b', re.IGNORECASE))
    return pats


def _compute_media_landscape(brand, category, all_responses, ranked_domains):
    """The client's MEDIA WATCHLIST — the outlets a PR team in this category
    considers essential REGARDLESS of whether AI currently cites them — each
    cross-referenced against the citation data and tagged with a presence
    status the way a PR buyer thinks about their list:

      'driving'   — AI cites this outlet AND names the brand in those answers.
      'open_lane' — AI cites this outlet for the category but NOT the brand
                    (pitch target: the outlet is in the AI's picture, you're not).
      'off_radar' — AI doesn't cite this outlet for the category at all
                    (brand-equity / early-mover play, not an AI-visibility move).

    One Claude call builds the watchlist (tagged consumer/trade/business/
    advocacy); status is deterministic. This is the answer to "why isn't WWD in
    my report" — it appears, with an honest reason. Returns {} on any failure so
    the report simply omits the section.
    """
    if not all_responses:
        return {}
    try:
        prompt = (
            f"You are a senior PR strategist building the media list for a brand in this category:\n"
            f"\"{category or 'this category'}\".\n\n"
            f"List the 18 publications/outlets a PR team for such a brand would consider ESSENTIAL "
            f"to its media list — the outlets that matter to them regardless of AI. Span marquee "
            f"CONSUMER titles, the key TRADE/industry press, relevant BUSINESS/mainstream press, and "
            f"(only if genuinely central to this category) ADVOCACY/institutional orgs.\n\n"
            f"Return ONLY a JSON array of objects, each exactly: "
            f'{{"name":"Outlet Name","domain":"outlet.com","type":"consumer|trade|business|advocacy"}}. '
            f"Real primary domains only (no www, no paths). No prose, no markdown."
        )
        resp = anthropic.messages.create(
            model=CLAUDE_SONNET, max_tokens=1400, timeout=45.0,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.content[0].text or "").strip()
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        watch = json.loads(m.group(0) if m else txt)
    except Exception as e:
        print("media-landscape watchlist failed (skipping):", str(e)[:160])
        return {}

    # Cited domains + their sample article URLs (from the ranked-domain list).
    cited_urls = {}
    for d in (ranked_domains or []):
        dom = (d.get('domain') or '').lower().replace('www.', '')
        if dom:
            cited_urls[dom] = [u for u in (d.get('specific_urls') or d.get('urls') or [])
                               if _is_specific_article(u)]
    for r in all_responses:
        for c in (r.get('citations') or []):
            dd = (c.get('domain') or '').lower().replace('www.', '')
            if dd:
                cited_urls.setdefault(dd, [])

    def _cited_key(dom):
        if dom in cited_urls:
            return dom
        for k in cited_urls:
            if k.endswith('.' + dom) or dom.endswith('.' + k):
                return k
        return None

    rows, seen_dom = [], set()
    for w in (watch or []):
        if not isinstance(w, dict):
            continue
        name = (w.get('name') or '').strip()
        dom = (w.get('domain') or '').lower().replace('www.', '').strip('/')
        typ = (w.get('type') or 'consumer').strip().lower()
        if typ not in ('consumer', 'trade', 'business', 'advocacy'):
            typ = 'consumer'
        if not name or not dom or dom in seen_dom:
            continue
        seen_dom.add(dom)
        rows.append({'name': name, 'domain': dom, 'type': typ, 'cited': _cited_key(dom)})

    # Fetch-verify cited outlets so 'driving' means the brand is ACTUALLY on the
    # outlet's cited page — not just co-occurring in an answer that cites it
    # (the "cited in a roundup that also names you" false positive). Bounded.
    page_hit = {}
    fjobs = []
    for ri, row in enumerate(rows):
        if row['cited']:
            for u in (cited_urls.get(row['cited']) or [])[:1]:
                fjobs.append((ri, u))
    if fjobs and brand:
        ex = ThreadPoolExecutor(max_workers=10)
        try:
            futs = {ex.submit(_page_brand_mentions, u, brand): ri for (ri, u) in fjobs}
            for fut in as_completed(list(futs.keys()), timeout=22):
                try:
                    if (fut.result() or 0) >= 1:
                        page_hit[futs[fut]] = True
                except Exception:
                    pass
        except FuturesTimeoutError:
            pass
        except Exception:
            pass
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    out = {}
    for ri, row in enumerate(rows):
        if not row['cited']:
            status = 'off_radar'
        elif page_hit.get(ri):
            status = 'driving'
        else:
            status = 'open_lane'
        out.setdefault(row['type'], []).append(
            {'name': row['name'], 'domain': row['domain'], 'type': row['type'], 'status': status})
    order = {'driving': 0, 'open_lane': 1, 'off_radar': 2}
    for typ in out:
        out[typ].sort(key=lambda o: order.get(o['status'], 3))
    return out


_URL_VALIDATION_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


# Regexes used by _resolve_vertex_redirect — module-level so we compile once.
_META_REFRESH_RE = re.compile(
    r'<meta\s+[^>]*http-equiv=["\']?refresh["\']?\s+content=["\'][^"\']*url=([^"\'>\s]+)',
    re.IGNORECASE,
)
_JS_LOCATION_RE = re.compile(
    r'window\.location(?:\.(?:href|replace))?\s*[=\(]\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _resolve_vertex_redirect(url, timeout=4):
    """Gemini grounding URLs (vertexaisearch.cloud.google.com/...) redirect via
    a <meta http-equiv="refresh"> tag or JS location-assignment — NOT via an
    HTTP Location header. requests.head + allow_redirects can't follow them,
    so without this we leave the vertex redirector itself in the citation
    list, polluting the top of the domain ranking.

    Fetches the redirector body and extracts the real target URL. Returns
    None if the redirector itself returns an error or no target can be parsed.
    """
    try:
        if not _is_safe_url(url):
            return None
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers=_URL_VALIDATION_HEADERS)
        if r.status_code >= 400:
            return None
        body = r.text or ''
        m = _META_REFRESH_RE.search(body)
        if m:
            return m.group(1)
        m = _JS_LOCATION_RE.search(body)
        if m:
            return m.group(1)
        return None
    except Exception:
        return None


def _is_specific_article(url):
    """Heuristic: does this URL point to a specific article, not a homepage/tag/category?

    Used to filter the `sample_urls` surfaced in the final report (so we never
    show a Forbes homepage or a /tag/ai/ index as evidence). The underlying
    citation data (`raw_citation_domains[*].urls`, `all_responses[*].citations`)
    is preserved untouched — only the presentation layer filters.

    Returns True when the URL looks like it leads to an actual article:
    - Has a path beyond `/` (not bare domain, not /index.html)
    - Doesn't contain known index/listing segments (/tag/, /category/, /search?, etc.)
    - Has 2+ path segments OR a single segment with a long slug (>= 20 chars,
      proxy for "this is a slug, not a section name")

    Errs on the side of KEEPING unknown URLs (returns True on parse failure)
    so legitimate-but-unusual URLs aren't silently dropped.

    KNOWN LIMITATION: A URL that LOOKS like a specific article but is actually
    off-topic (e.g. a Verge article about an electric golf cart cited as
    evidence for an AI creative platform audit) will still pass this filter.
    TODO: optional GET + title-check pass that verifies the article title
    mentions the brand or category — expensive (network call per URL), so
    we'd want it as an opt-in for paid audits.
    """
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        path = (p.path or '').rstrip('/')
        # Bare domain or root-only = homepage.
        if not path or path == '/' or path.lower() == '/index.html':
            return False
        path_lower = path.lower()
        # Tag / category / search / archive / pagination / author index pages.
        bad_segments = (
            '/tag/', '/tags/', '/category/', '/categories/', '/topic/', '/topics/',
            '/search/', '/search?', '/archive/', '/archives/', '/page/', '/author/',
            '/by/', '/feed', '/rss', '/sitemap', '/atom', '/section/',
        )
        if any(seg in path_lower for seg in bad_segments):
            return False
        # Query-string-only "search" pages.
        if (p.query or '').lower().startswith(('q=', 'query=', 's=', 'search=')):
            return False
        segs = [s for s in path.split('/') if s]
        if not segs:
            return False
        # Single short segment is almost always a section, not an article.
        # 2+ segments OR a long slug (proxy for "this is a real article slug") = keep.
        if len(segs) < 2 and len(segs[0]) < 20:
            return False
        return True
    except Exception:
        # Err on the side of keeping unknown URLs.
        return True


def _resolve_and_verify_urls(urls, timeout=2.5, max_workers=40, on_progress=None):
    """HEAD-check + redirect-resolve a batch of URLs concurrently.

    Returns a dict {original_url: final_url_or_None}.
    - None means: confabulated (404/410), DNS failure, or refused connection.
    - final_url may differ from original (Gemini's grounding URLs redirect from
      vertexaisearch.cloud.google.com → real source; we use the real URL/domain).
    - For 403 / 5xx / timeouts we err toward KEEPING the URL (might be bot-blocked
      but real). Only definitive negatives are dropped.

    Tuned aggressively: timeout 2.5s + 40 workers. The bottleneck used to be
    waiting on slow servers; with these settings 100+ URLs verify in 5-10s
    instead of 30s+. Only retries with GET for 403 (the main bot-block signal).
    """
    out = {}

    def check(u):
        # SSRF guard: never fire HEAD against private/loopback/reserved IPs.
        # LLM citation URLs are untrusted input; without this a confabulated
        # internal URL could probe our own infra. Drop on failure (treat as
        # confabulated).
        if not _is_safe_url(u):
            return (u, None)
        # Vertex grounding redirects need a body-parse to recover the real
        # target (meta-refresh / JS, not HTTP 30x). Do that BEFORE the normal
        # HEAD logic — without this the vertexaisearch domain ends up at the
        # TOP of the ranking, which is useless. If we can't recover the real
        # source, drop the URL entirely (the redirector itself is not a
        # citable source).
        original_u = u
        if 'vertexaisearch.cloud.google.com' in u.lower():
            real = _resolve_vertex_redirect(u)
            if not real:
                return (original_u, None)
            u = real
            if not _is_safe_url(u):
                return (original_u, None)
        try:
            r = requests.head(u, timeout=timeout, allow_redirects=True,
                              headers=_URL_VALIDATION_HEADERS)
            if r.status_code in (404, 410):
                return (original_u, None)
            if r.status_code < 400:
                return (original_u, r.url)
            if r.status_code == 403:
                # Possible bot-block of HEAD; one retry with streaming GET.
                try:
                    r2 = requests.get(u, timeout=timeout, allow_redirects=True, stream=True,
                                      headers=_URL_VALIDATION_HEADERS)
                    try:
                        r2.close()
                    except Exception:
                        pass
                    if r2.status_code in (404, 410):
                        return (original_u, None)
                    if r2.status_code < 400:
                        return (original_u, r2.url)
                except Exception:
                    pass
                return (original_u, u)
            # 4xx (other than 404/410/403) or 5xx — assume real, keep resolved URL.
            return (original_u, u)
        except requests.exceptions.ConnectionError:
            return (original_u, None)  # DNS / connection refused = confabulated
        except Exception:
            return (original_u, u)  # timeout / TLS quirks — keep, don't penalize real URLs

    completed = 0
    total = len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(check, u) for u in urls]
        for fut in as_completed(futures):
            try:
                orig, final = fut.result()
                out[orig] = final
            except Exception:
                pass
            completed += 1
            if on_progress:
                # Emit progress periodically (every 5 URLs or on completion) so the
                # frontend ETA can update smoothly through this step.
                if completed % 5 == 0 or completed == total:
                    try:
                        on_progress(completed, total)
                    except Exception:
                        pass
    return out


_TOPIC_CHECK_HEADERS = dict(_URL_VALIDATION_HEADERS)  # reuse browser UA


def _check_url_topic(url, brand_lower, category_keywords, timeout=4):
    """Fetch a URL and check whether its title or first ~1500 chars of body text
    actually mentions the brand or any category keyword. Catches the failure
    mode where a URL passes _is_specific_article (looks article-shaped) but
    the article is wholly off-topic — e.g. an LLM citing a Verge article about
    an electric golf cart as "evidence" for an AI creative platforms audit.

    Returns one of:
      'verified'     — page fetched 200, title or body sample contains the brand
                       (word-boundary, case-insensitive) OR any category_keyword
                       (case-insensitive substring).
      'off_topic'    — page fetched fine with parseable body, but no match. Drop.
      'confabulated' — definitive negative (404 or 410). Always drop, never
                       eligible for transitive trust — the URL truly does not
                       exist at the publisher. Distinct from 'inaccessible' so
                       the caller can apply the right policy.
      'inaccessible' — non-200 we can't be sure about (403, 5xx, timeout, GET
                       error, empty body). Could be bot-blocked legitimate
                       content. Caller applies transitive-trust rules.
    """
    try:
        if not _is_safe_url(url):
            return 'inaccessible'
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers=_TOPIC_CHECK_HEADERS, stream=False)
        # 404/410 = definitive negative. The publisher's webserver explicitly
        # says this URL is not (and never was, in the case of 410) here. This
        # is exactly the LLM-hallucination fingerprint — a plausibly-shaped URL
        # that doesn't exist. Never give these transitive trust.
        if r.status_code in (404, 410):
            return 'confabulated'
        if r.status_code >= 400:
            return 'inaccessible'
        body = (r.text or '')[:8000]  # generous slice for title-then-body extraction
        if not body.strip():
            return 'inaccessible'
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', body, re.IGNORECASE)
        title = (title_match.group(1) if title_match else '').strip()
        # Strip HTML tags from body sample for keyword check.
        text_sample = re.sub(r'<[^>]+>', ' ', body[:4000])
        combined = (title + ' ' + text_sample).lower()
        # Brand check (word boundary).
        if brand_lower and re.search(r'\b' + re.escape(brand_lower) + r'\b', combined):
            return 'verified'
        # Category keyword check (any substring match).
        for kw in category_keywords:
            if kw and kw in combined:
                return 'verified'
        return 'off_topic'
    except Exception:
        return 'inaccessible'


def _page_brand_mentions(url, brand, timeout=10):
    """Fetch a URL and count how many times `brand` appears in its title+body
    (word-boundary, case-insensitive, with the multi-word first-word fallback).
    Returns the count, or -1 if the page couldn't be fetched/parsed.

    This is what separates CONFIRMED COVERAGE (the brand is actually on the
    cited page) from a CATEGORY OUTLET (the AI cited the outlet for the topic,
    but the page is about a competitor or generic trends). The existing
    topic-verify pass returns 'verified' on brand OR category-keyword match, so
    a Rare-Beauty packaging article passes as "on-topic" without covering the
    brand at all — this check closes that gap.

    Scans the FULL page body (capped at 3MB), not a 200KB prefix — large
    publisher pages (Hearst ~2MB, CMSWire ~800KB) front-load hundreds of KB of
    script before the article, so a prefix scan false-negatives real coverage
    (observed: Adobe first appears at byte 365K of a CMSWire page that names it
    64 times). Script/style blocks are stripped first so JSON-LD and tracking
    payloads neither hide nor inflate the count.
    """
    try:
        if not _is_safe_url(url):
            return -1
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers=_TOPIC_CHECK_HEADERS, stream=False)
        if r.status_code >= 400:
            return -1
        raw = (r.text or '')[:3_000_000]
        if not raw.strip():
            return -1
        title_m = re.search(r'<title[^>]*>([^<]+)</title>', raw, re.IGNORECASE)
        body = re.sub(r'(?is)<(script|style)\b[^>]*>.*?</\1>', ' ', raw)
        text = (title_m.group(1) if title_m else '') + ' ' + re.sub(r'<[^>]+>', ' ', body)
        if not brand:
            return 0
        cnt = len(re.findall(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE))
        parts = brand.split()
        # Single-word common-word guardrail (same logic as _count_brand_mentions):
        # short single-word brands ("On", "Apple") collide with English words on
        # the page. If the case-insensitive count dominates the case-sensitive
        # one, the brand is colliding with the lowercase common word — use the
        # case-sensitive (proper-noun-only) count instead. Otherwise a page about
        # something unrelated could falsely register as confirmed brand coverage.
        if len(parts) == 1 and len(brand) <= 6:
            cs_cnt = len(re.findall(r'\b' + re.escape(brand) + r'\b', text))
            if cnt > max(cs_cnt * 2, 3):
                cnt = cs_cnt
        if cnt == 0 and len(parts) > 1 and len(parts[0]) > 5:
            cnt = len(re.findall(r'\b' + re.escape(parts[0]) + r'\b', text, re.IGNORECASE))
        return cnt
    except Exception:
        return -1


def _verify_brand_coverage(brand, media_targets, ranked_domains=None,
                           deadline_seconds=25, max_urls_per_target=2):
    """For each media target, fetch its sample article URL(s) and check whether
    the brand is ACTUALLY on the page. Mutates each target in place:
      coverage = 'confirmed' (brand on a cited page)
               | 'category'  (page(s) fetched, brand NOT present — pitch target,
                              not current coverage)
               | 'unverified'(no fetchable article URL — e.g. parametric
                              hallucinated URLs, or homepage-only citations)
      brand_confirmed (bool), brand_page_mentions (int, -1 if unverified).
    Bounded by a wall-clock deadline so it never blows the audit budget.
    """
    if not media_targets or not brand:
        return
    dom_urls = {}
    for d in (ranked_domains or []):
        dom_urls[(d.get('domain') or '').lower()] = (d.get('specific_urls') or d.get('urls') or [])
    jobs = []  # (target_index, url)
    for i, t in enumerate(media_targets):
        urls = [u for u in (t.get('sample_urls') or []) if _is_specific_article(u)]
        if not urls:
            urls = [u for u in dom_urls.get((t.get('domain') or '').lower(), []) if _is_specific_article(u)]
        for u in list(dict.fromkeys(urls))[:max_urls_per_target]:
            jobs.append((i, u))
    results = {}
    if jobs:
        ex = ThreadPoolExecutor(max_workers=12)
        try:
            futs = {ex.submit(_page_brand_mentions, u, brand): u for (i, u) in jobs}
            for fut in as_completed(list(futs.keys()), timeout=deadline_seconds):
                try:
                    results[futs[fut]] = fut.result()
                except Exception:
                    results[futs[fut]] = -1
        except FuturesTimeoutError:
            pass
        except Exception as e:
            print("brand-coverage verify error (continuing):", str(e)[:120])
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    for i, t in enumerate(media_targets):
        urls = [u for (ti, u) in jobs if ti == i]
        fetched = [results[u] for u in urls if results.get(u, -1) >= 0]
        best = max(fetched) if fetched else -1
        t['brand_page_mentions'] = best
        if best >= 2:
            # Brand named multiple times on the page = substantive coverage.
            t['brand_confirmed'] = True
            t['coverage'] = 'confirmed'
        elif best == 1:
            # A single passing mention — often the brand listed in a roundup or
            # a competitor's article (e.g. Tilt named once in a Rare Beauty
            # packaging story). Honest middle ground, not "they cover you."
            t['brand_confirmed'] = False
            t['coverage'] = 'mention'
        elif fetched:
            t['brand_confirmed'] = False
            t['coverage'] = 'category'
        else:
            t['brand_confirmed'] = False
            t['coverage'] = 'unverified'


# --- Citation-page scraping (page-evidence layer) ----------------------------
# Fetches the actual pages AI cited and counts EVERY tracked brand name on
# each, producing per-outlet "on-page share of voice" alongside the in-answer
# SoV. The divergence between the two is the insight: a competitor named in
# AI answers but absent from the cited pages is FREE-RIDING on the outlet's
# coverage; a brand on the pages but missing from answers has coverage that
# ISN'T CONVERTING. Results are cached cross-audit in CitedPage.

_PAGE_CACHE_TTL_DAYS = 30
_READER_PREFIX = "https://r.jina.ai/"  # plain-text reader fallback for bot-blocked sites


def _count_names_in_text(text, names):
    """Word-boundary, case-insensitive count for each name, with the same
    common-word guardrail as _page_brand_mentions (short single-word brands
    like 'Gap' or 'On' collide with English words — when the case-insensitive
    count dwarfs the case-sensitive one, trust the proper-noun count).

    names[0] is treated as the AUDITED BRAND and gets a short-form fallback: if
    the full multi-word name isn't on the page, count a distinctive Capitalized
    first word (>=4 chars) — e.g. a Healthline page that says 'Hims' but not
    'Hims & Hers'. Without this the brand's real on-page coverage reads 0 and the
    coverage guard falsely downgrades a genuine strength. Case-sensitive so a
    lowercase common word (a competitor 'Bank of America' -> 'bank') can't false-
    match; and competitors (names[1:]) stay conservative (full name only)."""
    out = {}
    for i, name in enumerate(names or []):
        if not name:
            continue
        cnt = len(re.findall(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
        parts = name.split()
        if len(parts) == 1 and len(name) <= 6:
            cs = len(re.findall(r'\b' + re.escape(name) + r'\b', text))
            if cnt > max(cs * 2, 3):
                cnt = cs
        if i == 0 and cnt == 0 and len(parts) > 1 and len(parts[0]) >= 4:
            cnt = len(re.findall(r'\b' + re.escape(parts[0]) + r'\b', text))
        out[name] = cnt
    return out


def _extract_page_meta(raw):
    """Pull title / author / published from raw HTML via meta tags. Cheap
    regexes, no parsing dependencies; absent fields come back as ''."""
    def _meta(*patterns):
        for p in patterns:
            m = re.search(p, raw, re.IGNORECASE)
            if m:
                return (m.group(1) or '').strip()[:255]
        return ''
    title_m = re.search(r'<title[^>]*>([^<]+)</title>', raw, re.IGNORECASE)
    return {
        'title': (title_m.group(1).strip()[:300] if title_m else ''),
        'author': _meta(
            r'<meta\s+name=["\']author["\']\s+content=["\']([^"\']+)',
            r'<meta\s+name=["\']parsely-author["\']\s+content=["\']([^"\']+)',
            r'<meta\s+property=["\']article:author["\']\s+content=["\']([^"\']+)'),
        'published': _meta(
            r'<meta\s+property=["\']article:published_time["\']\s+content=["\']([^"\']+)',
            r'<meta\s+name=["\']parsely-pub-date["\']\s+content=["\']([^"\']+)',
            r'<meta\s+name=["\']date["\']\s+content=["\']([^"\']+)',
            r'<time[^>]+datetime=["\']([^"\']+)'),
    }


def _scrape_cited_page(url, names, timeout=10):
    """Fetch one cited page (direct first, reader-proxy fallback for
    bot-blocked sites) and count every tracked name on it. Returns a dict
    shaped like a CitedPage row, never raises."""
    result = {'status': 'error', 'via': 'direct', 'title': '', 'author': '',
              'published': '', 'content_len': 0, 'counts': {}}
    raw = None
    try:
        if _is_safe_url(url):
            r = requests.get(url, timeout=timeout, allow_redirects=True,
                             headers=_TOPIC_CHECK_HEADERS, stream=False)
            if r.status_code < 400 and (r.text or '').strip():
                raw = r.text[:3_000_000]
    except Exception:
        raw = None
    # Bot-stub heuristic: blocked publishers (Hearst et al.) serve tiny husk
    # pages with a 200. Anything under ~15KB from a major-publisher article
    # URL is suspect — try the reader proxy, which fetches from its own infra.
    if raw is None or len(raw) < 15000:
        try:
            rr = requests.get(_READER_PREFIX + url, timeout=timeout + 10,
                              headers={'User-Agent': _TOPIC_CHECK_HEADERS.get('User-Agent', '')})
            if rr.status_code < 400 and len(rr.text or '') > len(raw or ''):
                text = rr.text[:3_000_000]
                result.update(status='ok', via='reader', content_len=len(text),
                              counts=_count_names_in_text(text, names))
                tm = re.match(r'\s*Title:\s*(.+)', text)
                if tm:
                    result['title'] = tm.group(1).strip()[:300]
                return result
        except Exception:
            pass
    if raw is None:
        result['status'] = 'blocked'
        return result
    meta = _extract_page_meta(raw)
    body = re.sub(r'(?is)<(script|style)\b[^>]*>.*?</\1>', ' ', raw)
    text = meta['title'] + ' ' + re.sub(r'<[^>]+>', ' ', body)
    result.update(status='ok', content_len=len(raw),
                  counts=_count_names_in_text(text, names), **meta)
    return result


def _attach_page_evidence(outlet_sov, ranked_domains, brand, competitor_counts,
                          max_urls_per_outlet=6, deadline_seconds=75):
    """For each SoV outlet, scrape its cited article URLs (cache-first) and
    attach `page_evidence` to the row:
      {pages_checked, pages_ok, brand_pages, brand_mentions,
       competitors: {name: pages_containing}, pages: [{url,title,published}]}
    Mutates rows in place; safe no-op on any failure."""
    if not outlet_sov or not brand:
        return
    names = [brand] + [c.get('name') for c in (competitor_counts or [])
                       if c.get('name') and c.get('type', 'brand_peer') == 'brand_peer'][:8]
    dom_urls = {}
    for d in (ranked_domains or []):
        dom = (d.get('domain') or '').lower()
        urls = [u for u in (d.get('specific_urls') or d.get('urls') or [])
                if _is_specific_article(u)]
        if urls:
            dom_urls[dom] = list(dict.fromkeys(urls))[:max_urls_per_outlet]
    jobs = []  # (domain, url)
    for row in outlet_sov:
        dom = (row.get('domain') or '').lower()
        for u in dom_urls.get(dom, []):
            jobs.append((dom, u))
    if not jobs:
        return
    # Cache pass (main thread — SQLAlchemy sessions aren't thread-safe).
    import hashlib as _hl
    by_url = {}
    hashes = {u: _hl.sha1(u.encode()).hexdigest() for (_, u) in jobs}
    try:
        cutoff = datetime.utcnow() - timedelta(days=_PAGE_CACHE_TTL_DAYS)
        cached = CitedPage.query.filter(CitedPage.url_hash.in_(list(hashes.values()))).all()
        for c in cached:
            if c.fetched_at and c.fetched_at >= cutoff and c.status == 'ok':
                counts = json.loads(c.name_counts or '{}')
                if all(n in counts for n in names):
                    by_url[c.url] = {'status': c.status, 'via': c.via,
                                     'title': c.title or '', 'author': c.author or '',
                                     'published': c.published or '',
                                     'content_len': c.content_len or 0, 'counts': counts}
    except Exception as e:
        print("page-evidence cache read failed (continuing):", str(e)[:120])
    to_fetch = [(d, u) for (d, u) in jobs if u not in by_url]
    if to_fetch:
        ex = ThreadPoolExecutor(max_workers=12)
        try:
            futs = {ex.submit(_scrape_cited_page, u, names): u for (_, u) in to_fetch}
            for fut in as_completed(list(futs.keys()), timeout=deadline_seconds):
                try:
                    by_url[futs[fut]] = fut.result()
                except Exception:
                    pass
        except FuturesTimeoutError:
            print(f"page-evidence: deadline hit with {len(jobs) - len(by_url)} pages unfetched")
        except Exception as e:
            print("page-evidence scrape error (continuing):", str(e)[:120])
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        # Cache write-back (main thread).
        try:
            for (_, u) in to_fetch:
                res = by_url.get(u)
                if not res:
                    continue
                h = hashes[u]
                rec = CitedPage.query.filter_by(url_hash=h).first()
                if rec is None:
                    rec = CitedPage(url_hash=h, url=u)
                    db.session.add(rec)
                rec.domain = u.split('/')[2].lower() if '://' in u else ''
                rec.status, rec.via = res['status'], res['via']
                rec.title, rec.author = res['title'], res['author']
                rec.published, rec.content_len = res['published'], res['content_len']
                rec.name_counts = json.dumps(res['counts'])
                rec.fetched_at = datetime.utcnow()
            db.session.commit()
        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            print("page-evidence cache write failed (continuing):", str(e)[:120])
    # Aggregate per outlet.
    scraped = 0
    for row in outlet_sov:
        dom = (row.get('domain') or '').lower()
        urls = dom_urls.get(dom, [])
        results = [by_url[u] for u in urls if u in by_url]
        ok = [r for r in results if r['status'] == 'ok']
        if not urls:
            continue
        comp_pages = {}
        for n in names[1:]:
            comp_pages[n] = sum(1 for r in ok if (r['counts'].get(n) or 0) > 0)
        row['page_evidence'] = {
            'pages_checked': len(urls),
            'pages_ok': len(ok),
            'brand_pages': sum(1 for r in ok if (r['counts'].get(brand) or 0) > 0),
            'brand_mentions': sum(r['counts'].get(brand) or 0 for r in ok),
            'competitors': comp_pages,
            'pages': [{'url': u, 'title': by_url[u]['title'],
                       'published': by_url[u]['published']}
                      for u in urls if u in by_url and by_url[u]['status'] == 'ok'][:4],
        }
        scraped += len(ok)
    print(f"page-evidence: {scraped} pages analyzed across {len(dom_urls)} outlets "
          f"({len(jobs) - len(to_fetch)} from cache)")


_COVERAGE_ORDER = {'confirmed': 0, 'mention': 1, 'category': 2, 'unverified': 3}


def _sort_targets_by_coverage(targets):
    """Float CONFIRMED coverage to the top of the report, then passing mentions,
    then category pitch-targets, then unverified — and re-rank in place. Makes
    the most credible, verifiable outlets lead every report."""
    if not targets:
        return
    targets.sort(key=lambda t: (_COVERAGE_ORDER.get(t.get('coverage'), 4),
                                -(t.get('brand_page_mentions') or 0)))
    for i, t in enumerate(targets):
        t['rank'] = i + 1


# --- Outlet relevance filter -------------------------------------------------
# Some publications are genuinely "editorial" (classify_citation_domain says so)
# but are the WRONG AUDIENCE for the brand being audited: trade / industrial /
# professional outlets the AI cited as a topical source, not pitchable earned
# media. Surfacing them as "media targets" is noise — e.g. packagingeurope.com
# cited on a beauty-packaging trend, or professionalbeauty.in for a consumer
# brand that isn't sold there and isn't for salon pros. We drop them from
# media_targets + share-of-voice but KEEP them in raw_citation_domains, so the
# CSV/JSON backstage data stays complete and the call is reversible.

# Supply-side / industrial trade — wrong for almost every brand unless the
# brand's OWN category is in that industry (guarded against category below).
_TRADE_INDUSTRIAL_SIGNALS = (
    'packaging', 'supplychain', 'supply-chain', 'wholesale', 'logistics',
    'procurement', 'manufacturing', 'plasticsnews', 'industryweek',
)
# Practitioner / professional trade — right for B2B / practitioner brands,
# wrong for CONSUMER brands (a consumer makeup line doesn't earn coverage
# pitching a salon-professional trade title).
_PROFESSIONAL_TRADE_SIGNALS = ('professional', 'practitioner')
# If the category reads as consumer-facing, the professional-trade filter turns
# on. Enterprise / B2B categories (Adobe, a VC firm) keep professional outlets.
_CONSUMER_CATEGORY_HINTS = (
    'beauty', 'makeup', 'cosmetic', 'skincare', 'haircare', 'hair care',
    'fragrance', 'perfume', 'fashion', 'apparel', 'clothing', 'footwear',
    'jewelry', 'food', 'beverage', 'snack', 'grocery', 'consumer', 'retail',
    'shopper', 'wellness', 'supplement', 'personal care', 'cpg', 'lifestyle',
    'pet ', 'toy', 'baby',
)


def _category_is_consumer(category):
    cat_l = (category or '').lower()
    return any(h in cat_l for h in _CONSUMER_CATEGORY_HINTS)


def _is_irrelevant_outlet(domain, outlet_name, is_consumer, category_l):
    """True if this editorial outlet is the wrong audience for the brand — a
    trade / industrial / professional publication, not pitchable consumer or
    corporate earned media. `category_l` (lowercased category) guards against
    dropping an outlet whose industry IS the brand's category (a packaging
    company should keep packaging-trade press)."""
    hay = f"{(domain or '').lower()} {(outlet_name or '').lower()}"
    for sig in _TRADE_INDUSTRIAL_SIGNALS:
        if sig in hay and sig not in category_l:
            return True
    if is_consumer:
        for sig in _PROFESSIONAL_TRADE_SIGNALS:
            if sig in hay and sig not in category_l:
                return True
    return False


def _filter_relevant_editorial(editorial_domains, category, log_prefix=""):
    """Drop off-audience trade / industrial / professional outlets from a list
    of ranked-domain dicts. Returns the filtered list; logs what was dropped."""
    cat_l = (category or '').lower()
    is_consumer = _category_is_consumer(category)
    kept, dropped = [], []
    for d in (editorial_domains or []):
        if _is_irrelevant_outlet(d.get('domain'), d.get('outlet_name'), is_consumer, cat_l):
            dropped.append(d.get('domain'))
        else:
            kept.append(d)
    if dropped:
        print(f"{log_prefix}relevance filter dropped {len(dropped)} off-audience "
              f"outlet(s): {', '.join(x for x in dropped[:12] if x)}")
    return kept


def _editorial_dicts_for_targets(ranked_domains, media_targets):
    """Return the ranked-domain dicts (each carrying `_citing_idx`) for exactly
    the media-target outlets, in target order. Share-of-voice is then computed
    for WHAT THE REPORT SHOWS — so every card gets its competitor breakdown, and
    the strength/opportunity verdicts aren't polluted by (or capped out in favor
    of) obscure outlets that never become cards. Fixes the bug where a marquee
    'neutral' outlet (NewBeauty, Allure) was dropped by the SoV top-N cap while
    a one-off blog surfaced as a 'strength'."""
    by = {(d.get('domain') or '').lower(): d for d in (ranked_domains or [])}
    out, seen = [], set()
    for t in (media_targets or []):
        dom = (t.get('domain') or '').lower()
        if dom and dom in by and dom not in seen:
            out.append(by[dom])
            seen.add(dom)
    return out


def _sort_targets_by_prominence(targets, outlet_sov=None):
    """Rank media targets by how prominently the AI surfaces them — responses
    citing the outlet first, then raw citation frequency. Replaces the old
    coverage-tier sort: the report ranks by relevance + share-of-voice, not by
    whether a single fetched page happened to name the brand."""
    if not targets:
        return
    sov_by = {(r.get('domain') or '').lower(): r for r in (outlet_sov or [])}

    def _key(t):
        sov = sov_by.get((t.get('domain') or '').lower()) or {}
        return (-(sov.get('responses_citing') or 0), -(t.get('citation_frequency') or 0))

    targets.sort(key=_key)
    for i, t in enumerate(targets):
        t['rank'] = i + 1


def _reconcile_coverage_from_page_evidence(media_targets, outlet_sov):
    """Make the page-evidence layer the single source of truth for each
    target's `coverage` field, overriding the older _verify_brand_coverage
    result whenever page-evidence actually read pages.

    WHY: two coverage systems run per audit. _verify_brand_coverage samples
    up to 2 sample_urls; _attach_page_evidence samples up to 6 ranked URLs
    with a reader-proxy fallback for bot-blocked publishers — strictly more
    thorough. They can disagree, which produced a self-contradicting card on
    the Lululemon report: the headline said "the cited pages don't cover
    Lululemon" at Women's Health while that card's source-page check showed
    Lululemon on 2 of 3 pages. Downstream (coverage-guard verdict + headline
    move + card label) all read `coverage`, so reconciling it here fixes the
    contradiction everywhere at once.

    Mapping from page_evidence (only when pages were actually read). Coverage
    is judged RELATIVE to how many pages we could actually read, not on an
    absolute page count — otherwise an outlet where only one page was fetchable
    but the brand IS on it gets mislabeled "mention / in passing" and the guard
    then downgrades a real strength. (Observed on Oatly: tastingtable.com named
    Oatly in 12/12 AI responses and on its one readable page, yet was tagged
    'mention' and the headline said "pitch — deepen coverage" at an outlet the
    brand already dominates.)
      brand on >=2 pages, OR on ALL readable pages   -> 'confirmed'
      brand on >=1 page but not all (and <2)         -> 'mention'
      pages read, brand on 0                         -> 'category'
      no pages read (pages_ok==0)                    -> leave _verify_brand_coverage's value
    """
    if not media_targets or not outlet_sov:
        return
    pe_by_dom = {}
    for r in outlet_sov:
        pe = r.get('page_evidence')
        if pe and (pe.get('pages_ok') or 0) > 0:
            pe_by_dom[(r.get('domain') or '').lower()] = pe
    changed = 0
    for t in media_targets:
        pe = pe_by_dom.get((t.get('domain') or '').lower())
        if not pe:
            continue
        bp = pe.get('brand_pages') or 0
        ok = pe.get('pages_ok') or 0
        # Confirmed = brand on >=2 pages OR on EVERY readable page (so a lone
        # fetchable page that features the brand counts, instead of being
        # under-rated to 'mention' and triggering a false guard downgrade).
        if bp == 0:
            new_cov = 'category'
        elif bp >= 2 or (ok >= 1 and bp >= ok):
            new_cov = 'confirmed'
        else:
            new_cov = 'mention'
        if t.get('coverage') != new_cov:
            changed += 1
        t['coverage'] = new_cov
        t['brand_confirmed'] = bp >= 1
        t['brand_page_mentions'] = pe.get('brand_mentions') or 0
    if changed:
        print(f"_reconcile_coverage_from_page_evidence: updated {changed} target(s) "
              f"from the (more thorough) page-evidence layer")


def _coverage_guard_verdicts(media_targets, outlet_sov, brand):
    """An outlet can only be a STRENGTH TO DEFEND if the brand is VERIFIED on
    its cited page (coverage='confirmed'). Without that, the share-of-voice
    'strength' is only AI co-occurrence — the AI mentions you alongside an
    outlet that doesn't actually cover you. That's a pitch target, not a
    relationship to defend.

    Downgrades any verdict='strength' row to 'opportunity' when the matched
    media target's `coverage` is 'category', 'mention', or 'unverified'.
    Rewrites the verdict_label to be honest about why. Makes Media Targets
    agree with Media Landscape so 'strength' = 'they cover you and you lead.'

    Only runs when coverage data is present (grounded audits). Non-grounded
    audits leave verdicts untouched (no page-fetch evidence to guard against).
    """
    if not media_targets or not outlet_sov:
        return
    cov_by_dom = {}
    for t in media_targets:
        dom = (t.get('domain') or '').lower()
        if dom and t.get('coverage'):
            cov_by_dom[dom] = t.get('coverage')
    if not cov_by_dom:
        return  # non-grounded audit; nothing to guard against
    b = brand or 'this brand'
    downgraded = []
    for row in outlet_sov:
        dom = (row.get('domain') or '').lower()
        cov = cov_by_dom.get(dom)
        # Downgrade only on EVIDENCE of absence — the page was fetched and the
        # brand genuinely isn't there ('category') or appears once in passing
        # ('mention'). 'unverified' means we COULDN'T see the page (bot-blocked,
        # timeout); asserting "the page doesn't cover the brand" from a failed
        # fetch is a false claim — observed on menshealth.com, which bot-stubs
        # datacenter IPs while its cited roundups feature the brand 100+ times.
        if row.get('verdict') == 'strength' and cov in ('category', 'mention'):
            # Don't downgrade a DOMINANT presence on a failed/partial page-scrape.
            # If the brand is named in a strong majority of the answers citing this
            # outlet, AI clearly ties the brand to its coverage — a sampled scrape
            # finding the brand on 0 pages is far more likely a bot-stub / JS render
            # / the brand-specific article not being in the sample than genuine
            # absence (e.g. JetBlue at thepointsguy.com: in 97% of citing answers,
            # with a dedicated launch article, yet scraped 0/6). Recommending 'pitch
            # them' when you're in nearly all their answers is just wrong.
            n_cit = row.get('responses_citing') or 0
            b_at = row.get('brand_mentions_at_outlet') or 0
            presence = (b_at / n_cit) if n_cit else 0.0
            if presence >= 0.5:
                row['_guard_skipped'] = f"{cov}@{round(presence, 2)}"
                continue
            row['_pre_guard_verdict'] = 'strength'  # diagnostic
            row['_guard_reason'] = cov
            row['verdict'] = 'opportunity'
            # Precise + direction-correct: AI names the brand when it cites this
            # outlet, but the pages it cites don't feature the brand, so that
            # visibility isn't anchored in the source. The play is to EARN
            # coverage here (coverage -> AI presence), and we only assert absence
            # for pages we actually fetched ('unverified' is excluded above).
            if cov == 'mention':
                row['verdict_label'] = (
                    f"AI names {b} when it cites this outlet, but the cited pages we "
                    f"checked mention {b} only in passing — the visibility isn't "
                    f"anchored in the source. Pitch to earn coverage that sticks."
                )
            else:
                row['verdict_label'] = (
                    f"AI names {b} when it cites this outlet, but the cited pages we "
                    f"checked don't mention {b} — the visibility isn't anchored in the "
                    f"source. Pitch to earn coverage that puts {b} on the page."
                )
            downgraded.append(f"{dom}({cov})")
    if downgraded:
        print(f"_coverage_guard_verdicts: downgraded {len(downgraded)} 'strength' -> "
              f"'opportunity' (no on-page coverage): {', '.join(downgraded[:10])}")


def _resort_sample_urls(media_targets, brand):
    """Surface brand-specific articles first in each outlet's sample_urls. A
    dedicated piece about the brand/launch (URL names the brand, e.g.
    .../jetblue-domestic-first-class-cabin) is better sample coverage than a generic
    category listicle, even one cited more often. Stable sort: preserves the existing
    (cited-frequency / LLM-selected) order within each group."""
    tokens = [f.lower() for f in _brand_match_forms(brand) if ' ' not in f]
    if not tokens:
        return
    for t in (media_targets or []):
        urls = t.get('sample_urls') or []
        if len(urls) > 1:
            t['sample_urls'] = sorted(
                urls, key=lambda u: 0 if any(tk in (u or '').lower() for tk in tokens) else 1)


def _category_keywords(category):
    """Pull noun-ish tokens from the category string for substring matching in
    _check_url_topic. Drops common stopwords; lowercases; caps at 10 keywords.

    Example: "AI creative platforms with transparency and commercial safety"
    -> ['creative', 'platforms', 'transparency', 'commercial', 'safety']
    ('ai' is too short and dropped by the 3-char minimum — 'creative' alone
    catches Adobe articles.)
    """
    if not category:
        return []
    STOPWORDS = {'and', 'or', 'the', 'for', 'with', 'a', 'an', 'in', 'of', 'on', 'to',
                 'is', 'are', 'as', 'at', 'by', 'be', 'this', 'that', 'these', 'those',
                 'best', 'top', 'leading'}
    toks = re.findall(r'[A-Za-z][A-Za-z\-]{2,}', category.lower())
    seen = set()
    result = []
    for t in toks:
        if t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        result.append(t)
    return result[:10]


def _apply_url_resolution(all_responses, url_map):
    """Rewrite each citation in-place: drop unverified, re-key valid ones under
    their final (post-redirect) URL + domain. Returns count of dropped URLs."""
    dropped = 0
    for r in all_responses:
        new_cits = []
        for c in r.get('citations', []) or []:
            final = url_map.get(c['url'])
            if not final:
                dropped += 1
                continue
            try:
                host = final.split('/')[2].lower()
                host = host[4:] if host.startswith('www.') else host
                if host in DOMAIN_BLACKLIST:
                    dropped += 1
                    continue
                new_cits.append({'url': final, 'domain': host})
            except Exception:
                dropped += 1
                continue
        r['citations'] = new_cits
    return dropped


TIER_CONFIG = {
    "free": {
        "prompt_count": 10,
        # Five-LLM mix: Claude/ChatGPT/Gemini covers the conversational big-3,
        # Perplexity adds retrieval-augmented search signal (structured
        # citations[] API), Grok adds breadth + xAI's real-time data lean.
        # 10 prompts × 5 LLMs = 50 responses per audit, yielding more
        # granular outlet-level share-of-voice than the prior 30-response
        # baseline. Perplexity + Grok credits funded by operator
        # (perplexity.ai/settings/api + openrouter.ai dashboard).
        "llms": ["Claude", "ChatGPT", "Gemini", "Perplexity", "Grok"],
        # Editorial media targets: bumped 5→10 so the SoV strength/opportunity
        # section has a meaningful denominator (was producing 1-3 cards on
        # narrow categories). Keeps the SoV scope aligned with the displayed
        # target list.
        "media_target_count": 10,
        "institutional_target_count": 5,
        "analyst_target_count": 5,
        # max_workers = 30 means all 10 prompts × 3 LLMs run truly concurrently
        # in a single batch round. With the bigger Standard instance (2 GB RAM,
        # 1 CPU) we have headroom for this. Was 10 workers × 3 batch rounds =
        # ~30s wall time; now 30 workers × 1 round = ~10-15s wall time on the
        # LLM batch step. Free-tier audits end-to-end should drop from ~75-120s
        # to ~45-75s.
        "max_workers": 30,
    },
    "paid": {
        "prompt_count": 100,
        # Perplexity temporarily disabled — API account out of credit (insufficient_quota).
        # To re-enable: top up at https://www.perplexity.ai/settings/api, then add
        # "Perplexity" back into this list. The provider call site in _call_llm is unchanged.
        "llms": ["Claude", "ChatGPT", "Gemini", "Grok"],
        "media_target_count": 25,
        "institutional_target_count": 10,
        "analyst_target_count": 10,
        "max_workers": 12,
    },
}


def _append_sources(text, urls):
    """Append retrieved source URLs as a Sources block so extract_urls() picks
    them up — mirrors the Gemini/Perplexity grounding-append pattern."""
    urls = [u for u in dict.fromkeys(urls) if u]
    if not urls:
        return text or ''
    return (text or '').rstrip() + "\n\nSources:\n" + "\n".join(f"- {u}" for u in urls)


def _annotation_urls(msg):
    """Harvest url_citation URLs from an OpenAI-compatible message.annotations
    list (used by ChatGPT search-preview + OpenRouter ":online")."""
    urls = []
    for ann in (getattr(msg, 'annotations', None) or []):
        atype = ann.get('type') if isinstance(ann, dict) else getattr(ann, 'type', None)
        if atype != 'url_citation':
            continue
        uc = ann.get('url_citation') if isinstance(ann, dict) else getattr(ann, 'url_citation', None)
        if not uc:
            continue
        u = uc.get('url') if isinstance(uc, dict) else getattr(uc, 'url', None)
        if u:
            urls.append(u)
    return urls


def _claude_grounded(prompt):
    """Claude Sonnet 4 with the native web_search tool, bounded to 3 searches
    (the bound is what keeps latency/memory sane — the reverted version ran
    unbounded). Builds the answer from text blocks + harvests cited URLs."""
    resp = anthropic.with_options(timeout=120.0).messages.create(
        model=CLAUDE_SONNET,
        max_tokens=2500,
        system=CITATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    )
    parts, urls = [], []
    for block in (resp.content or []):
        btype = getattr(block, 'type', None)
        if btype == 'text':
            parts.append(getattr(block, 'text', '') or '')
            for cit in (getattr(block, 'citations', None) or []):
                u = getattr(cit, 'url', None)
                if u:
                    urls.append(u)
        elif btype == 'web_search_tool_result':
            for item in (getattr(block, 'content', None) or []):
                u = getattr(item, 'url', None)
                if u:
                    urls.append(u)
    return _append_sources("".join(parts), urls)


def _chatgpt_grounded(prompt):
    """ChatGPT via the single-pass gpt-4o-search-preview model (one search pass,
    not the agentic Responses-API loop that OOM'd). Cited URLs come from
    message.annotations."""
    resp = openai_client.with_options(timeout=90.0).chat.completions.create(
        model="gpt-4o-search-preview",
        max_tokens=2500,
        web_search_options={},
        messages=[
            {"role": "system", "content": CITATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    msg = resp.choices[0].message
    return _append_sources(msg.content or '', _annotation_urls(msg))


def _grok_grounded(prompt):
    """Grok via OpenRouter's ":online" web plugin — adds live web search to the
    model with no separate xAI key. Cited URLs come from message.annotations."""
    base = os.environ.get("XAI_OPENROUTER_MODEL", "x-ai/grok-4.3")
    model = base if base.endswith(":online") else base + ":online"
    resp = openrouter_client.with_options(timeout=90.0).chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": CITATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    msg = resp.choices[0].message
    return _append_sources(msg.content or '', _annotation_urls(msg))


def _call_llm(provider, enriched_prompt):
    """Send one citation-forcing prompt to one provider; return (text, grounded).

    `grounded` is True when the response was produced with live web retrieval
    (always for Gemini/Perplexity; for Claude/ChatGPT/Grok only when ALL_GROUNDED
    is on AND the grounded call succeeded — a fallback to parametric sets it
    False).

    Citation-extraction strategy by provider:
      - Claude:     plain messages.create — URLs extracted from response text via regex.
      - ChatGPT:    plain chat.completions — same.
      - Gemini:     native `google_search` grounding (real, current URLs).
      - Perplexity: native `citations` array on the response object (real URLs;
                    lightweight, doesn't add latency).
      - Grok:       plain chat.completions via OpenRouter — regex only.

    HISTORICAL NOTE (2026-05-30): an earlier version tried Claude's
    `web_search_20250305` and ChatGPT's Responses-API `web_search` tools to
    get structured citation objects (less hallucination by construction). Both
    paths were reverted because: (a) the tool calls either errored on this
    account → fell through to a second non-tool call, doubling per-call
    latency, or (b) actually ran multiple web searches per call, blowing the
    free-tier 180s LLM-batch deadline AND triggering Render Starter OOMs from
    the larger structured payloads buffered by the SDKs. See the
    f543473→7f360e3→hotfix sequence of commits. Re-introduce the tool path
    only after profiling on a larger instance with a guard against double-call
    fallback.
    """
    if provider == "Claude":
        if ALL_GROUNDED:
            try:
                return (_claude_grounded(enriched_prompt), True)
            except Exception as e:
                print("Claude web_search failed; parametric fallback:", str(e)[:160])
        resp = anthropic.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=2000,
            system=CITATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": enriched_prompt}],
        )
        return (resp.content[0].text, False)
    if provider == "ChatGPT":
        if not openai_client:
            raise RuntimeError("OPENAI_API_KEY not configured")
        if ALL_GROUNDED:
            try:
                return (_chatgpt_grounded(enriched_prompt), True)
            except Exception as e:
                print("ChatGPT search-preview failed; parametric fallback:", str(e)[:160])
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": enriched_prompt},
            ],
        )
        return (resp.choices[0].message.content, False)
    if provider == "Gemini":
        if not gemini_client:
            raise RuntimeError("GEMINI_API_KEY not configured")
        gemini_prompt = enriched_prompt + "\n\nWhen providing information, please include specific sources and references. Format any sources as: Reference: [Source Name] - URL. Include authoritative sources like news sites, company websites, industry reports."
        # Try gemini-2.5-flash with Google Search grounding (yields real, current URLs).
        # Fall back to plain 2.0-flash without grounding if the SDK / model / tool isn't available.
        try:
            from google.genai import types as _genai_types
            grounding_config = _genai_types.GenerateContentConfig(
                tools=[_genai_types.Tool(google_search=_genai_types.GoogleSearch())],
                temperature=0.7,
            )
            resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=gemini_prompt,
                config=grounding_config,
            )
            text = resp.text or ""
            # Append grounding URIs so extract_urls() picks them up. The chunk
            # `.web.uri` is almost always a vertexaisearch.cloud.google.com
            # redirector — useless as a domain. Prefer any real source signal
            # the SDK exposes (`web.domain`, `web.url`, or a URL-shaped
            # `web.title`); fall back to the redirector URI only when no
            # better source is present (the URL-resolution pass later attempts
            # to follow the vertex meta-refresh to recover the real source).
            try:
                for cand in (resp.candidates or []):
                    gm = getattr(cand, 'grounding_metadata', None)
                    if not gm:
                        continue
                    for chunk in (getattr(gm, 'grounding_chunks', None) or []):
                        web = getattr(chunk, 'web', None)
                        if not web:
                            continue
                        uri = getattr(web, 'uri', None)
                        real_url = getattr(web, 'url', None)
                        real_domain = getattr(web, 'domain', None)
                        web_title = getattr(web, 'title', None)
                        appended = False
                        # 1) Explicit non-vertex URL field if present.
                        if real_url and 'vertexaisearch.cloud.google.com' not in real_url.lower():
                            text += f"\nSource: {real_url}"
                            appended = True
                        # 2) Bare domain — synthesize an https URL so extract_urls picks it up.
                        if not appended and real_domain:
                            domain_clean = real_domain.strip().rstrip('/')
                            if domain_clean and 'vertexaisearch' not in domain_clean.lower():
                                text += f"\nSource: https://{domain_clean}/"
                                appended = True
                        # 3) Some SDK builds put a URL in `title`. Use it only if it parses as a URL.
                        if not appended and web_title and web_title.lower().startswith(('http://', 'https://')):
                            text += f"\nSource: {web_title}"
                            appended = True
                        # 4) Last resort: the vertex redirector. The URL validator
                        # will try to follow the meta-refresh to recover the real source.
                        if not appended and uri:
                            text += f"\nSource: {uri}"
            except Exception:
                pass
            return (text, True)
        except Exception as e:
            print("Gemini grounding fallback:", e)
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=gemini_prompt,
            )
            return (resp.text, False)
    if provider == "Perplexity":
        if not perplexity_client:
            raise RuntimeError("PERPLEXITY_API_KEY not configured")
        resp = perplexity_client.chat.completions.create(
            model="sonar-pro",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": enriched_prompt},
            ],
        )
        text = resp.choices[0].message.content or ''
        # Perplexity returns a `citations` array on the top-level response
        # object (NOT on the message). These are the actual sources Perplexity
        # retrieved, so they're structurally non-hallucinated. Append as a
        # Sources block so extract_urls picks them up.
        structured_urls = []
        for attr in ('citations', 'search_results'):
            raw = getattr(resp, attr, None)
            if raw:
                for c in raw:
                    if isinstance(c, str):
                        structured_urls.append(c)
                    elif isinstance(c, dict):
                        u = c.get('url')
                        if u:
                            structured_urls.append(u)
                    else:
                        u = getattr(c, 'url', None)
                        if u:
                            structured_urls.append(u)
                if structured_urls:
                    break
        unique_urls = list(dict.fromkeys(structured_urls))
        if unique_urls:
            text = (text or '').rstrip() + "\n\nSources:\n" + "\n".join(f"- {u}" for u in unique_urls)
        return (text, True)
    if provider == "Grok":
        if not openrouter_client:
            raise RuntimeError("OPENROUTER_API_KEY not configured")
        if ALL_GROUNDED:
            try:
                return (_grok_grounded(enriched_prompt), True)
            except Exception as e:
                print("Grok :online failed; parametric fallback:", str(e)[:160])
        resp = openrouter_client.chat.completions.create(
            model=os.environ.get("XAI_OPENROUTER_MODEL", "x-ai/grok-4.3"),
            max_tokens=2000,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": enriched_prompt},
            ],
        )
        return (resp.choices[0].message.content, False)
    raise ValueError(f"Unknown provider: {provider}")


def _compute_audit_anomaly_flags(result, duration_seconds, per_provider_done, per_provider_errors):
    """Inspect a completed audit and return a list of anomaly flag strings.

    Empty list means everything looked normal. Each flag is a short uppercase
    code (BRAND_INVISIBLE, HIGH_LLM_ERROR_RATE, etc.) surfaced in the debug
    email subject + body.
    """
    flags = []
    try:
        total_responses = int(result.get("total_responses") or 0)
        brand_mentions = int(result.get("brand_mention_count") or 0)
        total_citations = int(result.get("total_citations_extracted") or 0)
        media_targets = result.get("media_targets") or []
        institutional_targets = result.get("institutional_targets") or []
        analyst_targets = result.get("analyst_targets") or []

        if brand_mentions == 0:
            flags.append("BRAND_INVISIBLE")
        elif total_responses > 0 and brand_mentions < (total_responses * 0.1):
            flags.append("LOW_BRAND_VISIBILITY")

        # Per-provider error rate. Use the per-provider DONE counts where
        # available so partial-completion batches don't false-flag every
        # provider as 100% error.
        for provider, errs in (per_provider_errors or {}).items():
            done = (per_provider_done or {}).get(provider, 0)
            if done > 0 and (errs / done) > 0.30:
                flags.append("HIGH_LLM_ERROR_RATE")
                break

        # LLM dominance: if one provider produced >70% of total citations.
        provider_citation_counts = {}
        for r in (result.get("all_responses") or []):
            n = len(r.get("citations") or [])
            provider_citation_counts[r.get("llm") or "?"] = provider_citation_counts.get(r.get("llm") or "?", 0) + n
        cite_total = sum(provider_citation_counts.values())
        if cite_total > 0:
            top = max(provider_citation_counts.values())
            if (top / cite_total) > 0.70:
                flags.append("SINGLE_LLM_DOMINANCE")

        if total_citations < 20:
            flags.append("LOW_CITATION_COUNT")

        if duration_seconds and duration_seconds > 600:
            flags.append("SLOW_AUDIT")

        if len(media_targets) < 3:
            flags.append("THIN_RESULTS")

        if not institutional_targets and not analyst_targets:
            flags.append("NO_INSTITUTIONAL_OR_ANALYST")

        # Partial-delivery flag: the LLM batch hit the wall-clock deadline and
        # cancelled in-flight calls. <95% = degraded; surfaces in the UI +
        # debug-email so we know to consider re-running.
        completion_rate = result.get("completion_rate")
        if isinstance(completion_rate, (int, float)) and completion_rate < 0.95:
            flags.append("PARTIAL_DELIVERY")
    except Exception as e:
        print("anomaly-flag computation failed:", e)
    return flags


class AuditAnalysisError(Exception):
    """Raised when the analysis-step Claude response cannot be parsed as JSON.

    Carries the raw response text and the parser's debug trace so the SSE
    worker can email diagnostics to the owner before refunding the credit.
    """

    def __init__(self, message, raw_response="", debug_info=None):
        super().__init__(message)
        self.raw_response = raw_response or ""
        self.debug_info = debug_info or {}


def _parse_analysis_json(raw_text, retry_callback=None):
    """Try multiple strategies to extract valid JSON from Claude's response.

    Returns (parsed_dict, debug_info_dict). Raises AuditAnalysisError on
    irrecoverable failure. debug_info has keys: strategy_used, attempts,
    raw_first_500, raw_last_500.
    """
    debug = {
        'attempts': [],
        'raw_first_500': raw_text[:500],
        'raw_last_500': raw_text[-500:],
        'raw_length': len(raw_text),
    }

    def attempt(name, text):
        try:
            d = json.loads(text)
            debug['attempts'].append({'name': name, 'ok': True})
            debug['strategy_used'] = name
            return d
        except Exception as e:
            debug['attempts'].append({'name': name, 'ok': False, 'err': str(e)[:200]})
            return None

    # Strategy 1: strip ```json fences if present, parse as-is.
    cleaned = raw_text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)

    d = attempt('cleaned_strip_fences', cleaned)
    if d:
        return d, debug

    # Strategy 2: find first { ... last } via regex.
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        d = attempt('regex_first_to_last_brace', m.group())
        if d:
            return d, debug

    # Strategy 3: fix common errors (trailing commas before } or ]).
    if m:
        candidate = re.sub(r',(\s*[}\]])', r'\1', m.group())
        d = attempt('fix_trailing_commas', candidate)
        if d:
            return d, debug

    # Strategy 4: if it looks truncated (more { than }), try to close it.
    if m:
        candidate = m.group()
        if candidate.count('{') > candidate.count('}'):
            for trunc_marker in ['"\n  ]', '"\n]', '"\n  }', '"\n}', '"}', '"]']:
                idx = candidate.rfind(trunc_marker)
                if idx > 0:
                    truncated = candidate[:idx + len(trunc_marker)]
                    opens = truncated.count('{')
                    closes = truncated.count('}')
                    truncated = truncated + ('}' * (opens - closes))
                    d = attempt(f'truncate_at_{trunc_marker[:5]!r}_pad_braces', truncated)
                    if d:
                        return d, debug

    # Strategy 5: ask Claude to fix its own broken JSON.
    if retry_callback:
        try:
            fixed = retry_callback(raw_text, debug['attempts'])
            d = attempt('claude_retry', fixed)
            if d:
                return d, debug
            # If the retry came back but still wouldn't parse, try the same
            # strip-fences logic on it as well.
            fixed_clean = (fixed or '').strip()
            if fixed_clean.startswith('```'):
                fixed_clean = re.sub(r'^```(?:json)?\s*\n?', '', fixed_clean)
                fixed_clean = re.sub(r'\n?```\s*$', '', fixed_clean)
                d = attempt('claude_retry_stripped', fixed_clean)
                if d:
                    return d, debug
        except Exception as e:
            debug['attempts'].append({'name': 'claude_retry', 'ok': False, 'err': str(e)[:200]})

    last_err = (debug['attempts'][-1] if debug['attempts'] else {}).get('err', 'unknown')
    raise AuditAnalysisError(
        f"Failed to parse analysis JSON after {len(debug['attempts'])} strategies. Last error: {last_err}",
        raw_response=raw_text,
        debug_info=debug,
    )


def _retry_analysis_json(raw_text, prior_attempts):
    """Ask Claude to fix its own broken JSON. Returns the corrected text."""
    last_err = (prior_attempts[-1] if prior_attempts else {}).get('err', 'unknown')
    fix_resp = anthropic.messages.create(
        model=CLAUDE_SONNET,
        max_tokens=8000,
        timeout=150.0,  # explicit: the client default (60s) was too short to
                        # regenerate a large JSON object, so the repair itself
                        # timed out — turning a fixable truncation into a hard
                        # audit failure (observed on the ServiceNow category).
        messages=[{
            "role": "user",
            "content": (
                f"The following response was supposed to be valid JSON but "
                f"failed to parse with error: {last_err}\n\n"
                f"Fix the JSON and return ONLY the corrected JSON object (no "
                f"markdown fences, no explanation). Preserve all the data; "
                f"just fix syntax errors and add any missing closing "
                f"braces/brackets.\n\nBroken response:\n{raw_text}"
            ),
        }]
    )
    return fix_resp.content[0].text


def _send_mail_object(msg):
    """Single email transport for every sender in the app. Takes an
    already-constructed sendgrid Mail object (all call sites build one) and
    ships it via Resend when RESEND_API_KEY is set, else via SendGrid — so the
    provider is swappable with one env var and zero call-site changes. Every
    site's key-gate checks (RESEND_API_KEY or SENDGRID_API_KEY), so setting
    either turns all email paths on. Raises on failure; call sites already
    wrap sends in try/except and log."""
    payload = msg.get()   # sendgrid Mail serializes to the v3 API JSON shape
    resend_key = os.environ.get("RESEND_API_KEY")
    if resend_key:
        pers = (payload.get('personalizations') or [{}])[0]
        frm = payload.get('from') or {}
        body = {
            "from": (f"{frm.get('name')} <{frm.get('email')}>"
                     if frm.get('name') else frm.get('email')),
            "to": [t.get('email') for t in (pers.get('to') or []) if t.get('email')],
            "subject": payload.get('subject') or pers.get('subject') or '(no subject)',
        }
        for c in payload.get('content') or []:
            if c.get('type') == 'text/plain':
                body['text'] = c.get('value')
            elif c.get('type') == 'text/html':
                body['html'] = c.get('value')
        r = requests.post("https://api.resend.com/emails",
                          headers={"Authorization": f"Bearer {resend_key}",
                                   "Content-Type": "application/json"},
                          json=body, timeout=15)
        if r.status_code >= 300:
            raise RuntimeError(f"Resend {r.status_code}: {r.text[:300]}")
        return r
    sg_key = os.environ.get("SENDGRID_API_KEY")
    if not sg_key:
        raise RuntimeError("no email provider configured (RESEND_API_KEY / SENDGRID_API_KEY)")
    from sendgrid import SendGridAPIClient
    return SendGridAPIClient(sg_key).send(msg)


def _send_audit_failure_email(error, problem_statement, tier):
    """Fire-and-forget diagnostic email when an audit fails (AuditAnalysisError
    or any other exception during run_citation_audit).

    Includes the raw Claude response (if available), the parse strategies
    attempted, and the problem statement. Skipped silently if SENDGRID_API_KEY
    or AUDIT_DEBUG_EMAIL is unset.
    """
    recipient = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not recipient or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        err_type = type(error).__name__
        err_msg = str(error)
        raw_response = getattr(error, 'raw_response', '') or ''
        debug_info = getattr(error, 'debug_info', {}) or {}

        # Cap raw response at 5000 chars to keep the email manageable.
        raw_clip = raw_response[:5000]
        raw_truncated_note = ""
        if len(raw_response) > 5000:
            raw_truncated_note = f"\n\n[...truncated; full length was {len(raw_response)} chars]"

        attempts = debug_info.get('attempts', [])
        strategy_lines = []
        for i, a in enumerate(attempts, 1):
            ok = "OK" if a.get('ok') else "FAIL"
            line = f"  {i}. [{ok}] {a.get('name', '?')}"
            if not a.get('ok'):
                line += f" — {a.get('err', '?')}"
            strategy_lines.append(line)
        strategies_text = "\n".join(strategy_lines) or "  (no attempts recorded)"

        subject = f"[Signal Finder · {tier} · FAILED] {err_type}: {err_msg[:80]}"

        text_lines = [
            "PR Signal Finder audit FAILED",
            "",
            f"Error type: {err_type}",
            f"Error: {err_msg}",
            f"Tier: {tier}",
            "",
            f"Problem statement: {problem_statement[:500]}",
            "",
            "Parse strategies attempted:",
            strategies_text,
            "",
            f"Raw Claude response length: {debug_info.get('raw_length', len(raw_response))}",
            "",
            "Raw response (first 5000 chars):",
            "---",
            raw_clip + raw_truncated_note,
            "---",
        ]
        text_body = "\n".join(text_lines)

        html_body = (
            f'<h3 style="color:#b00020">PR Signal Finder audit FAILED</h3>'
            f'<p><strong>Error type:</strong> {html.escape(err_type)}<br>'
            f'<strong>Error:</strong> {html.escape(err_msg)}<br>'
            f'<strong>Tier:</strong> {html.escape(tier)}</p>'
            f'<p><strong>Problem statement:</strong> {html.escape(problem_statement[:500])}</p>'
            f'<h4>Parse strategies attempted</h4>'
            f'<pre style="background:#f4f4f4;padding:10px;font-size:12px">{html.escape(strategies_text)}</pre>'
            f'<h4>Raw Claude response (first 5000 chars; full length {debug_info.get("raw_length", len(raw_response))})</h4>'
            f'<pre style="background:#f4f4f4;padding:10px;font-size:11px;white-space:pre-wrap;word-wrap:break-word">{html.escape(raw_clip + raw_truncated_note)}</pre>'
        )

        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[recipient],
            subject=subject,
            plain_text_content=text_body,
            html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("Audit FAILURE-email send failed:", e)


def _send_audit_debug_email(result, duration_seconds, per_provider_done, per_provider_errors, tier, problem_statement=""):
    """Fire-and-forget summary email to the owner after each audit.

    Skipped silently if SENDGRID_API_KEY or AUDIT_DEBUG_EMAIL is unset.
    Never raises — all errors are caught and logged.
    """
    recipient = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not recipient or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        brand = result.get("brand") or "(unknown brand)"
        category = result.get("category") or "(unknown category)"
        slug = result.get("slug") or "(no slug)"
        total_responses = int(result.get("total_responses") or 0)
        brand_mentions = int(result.get("brand_mention_count") or 0)
        total_citations = int(result.get("total_citations_extracted") or 0)
        competitors = result.get("competitors") or []
        media_targets = result.get("media_targets") or []
        institutional_targets = result.get("institutional_targets") or []
        analyst_targets = result.get("analyst_targets") or []
        domains = result.get("raw_citation_domains") or []
        base = (SIGNAL_BASE_URL or "").rstrip("/")
        if not base:
            try:
                base = request.url_root.rstrip("/")
            except Exception:
                base = ""
        report_link = f"{base}/signal/{slug}" if slug != "(no slug)" else "(unavailable)"
        json_link = f"{base}/signal/{slug}.json" if slug != "(no slug)" else "(unavailable)"

        flags = _compute_audit_anomaly_flags(result, duration_seconds, per_provider_done, per_provider_errors)
        flags_label = ", ".join(flags) if flags else "OK"

        # Per-provider error summary.
        err_lines = []
        for provider, done in (per_provider_done or {}).items():
            errs = (per_provider_errors or {}).get(provider, 0)
            err_lines.append(f"{provider}: {errs} errors / {done}")
        err_summary = " · ".join(err_lines) if err_lines else "(no provider data)"

        # Top 3 outlets / competitors.
        top_outlets = sorted(domains, key=lambda d: d.get("count", 0), reverse=True)[:3]
        top_outlet_lines = [f"{d.get('domain', '?')} ({d.get('count', 0)})" for d in top_outlets] or ["(none)"]
        top_competitors = sorted(competitors, key=lambda c: c.get("mention_count", 0), reverse=True)[:3]
        top_comp_lines = [f"{c.get('name', '?')} ({c.get('mention_count', 0)})" for c in top_competitors] or ["(none)"]

        subject = f"[Signal Finder · {tier}] {brand} — {flags_label}"

        # Plain text body
        text_lines = [
            f"PR Signal Finder audit completed",
            f"",
            f"Brand: {brand}",
            f"Category: {category}",
            f"Tier: {tier}",
            f"Slug: {slug}",
            f"Report: {report_link}",
            f"Raw JSON: {json_link}",
            f"",
            f"Duration: {duration_seconds:.1f}s" if duration_seconds else "Duration: (unknown)",
            f"",
            f"Stats:",
            f"  Total LLM responses: {total_responses}",
            f"  Brand mentions: {brand_mentions} / {total_responses}",
            f"  Total citations extracted: {total_citations}",
            f"  Competitors surfaced: {len(competitors)}",
            f"  Editorial targets: {len(media_targets)}",
            f"  Institutional targets: {len(institutional_targets)}",
            f"  Analyst targets: {len(analyst_targets)}",
            f"",
            f"Per-provider errors: {err_summary}",
            f"",
            f"Top 3 outlets:",
        ] + [f"  - {l}" for l in top_outlet_lines] + [
            f"",
            f"Top 3 competitors:",
        ] + [f"  - {l}" for l in top_comp_lines] + [
            f"",
            f"Anomaly flags: {flags_label}",
        ]
        if problem_statement:
            text_lines += ["", f"Problem statement: {problem_statement[:500]}"]
        text_body = "\n".join(text_lines)

        # HTML body
        flag_html = ""
        if flags:
            flag_html = (
                '<p style="color:#b00020;font-weight:700;margin:10px 0">'
                'Anomaly flags: ' + html.escape(flags_label) + '</p>'
            )
        else:
            flag_html = (
                '<p style="color:#2a7a3a;font-weight:600;margin:10px 0">Anomaly flags: OK</p>'
            )
        report_html = (
            f'<a href="{html.escape(report_link)}">{html.escape(report_link)}</a>'
            if slug != "(no slug)" else '<em>(unavailable)</em>'
        )
        json_html = (
            f'<a href="{html.escape(json_link)}">{html.escape(json_link)}</a>'
            if slug != "(no slug)" else '<em>(unavailable)</em>'
        )
        outlets_html = "".join(f"<li>{html.escape(l)}</li>" for l in top_outlet_lines)
        comps_html = "".join(f"<li>{html.escape(l)}</li>" for l in top_comp_lines)
        problem_html = (
            f'<p style="color:#555"><strong>Problem statement:</strong> {html.escape(problem_statement[:500])}</p>'
            if problem_statement else ''
        )
        duration_html = f'{duration_seconds:.1f}s' if duration_seconds else '(unknown)'

        html_body = (
            f'<h3>PR Signal Finder audit completed</h3>'
            f'{flag_html}'
            f'<p><strong>Brand:</strong> {html.escape(brand)}<br>'
            f'<strong>Category:</strong> {html.escape(category)}<br>'
            f'<strong>Tier:</strong> {html.escape(tier)}<br>'
            f'<strong>Slug:</strong> {html.escape(slug)}<br>'
            f'<strong>Duration:</strong> {duration_html}</p>'
            f'<p><strong>Report:</strong> {report_html}<br>'
            f'<strong>Raw JSON:</strong> {json_html}</p>'
            f'<h4>Stats</h4>'
            f'<ul style="margin:0;padding-left:20px">'
            f'<li>Total LLM responses: {total_responses}</li>'
            f'<li>Brand mentions: {brand_mentions} / {total_responses}</li>'
            f'<li>Total citations extracted: {total_citations}</li>'
            f'<li>Competitors surfaced: {len(competitors)}</li>'
            f'<li>Editorial targets: {len(media_targets)}</li>'
            f'<li>Institutional targets: {len(institutional_targets)}</li>'
            f'<li>Analyst targets: {len(analyst_targets)}</li>'
            f'</ul>'
            f'<p><strong>Per-provider errors:</strong> {html.escape(err_summary)}</p>'
            f'<h4>Top 3 outlets</h4><ul style="margin:0;padding-left:20px">{outlets_html}</ul>'
            f'<h4>Top 3 competitors</h4><ul style="margin:0;padding-left:20px">{comps_html}</ul>'
            f'{problem_html}'
        )

        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[recipient],
            subject=subject,
            plain_text_content=text_body,
            html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("Audit debug-email send failed:", e)


def _generate_audit_prompts(problem_statement, tier="free"):
    """Run ONLY the prompt-generation phase. Returns {brand, category, prompts}.

    Extracted from run_citation_audit so the paid-tier prompt editor can call
    this without the LLM batch / analysis steps. Tier governs prompt_count.
    """
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    prompt_count = cfg["prompt_count"]

    prompt_gen_response = anthropic.messages.create(
        model=CLAUDE_SONNET,
        max_tokens=4000 if prompt_count > 20 else 2000,
        messages=[{
            "role": "user",
            "content": f"""You are an AI citation strategist. A communications professional wants their brand to be the answer when people ask AI assistants about a specific problem.

Their goal: "{problem_statement}"

Today's date is {_today_label()}. Frame the prompts as if a real person is searching right now — so when a prompt uses words like "latest", "recent", "today's", or "current", it should pull sources from the past 12 months, not 2023 or 2024.

Your job:
1. Identify the BRAND from their statement (the company/product they want to promote).
2. Identify the CATEGORY or PROBLEM SPACE.
3. Generate exactly {prompt_count} prompts that a real person would type into ChatGPT, Claude, or Gemini when researching this problem space. These should be natural, varied prompts — some broad ("best X for Y"), some question-based ("how do I solve Y?"), some recommendation-seeking ("what do experts recommend for Y?"). At higher prompt counts, cover adjacent angles: pricing, alternatives, troubleshooting, expert opinions, regulatory considerations, real-world reviews, integration questions, regional perspectives. Where appropriate, include time-current framing like "in {_today_label()}", "this year", or "the latest" — not year-specific shorthand.

CRITICAL — EVERY PROMPT MUST BE BRAND-AGNOSTIC (name NO brands at all):
- Do NOT name the searched brand ("{{brand}}") in ANY prompt. Not once. Every prompt is a neutral category query a person would type WITHOUT knowing the brand exists.
- Do NOT name any competitor brand either. NEVER write "BrandA vs BrandB" — that pre-anchors the responses.
- WHY: naming the brand in a prompt guarantees the AI mentions it back, which inflates the brand's measured visibility and makes the competitive comparison unfair (the brand gets prompted mentions while competitors are only ever organic). By keeping EVERY prompt brand-agnostic, we measure the TRUE unprompted mindshare — does AI surface this brand on its own merits when nobody named it? — and every brand (the client's and all competitors) is measured on identical footing.
- So instead of "{{brand}} reviews" or "{{brand}} alternatives", write the underlying category question: "best [category] for [audience]", "most trusted [category] brands this year", "what do experts recommend for [problem]", "[category] with [the attribute the brand wants to own]".
- The brand name is captured separately in the "brand" field below for the analysis step — it just never appears inside a prompt.

Respond with ONLY valid JSON in this exact format:
{{
  "brand": "the brand name",
  "category": "the problem/category",
  "prompts": ["prompt 1", "prompt 2", ... "prompt {prompt_count}"]
}}"""
        }]
    )
    prompt_gen_text = prompt_gen_response.content[0].text
    json_match = re.search(r'\{.*\}', prompt_gen_text, re.DOTALL)
    if not json_match:
        raise ValueError("Failed to parse prompt generation response")
    prompt_data = json.loads(json_match.group())
    return {
        "brand": prompt_data["brand"],
        "category": prompt_data["category"],
        "prompts": prompt_data["prompts"][:prompt_count],
    }


def _generate_announcement_prompts(brand, announcement_text, product_name=None, tier="free"):
    """Announcement-Anchored mode: from a specific launch/announcement, derive the
    SPECIFIC consumer category it enters (the report's scope — auto-fixes the generic-
    title problem) and generate brand-agnostic category prompts, so the brand and its
    new product surface ORGANICALLY in AI answers only if the launch actually landed.
    Returns {brand, category, product_name, announcement_summary, prompts}. Same
    unbranded discipline as _generate_audit_prompts."""
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    prompt_count = cfg["prompt_count"]
    pname = (product_name or '').strip()
    resp = anthropic.messages.create(
        model=CLAUDE_SONNET,
        max_tokens=4000 if prompt_count > 20 else 2000,
        messages=[{"role": "user", "content": f"""You are an AI citation strategist measuring whether a specific product ANNOUNCEMENT has propagated into AI assistants' answers.

Brand: "{brand}"
{f'New product / launch name: "{pname}"' if pname else ''}
The announcement:
\"\"\"{(announcement_text or '')[:4000]}\"\"\"

Today's date is {_today_label()}. Frame prompts as a real person searching right now (so "latest"/"current" pulls the past 12 months).

Your job:
1. From the announcement, identify the SPECIFIC consumer CATEGORY the launch competes in — the narrow category a shopper actually searches, NOT the broad industry. (e.g. a new lie-flat domestic premium seat -> "domestic first class", not "airlines"; a new telehealth GLP-1 program -> "online weight-loss prescriptions", not "healthcare".) This is the report's scope.
2. Write a one-sentence neutral SUMMARY of what was announced.
3. Generate exactly {prompt_count} brand-agnostic prompts a real person would type when researching THAT category — so the brand surfaces organically only if the launch has landed in AI answers.

CRITICAL — EVERY PROMPT MUST BE BRAND-AGNOSTIC (name NO brands: NOT "{brand}", NOT the product "{pname}", NOT any competitor):
- Naming the brand or product guarantees AI echoes it back, inflating the result and destroying it as a "did the launch land organically" signal.
- Write the underlying category question: "best [category] for [audience]", "what's new in [category] {_today_label()}", "most [attribute] [category] options this year", "which [category] should I choose".
- Several prompts SHOULD probe what's NEW / recently launched / changed in the category (so a just-launched entrant can surface) — WITHOUT naming it.

Respond with ONLY valid JSON:
{{
  "brand": "{brand}",
  "category": "the specific consumer category",
  "product_name": "{pname}",
  "announcement_summary": "one neutral sentence",
  "prompts": ["prompt 1", "... up to prompt {prompt_count}"]
}}"""}]
    )
    txt = resp.content[0].text
    m = re.search(r'\{.*\}', txt, re.DOTALL)
    if not m:
        raise ValueError("Failed to parse announcement prompt generation response")
    data = json.loads(m.group())
    return {
        "brand": data.get("brand") or brand,
        "category": data.get("category") or "",
        "product_name": data.get("product_name") or pname,
        "announcement_summary": data.get("announcement_summary") or "",
        "prompts": [p for p in (data.get("prompts") or []) if isinstance(p, str) and p.strip()][:prompt_count],
    }


# ── Ground-truth QA pass ─────────────────────────────────────────────────────
# Recounts every displayed number against the saved all_responses and flags an
# audit as NOT client-ready if a number-integrity check fails. Deterministic —
# no LLM call, so it's cheap enough to run on every audit automatically. See
# ~/Desktop/count_bug_fix_brief.md for the incidents this codifies.
# Shared with _apply_display_editorial_filter (the actual display-time filter)
# so the QA check always agrees with what a viewer sees. Two kinds of entries:
# low-authority vendor/agency sites, and CONFIRMED competitors whose domain
# surfaces as a media target without the competitor being extracted into the
# top-10 competitors[] list (so the name-match filter can't catch them) — e.g.
# Kyndryl's report cited ensono.com (a direct managed-services rival) as a
# "pitch target"; its own citation rationale even called it "a competitor," but
# it never made the top-10 cut. Curated, not auto-detected — extend on sighting.
_LOWQ_MEDIA_DOMAINS = {
    'amasty.com', 'classicinformatics.com', 'xcelacore.com', 'managedsolution.com',
    'in-com.com', 'gocorptech.com', 'auxis.com', 'ensono.com',
}
_CORP_CONTENT_SUFFIXES = (' blog', ' resource center', ' resource hub')


def _qa_wb(term, case_sensitive=False):
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])',
                      0 if case_sensitive else re.IGNORECASE)


def _qa_union(forms):
    return re.compile('|'.join(r'(?<![A-Za-z0-9])' + re.escape(f) + r'(?![A-Za-z0-9])' for f in forms
                              if f), re.IGNORECASE)


def _qa_audit(payload):
    """Ground-truth QA validation for a completed audit. Returns
    {checks[], flags[], client_ready, blocking_failures, corrected}. `corrected`
    carries the ground-truth values (brand count, per-LLM, standings) to use
    when the stored ones fail a check. Safe to call on any payload shape;
    missing fields degrade gracefully rather than raising."""
    ar = payload.get('all_responses') or []
    brand = (payload.get('brand') or '').strip()
    aliases = payload.get('brand_aliases') or []
    domains = payload.get('brand_domains') or []
    category = (payload.get('category') or '')
    stored_brand = payload.get('brand_mention_count')
    total = payload.get('total_responses') or len(ar)
    competitors = payload.get('competitors') or []
    per_llm = payload.get('per_llm_visibility') or []
    media_targets = payload.get('media_targets') or []
    owned = payload.get('owned') or {}
    checks, flags, corrected = [], [], {}

    def rc(pat):
        return sum(1 for r in ar if pat.search(r.get('response') or ''))

    def lowercase_dominated(term):
        ci = rc(_qa_wb(term)); cs = rc(_qa_wb(term, case_sensitive=True))
        return ci, cs, ((ci - cs) >= 3 and cs < ci * 0.5)

    BLOCKING = {'brand_recount', 'alias_quality', 'alias_domain_hallucination', 'brand_generic_overcount',
                'competitor_subbrand_parity', 'competitor_generic_overcount', 'owned_domain_exact',
                'subbrand_undercount'}

    def add(name, ok, observed, expected, note=''):
        checks.append({'check': name, 'pass': bool(ok), 'blocking': name in BLOCKING,
                       'observed': observed, 'expected': expected, 'note': note})
        if not ok:
            flags.append(name)

    # ---- verified aliases (reuses the same guard as the live counting path) ----
    def alias_verdict(a):
        if len(a) < 3:
            return False, 'under 3 chars'
        if a.lower() in _GENERIC_ALIAS_WORDS:
            return False, 'generic English word'
        ci, cs, dom = lowercase_dominated(a)
        if ci == 0:
            return False, 'zero mentions in raw text'
        if dom:
            return False, 'lowercase-dominated (common word)'
        return True, ''
    verified, bad = [], []
    for a in aliases:
        ok, why = alias_verdict(a)
        (verified if ok else bad).append((a, why))
    verified_names = [a for a, _ in verified]

    # ---- brand distinctive forms: full name + proper distinctive tokens + acronym ----
    forms = [brand]
    for t in re.findall(r"[A-Za-z][\w&'-]*", brand):
        if len(t) < 3:
            continue
        proper = re.search(r'[a-z][A-Z]', t) or (t.isupper() and len(t) >= 3)
        if proper:
            lower_cs = rc(re.compile(r'(?<![A-Za-z0-9])' + re.escape(t.lower()) + r'(?![A-Za-z0-9])'))
            if lower_cs < 3 and t.lower() not in _GENERIC_ALIAS_WORDS:
                forms.append(t)
    words = [w for w in re.findall(r"[A-Za-z]+", brand) if w.lower() not in ('the', 'of', 'and', '&')]
    if len(words) >= 2:
        acr = ''.join(w[0] for w in words).upper()
        if len(acr) >= 3 and rc(_qa_wb(acr, case_sensitive=True)) >= 2:
            forms.append(acr)
    all_forms = list(dict.fromkeys(forms + verified_names))
    union = _qa_union(all_forms)

    # ---- CHECK 1: brand recount + per-LLM ----
    true_brand = rc(union)
    true_pl = {}
    for r in ar:
        l = r.get('llm'); true_pl.setdefault(l, [0, 0]); true_pl[l][1] += 1
        if union.search(r.get('response') or ''):
            true_pl[l][0] += 1
    stored_pl = {x.get('llm'): x.get('mentions') for x in per_llm}
    pl_mismatch = {l: (stored_pl.get(l), true_pl[l][0]) for l in true_pl
                   if stored_pl.get(l) is not None and stored_pl.get(l) != true_pl[l][0]}
    total_diff = abs((stored_brand or 0) - true_brand)
    add('brand_recount', total_diff <= 1 and not pl_mismatch, f"stored {stored_brand}/{total}",
        f"true {true_brand}/{total}", f"per-LLM mismatch {pl_mismatch}" if pl_mismatch else f"diff {total_diff}")
    corrected['brand_mention_count'] = true_brand
    corrected['per_llm'] = {l: true_pl[l][0] for l in true_pl}
    corrected['forms_used'] = all_forms

    # ---- CHECK 2: alias substring inflation / quality ----
    add('alias_quality', not bad, f"bad {[a for a, _ in bad]}", "all >=3ch, in-raw, non-generic",
        '; '.join(f"{a} ({w})" for a, w in bad))

    # ---- CHECK 3: alias / domain hallucination ----
    zero_aliases = [a for a in aliases if rc(_qa_wb(a)) == 0]
    bl = re.sub(r'[^a-z0-9]', '', brand.lower())
    alias_labels = [re.sub(r'[^a-z0-9]', '', a.lower()) for a in aliases]
    def domain_plausible(dm):
        label = re.sub(r'[^a-z0-9]', '', dm.split('.')[0].lower())
        if label and (label in bl or bl.startswith(label) or bl.endswith(label) or label in alias_labels):
            return True
        return rc(re.compile(re.escape(dm), re.IGNORECASE)) > 0
    bad_domains = [dm for dm in domains if not domain_plausible(dm)]
    add('alias_domain_hallucination', not zero_aliases and not bad_domains,
        f"zero-mention aliases {zero_aliases}, off-brand domains {bad_domains}", "none")

    # ---- CHECK 4: brand-name generic-word overcount ----
    bshort = brand.split()[0] if brand else ''
    gflag, gnote = False, ''
    if bshort and len(bshort) <= 6 and len(brand.split()) == 1:
        ci, cs, dom = lowercase_dominated(bshort)
        if dom:
            gflag, gnote = True, f"'{bshort}' CI {ci} vs CS {cs} — generic-word inflation"
    add('brand_generic_overcount', not gflag, gnote or "proper noun", "no generic inflation", gnote)

    # ---- competitor symmetric recount (name + curated sub-brands; generic-word guard) ----
    def comp_count(name, with_subs=True):
        low = name.lower()
        subs = [s for s, par in _COMPETITOR_SUBBRAND_PARENTS.items() if par.lower() == low] if with_subs else []
        base = [name] + subs
        ci = rc(_qa_union(base))
        if len(name.split()) == 1 and len(name) <= 6:
            _, _, dom = lowercase_dominated(name)
            if dom:
                cs_pat = re.compile('|'.join(r'(?<![A-Za-z0-9])' + re.escape(x) + r'(?![A-Za-z0-9])' for x in base))
                return rc(cs_pat)
        return ci

    top = sorted(competitors, key=lambda c: -(c.get('mention_count') or 0))[:5]
    brand_credited_subs = any(
        a.lower() in _COMPETITOR_SUBBRAND_PARENTS and _COMPETITOR_SUBBRAND_PARENTS[a.lower()].lower() == brand.lower()
        for a in verified_names)

    # ---- CHECK 5: competitor sub-brand parity ----
    # Compares what's actually STORED/displayed for the competitor against the
    # symmetric recount (name + its known sub-brands) — NOT name-only vs
    # with-subs, which would fire even after _merge_competitor_subbrands has
    # already folded the sub-brand in correctly (a false positive that blocked
    # an already-fixed BMS rerender).
    parity = []
    if brand_credited_subs:
        for c in top:
            subs = [s for s, par in _COMPETITOR_SUBBRAND_PARENTS.items() if par.lower() == c['name'].lower()]
            if not subs:
                continue
            stored_c = c.get('mention_count') or 0
            with_s = comp_count(c['name'], with_subs=True)
            if with_s - stored_c >= 3:
                parity.append(f"{c['name']}: stored {stored_c} -> should be {with_s} (sub-brands not credited)")
    add('competitor_subbrand_parity', not parity, parity or "symmetric",
        "stored competitor counts already credit sub-brands", "; ".join(parity))

    # ---- CHECK 6: competitor generic-word overcount ----
    generic = []
    for c in top:
        nm = c['name']
        if len(nm.split()) == 1 and len(nm) <= 6:
            ci, cs, dom = lowercase_dominated(nm)
            if dom and abs((c.get('mention_count') or 0) - cs) >= 3:
                generic.append(f"{nm}: stored {c.get('mention_count')} vs company {cs}")
    add('competitor_generic_overcount', not generic, generic or "clean",
        "distinctive-form competitor counts", "; ".join(generic))

    standings = sorted(
        [(brand, true_brand)] + [(c['name'], comp_count(c['name'], with_subs=brand_credited_subs)) for c in top],
        key=lambda x: -x[1])
    corrected['standings'] = standings

    # ---- CHECK 7: supplier-as-competitor (warn) ----
    cat = category.lower()
    infra = any(w in cat for w in ('data center', 'data centre', 'datacenter', 'cloud', 'infrastructure',
                                   'compute', 'hosting', 'colocation', 'gpu'))
    brand_hw = brand.lower() in _COMPONENT_SUPPLIERS or any(
        w in cat for w in ('chip', 'semiconductor', 'processor', 'silicon'))
    suppliers = [c['name'] for c in competitors
                if c['name'].lower() in _COMPONENT_SUPPLIERS] if (infra and not brand_hw) else []
    add('supplier_as_competitor', not suppliers, suppliers or "none", "no suppliers in infra category", str(suppliers))

    # ---- CHECK 8: media-target quality (warn) ----
    comp_labels = {re.sub(r'[^a-z0-9]', '', c['name'].lower()) for c in competitors if c.get('name')}
    def media_bad(t):
        dom = (t.get('domain') or '').lower(); outlet = (t.get('outlet') or '').strip().lower()
        reg = re.sub(r'[^a-z0-9]', '', dom.split('.')[0]) if dom else ''; reasons = []
        if reg and reg in comp_labels and dom not in EDITORIAL_OUTLETS:
            reasons.append('competitor-owned')
        if dom in _LOWQ_MEDIA_DOMAINS:
            reasons.append('low-quality vendor/agency')
        if outlet.endswith(_CORP_CONTENT_SUFFIXES) and dom not in EDITORIAL_OUTLETS:
            reasons.append('corporate blog')
        return reasons
    media_flags = [f"{t.get('domain')} ({', '.join(r)})" for t in media_targets if media_bad(t)]
    add('media_target_quality', not media_flags, media_flags or "clean",
        "no competitor/low-quality domains", "; ".join(media_flags))

    # ---- CHECK 9: owned-domain exact/brand match ----
    owned_rows = (owned.get('owned_sov') or {}).get('rows') or []
    brand_row = next((r for r in owned_rows if r.get('is_brand')), None)
    brand_regs = {re.sub(r'[^a-z0-9]', '', x.split('.')[0].lower()) for x in domains}
    brand_regs.add(bl)
    owned_bad = []
    if brand_row:
        for dd in (brand_row.get('domains') or []):
            dm = (dd.get('domain') or '').lower()
            reg = re.sub(r'[^a-z0-9]', '', dm.split('.')[-2] if dm.count('.') >= 1 else dm)
            if reg and (reg in brand_regs or any(reg in br or br in reg for br in brand_regs if len(br) >= 4)):
                continue
            owned_bad.append(dm)
    add('owned_domain_exact', not owned_bad, f"non-brand owned domains {owned_bad}",
        "owned rows all brand-owned", "; ".join(owned_bad))

    # ---- CHECK 10: outlet-SoV / earned_wins teaser vs raw text (warn) ----
    ew_issue = []
    for w in ((payload.get('earned_wins') or {}).get('wins') or [])[:6]:
        dom = (w.get('outlet') or '').lower()
        stored_b = w.get('brand_mentions')
        if stored_b is None or not dom:
            continue
        citing = [r for r in ar if any(dom in (c.get('url') or c.get('domain') or '') for c in (r.get('citations') or []))]
        raw_b = sum(1 for r in citing if union.search(r.get('response') or ''))
        if citing and abs(raw_b - stored_b) >= 2:
            ew_issue.append(f"{dom}: stored {stored_b} vs raw {raw_b} of {len(citing)}")
    add('outlet_sov_vs_raw', not ew_issue, ew_issue or "consistent", "teasers match raw counts", "; ".join(ew_issue))

    # ---- CHECK 11: sub-brand undercount (brand's own PRIMARY sub-brand identity) ----
    prim = [s for s, par in _COMPETITOR_SUBBRAND_PARENTS.items()
           if par.lower() == brand.lower() and s.lower() not in {a.lower() for a in verified_names}]
    # Only the curated ETF/finance sub-brand pairs represent a brand's own primary
    # identity (BlackRock->iShares); pharma flagship-drug entries are handled via
    # brand_aliases already, so re-checking them here would double-count noise.
    _PRIMARY_ONLY = {'ishares', 'spdr', 'spdrs'}
    prim = [s for s in prim if s.lower() in _PRIMARY_ONLY]
    ok11, note11 = True, ''
    if prim:
        with_p = rc(_qa_union(all_forms + prim))
        if (stored_brand or 0) < with_p - 1:
            ok11, note11 = False, f"stored {stored_brand} < brand+{prim} {with_p}"
    add('subbrand_undercount', ok11, note11 or "n/a", "brand counts its primary sub-brands", note11)

    # ---- CHECK 12: delivery health (warn) ----
    grounded = sum(1 for r in ar if r.get('grounded')); errors = sum(1 for r in ar if r.get('error'))
    rate = payload.get('completion_rate')
    dh = (errors / max(1, len(ar))) <= 0.30 and (rate is None or rate >= 0.95)
    add('delivery_health', dh, f"grounded {grounded}/{len(ar)}, errors {errors}, rate {rate}",
        "grounded, <5% error", 'PARTIAL_DELIVERY/HIGH_ERROR' if not dh else '')

    blocking_fail = [c['check'] for c in checks if c['blocking'] and not c['pass']]
    return {'checks': checks, 'flags': flags, 'client_ready': not blocking_fail,
            'blocking_failures': blocking_fail, 'corrected': corrected}


def run_citation_audit(problem_statement, on_progress=None, tier="free", prompts_override=None):
    """Full agent pipeline: one prompt in, ranked media/partnership/analyst lists out.

    If `prompts_override` is provided (paid-tier prompt-editor flow), the
    prompt-generation step is skipped and the supplied prompts are used as-is.
    Brand + category are derived from a lightweight extraction call so the
    downstream analysis still gets them.
    """
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    prompt_count = cfg["prompt_count"]
    llms = cfg["llms"]
    media_limit = cfg["media_target_count"]
    institutional_limit = cfg["institutional_target_count"]
    analyst_limit = cfg["analyst_target_count"]
    max_workers = cfg["max_workers"]
    if ALL_GROUNDED:
        # Grounded calls buffer larger structured payloads (search results +
        # citations) in the SDKs; cap concurrency so peak memory stays bounded
        # (concurrency × payload size was part of the original Starter OOM).
        max_workers = min(max_workers, 8)

    def emit(step, detail, current=0, total=0):
        if on_progress:
            on_progress(step, detail, current, total)

    if prompts_override:
        # User-curated prompts (paid tier prompt-editor flow). Still need
        # brand + category for the analysis prompt; derive them from the
        # problem statement via a small Claude call.
        emit("prompts", "Preparing your curated prompts...", 0, 1)
        try:
            brand_resp = anthropic.messages.create(
                model=CLAUDE_SONNET,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f'From this brand-positioning statement, extract the BRAND name and the '
                        f'CATEGORY/problem space. Respond with ONLY valid JSON: '
                        f'{{"brand": "...", "category": "..."}}\n\n'
                        f'Statement: "{problem_statement}"'
                    ),
                }]
            )
            brand_text = brand_resp.content[0].text
            bm = re.search(r'\{.*\}', brand_text, re.DOTALL)
            if not bm:
                raise ValueError("brand extraction parse failed")
            bd = json.loads(bm.group())
            brand = bd["brand"]
            category = bd["category"]
        except Exception:
            # Fallback: strip leading "I want " / verbs from the statement.
            brand = problem_statement[:60].strip()
            category = "the brand's category"
        prompts = [p for p in prompts_override if isinstance(p, str) and p.strip()]
        if not prompts:
            raise ValueError("No prompts supplied for the audit.")
        emit("prompts", f"Using your {len(prompts)} curated prompts for \"{brand}\"", 1, 1)
    else:
        emit("prompts", "Generating search prompts...", 0, 1)
        gen = _generate_audit_prompts(problem_statement, tier=tier)
        brand = gen["brand"]
        category = gen["category"]
        prompts = gen["prompts"]
        emit("prompts", f"Generated {len(prompts)} prompts for \"{brand}\"", 1, 1)

    tasks = [(provider, pi, prompt_text)
             for pi, prompt_text in enumerate(prompts)
             for provider in llms]
    total_calls = len(tasks)

    time_note = _time_aware_note()

    def run_one(provider, pi, prompt_text):
        enriched = prompt_text + CITATION_SUFFIX + time_note
        try:
            resp_text, grounded = _call_llm(provider, enriched)
            return {"llm": provider, "prompt": prompt_text, "response": resp_text, "citations": extract_urls(resp_text), "grounded": grounded, "error": None}
        except Exception as e:
            return {"llm": provider, "prompt": prompt_text, "response": f"[Error: {e}]", "citations": [], "grounded": False, "error": str(e)[:120]}

    all_responses = []
    completed = 0
    per_provider_errors = {p: 0 for p in llms}
    per_provider_done = {p: 0 for p in llms}
    progress_lock = threading.Lock()

    # Global LLM-batch deadline. Sized generously per tier: free should fit
    # easily in 2 min; paid gets 15 min to protect the slow tail (Grok averages
    # 5-30s/call, and 12 workers can't drain 400 calls in 6 min — at 360s we
    # were shipping 57% completion on Adobe paid audits). Happy-path paid
    # audits still finish in 3-5 min; the higher ceiling only kicks in on slow
    # inference days. If one provider hangs, we still proceed to analysis with
    # the partial responses we've collected.
    if ALL_GROUNDED:
        # Grounded calls run live web searches (slower: ~15-40s each), so the
        # batch needs a longer wall-clock ceiling than the parametric path.
        batch_deadline_seconds = 420 if tier == "free" else 1200
    else:
        batch_deadline_seconds = 180 if tier == "free" else 900

    emit("llm", f"Querying {len(llms)} LLMs × {len(prompts)} prompts ({total_calls} calls)...", 0, total_calls)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_map = {executor.submit(run_one, *t): t for t in tasks}
        deadline = datetime.utcnow() + timedelta(seconds=batch_deadline_seconds)
        timed_out = False
        for fut in as_completed(future_map.keys(), timeout=batch_deadline_seconds):
            try:
                result = fut.result()
            except Exception as e:
                # Shouldn't happen since run_one catches, but defensively log.
                provider, pi, ptxt = future_map[fut]
                result = {"llm": provider, "prompt": ptxt, "response": f"[Error: {e}]", "citations": [], "error": str(e)[:120]}
            with progress_lock:
                all_responses.append(result)
                completed += 1
                per_provider_done[result["llm"]] = per_provider_done.get(result["llm"], 0) + 1
                if result.get("error"):
                    per_provider_errors[result["llm"]] = per_provider_errors.get(result["llm"], 0) + 1
                error_suffix = ""
                err_summary = [f"{p}: {n}" for p, n in per_provider_errors.items() if n > 0]
                if err_summary:
                    error_suffix = f" · errors → {', '.join(err_summary)}"
                emit(
                    "llm",
                    f"Completed {completed}/{total_calls} ({result['llm']}){error_suffix}",
                    completed,
                    total_calls,
                )
            if datetime.utcnow() >= deadline:
                timed_out = True
                break
    except FuturesTimeoutError:
        timed_out = True
    finally:
        if timed_out and completed < total_calls:
            # Cancel still-pending futures + report partial completion.
            cancelled = 0
            for f, t in future_map.items():
                if not f.done():
                    if f.cancel():
                        cancelled += 1
            emit(
                "llm",
                f"Batch deadline hit; proceeding with {completed}/{total_calls} responses ({cancelled} cancelled)",
                completed,
                total_calls,
            )
        # Don't wait for in-flight workers — shutdown(wait=False) lets the executor
        # release without blocking on hung Grok calls. Cancelled futures won't run.
        executor.shutdown(wait=False, cancel_futures=True)

    # Treat ERRORED responses like timed-out ones: NOT delivered. They were
    # being kept in all_responses and counted as "delivered non-mentions",
    # which deflated every brand's mindshare and fabricated false "dark on X"
    # findings when a whole provider failed. Observed: Grok returned 402
    # 'insufficient credits' on all 10 calls -> 10 empty responses -> the report
    # claimed "0/10 on Grok" as a real signal AND held the denominator at 50
    # instead of the 40 actually delivered. Drop errored responses so every
    # downstream metric (citations, mindshare, per-LLM, SoV, completion) is over
    # responses that actually came back; keep the per-provider error counts for
    # diagnostics + the partial-delivery indicator.
    _errored = [r for r in all_responses if r.get('error')]
    if _errored:
        all_responses = [r for r in all_responses if not r.get('error')]
        _dead = sorted({r.get('llm') for r in _errored
                        if per_provider_errors.get(r.get('llm'), 0) >= len(prompts)})
        print(f"[audit] dropped {len(_errored)} errored responses "
              f"(fully-failed providers: {_dead or 'none'}); metrics now over "
              f"{len(all_responses)} delivered responses")

    # URL verification consumes 80% of the extract step's progress budget; aggregation is the rest.
    # We size the step "total" relative to URL count so per-URL progress emits map cleanly.
    all_urls = list({c['url'] for r in all_responses for c in (r.get('citations') or [])})
    url_total = len(all_urls)
    # Step total = url_total + 1 (the trailing aggregation tick). Lets ETA interpolate smoothly.
    extract_total = max(1, url_total + 1)
    emit("extract", f"Verifying {url_total} cited URLs (filtering 404s & confabulations)...", 0, extract_total)

    if all_urls:
        def url_progress(done, total):
            emit("extract", f"Verifying URLs ({done}/{total})...", done, extract_total)
        url_map = _resolve_and_verify_urls(all_urls, on_progress=url_progress)
        dropped = _apply_url_resolution(all_responses, url_map)
        emit("extract", f"Verified {url_total - dropped} of {url_total} URLs (dropped {dropped} dead/confabulated)", url_total, extract_total)
    else:
        emit("extract", "No URLs to verify", 0, extract_total)

    ranked_domains = aggregate_citations(all_responses)
    # Promote outlets that are NAMED in responses (or cited via bare domain)
    # but whose URLs were hallucinated/pruned — so search-grounded models' prose
    # citations (WWD, Forbes, TechCrunch…) count instead of silently vanishing.
    # Allowlist-gated, so it cannot introduce non-editorial noise.
    ranked_domains = _augment_citations_with_named_outlets(ranked_domains, all_responses)
    total_citations = sum(d['count'] for d in ranked_domains)
    brand_aliases = _resolve_brand_aliases(brand)   # sub-brands (e.g. iShares for BlackRock)
    brand_aliases = _verify_brand_aliases(brand_aliases, all_responses)  # drop hallucinated/generic aliases
    brand_mention_count = sum(1 for r in all_responses
                              if _brand_present_in_text(_brand_match_forms(brand, brand_aliases), _response_text(r)))
    # Per-assistant visibility — computed here (before the analysis prompt) so
    # the summary can call out lopsided concentration. (Re-stored on the
    # analysis dict below for the UI.)
    _per_llm_vis = _compute_per_llm_visibility(brand, all_responses, aliases=brand_aliases)
    _per_llm_read = _llm_visibility_read(brand, _per_llm_vis)
    _per_llm_facts = "; ".join(f"{x['llm']} {x['mentions']}/{x['total']}" for x in _per_llm_vis) or "(n/a)"
    emit("extract", f"Found {total_citations} citations across {len(ranked_domains)} domains · brand cited in {brand_mention_count}/{len(all_responses)} responses", extract_total, extract_total)

    emit("analysis", "Identifying competitor brands...", 0, 1)
    # Deterministic competitor counts: discover candidates via Claude (judgement),
    # then count via _count_brand_mentions (regex over FULL response text). This
    # kills the prior failure mode where the analysis prompt's competitor mention
    # counts were derived from 500-char excerpts and didn't reconcile against
    # the raw responses a consultant might grep.
    competitor_candidates = _extract_competitor_candidates(brand, category, all_responses)
    competitor_counts = []
    for name in competitor_candidates:
        cnt = _count_brand_mentions(name, all_responses)
        if cnt > 0:
            competitor_counts.append({"name": name, "mention_count": cnt})
    competitor_counts.sort(key=lambda c: c["mention_count"], reverse=True)
    # Drop chip/hardware suppliers (NVIDIA, AMD...) BEFORE the top-10 cut so real
    # competitors backfill the freed slots instead of being crowded out.
    competitor_counts = _drop_supplier_noncompetitors(brand, category, competitor_counts)
    competitor_counts = competitor_counts[:10]
    # Merge same-entity duplicates (e.g. 'JPMorgan' + 'J.P. Morgan Asset Management').
    competitor_counts = _dedupe_competitors(competitor_counts, all_responses)
    # Pull the brand's own alias/parent out of the competitive set (e.g. 'Zendesk'
    # when the brand is 'Ultimate (Zendesk Ultimate)') so the analysis never frames
    # the brand as a rival of its own parent. Kept separately as related_brands.
    competitor_counts, related_brands = _split_self_referential_competitors(brand, competitor_counts)
    competitor_counts = _merge_competitor_subbrands(competitor_counts, all_responses)  # SPDR -> State Street
    # Classify each as brand_peer / retailer / marketplace so the SoV math and
    # the UI can treat them differently — retailers aren't real PR competitors.
    competitor_counts = _classify_competitor_types(brand, category, competitor_counts)
    top_competitor_block = "\n".join(
        f"  - {c['name']}: cited in {c['mention_count']} of {len(all_responses)} responses "
        f"({round(c['mention_count'] / len(all_responses) * 100) if all_responses else 0}%)"
        for c in competitor_counts if c.get('type', 'brand_peer') == 'brand_peer'
    ) or "  (no clear competitor brands surfaced)"

    emit("analysis", "Verifying editorial sources...", 0, 1)

    # Defunct domains (KNOWN_DEFUNCT_ORGS) are intentionally absent from all
    # three lists — classify_citation_domain returns 'defunct' for them, which
    # matches none of the filters below. They never reach the analysis prompt,
    # so Claude can never recommend partnering with an org that no longer exists.
    editorial_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'editorial']
    institutional_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'institutional']
    analyst_domains_found = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'analyst']
    defunct_domains_found = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'defunct']
    if defunct_domains_found:
        print(f"classify_citation_domain: excluded {len(defunct_domains_found)} defunct domains: " +
              ", ".join(d['domain'] for d in defunct_domains_found[:10]))

    # Deterministic self-domain filter: drop the brand's own properties from
    # editorial + institutional lists before they reach Claude. You can't pitch
    # yourself, and seeing "developer.adobe.com" as a media target torpedoes
    # report credibility. Belt-and-suspenders with the Claude-side prompt
    # update inside verify_editorial_domains.
    self_domains_dropped = []
    if brand:
        before_e = len(editorial_domains)
        editorial_domains = [d for d in editorial_domains if not _is_brand_own_domain(d['domain'], brand)]
        self_domains_dropped.extend(
            d['domain'] for d in (ranked_domains)
            if classify_citation_domain(d['domain']) == 'editorial'
            and _is_brand_own_domain(d['domain'], brand)
        )
        before_i = len(institutional_domains)
        institutional_domains = [d for d in institutional_domains if not _is_brand_own_domain(d['domain'], brand)]
        if self_domains_dropped or before_i > len(institutional_domains):
            print(f"_is_brand_own_domain: dropped {before_e - len(editorial_domains)} editorial + "
                  f"{before_i - len(institutional_domains)} institutional self-domains for brand '{brand}': "
                  + ", ".join(self_domains_dropped[:10]))

    # Competitor-domain filter — drop sites that belong to a competitor's
    # corporate property (e.g. newsroom.rei.com when REI Co-op is a Patagonia
    # competitor; press.northface.com when North Face is a competitor). These
    # are technically "editorial-looking" subdomains but aren't pitchable —
    # you can't get earned media from a competitor's own communications team.
    # Applies the same logic the SoV computation uses, hoisted here so the
    # analysis prompt and the displayed media_targets list never see them.
    competitor_stems = _competitor_domain_stems(competitor_counts)
    if competitor_stems:
        before_e = len(editorial_domains)
        dropped_comp_editorial = [
            d['domain'] for d in editorial_domains
            if _is_competitor_owned_domain(d['domain'], competitor_stems)
        ]
        editorial_domains = [
            d for d in editorial_domains
            if not _is_competitor_owned_domain(d['domain'], competitor_stems)
        ]
        before_i = len(institutional_domains)
        institutional_domains = [
            d for d in institutional_domains
            if not _is_competitor_owned_domain(d['domain'], competitor_stems)
        ]
        if before_e > len(editorial_domains) or before_i > len(institutional_domains):
            print(f"_is_competitor_owned_domain: dropped {before_e - len(editorial_domains)} editorial + "
                  f"{before_i - len(institutional_domains)} institutional competitor-owned domains "
                  f"(stems: {sorted(competitor_stems)[:10]}): "
                  + ", ".join(dropped_comp_editorial[:10]))

    # AI verification: filter out B2B vendors / marketplaces / non-media domains the heuristic missed.
    editorial_domains, rejected_editorial = verify_editorial_domains(editorial_domains, brand, category)
    if rejected_editorial:
        print(f"verify_editorial_domains: filtered out {len(rejected_editorial)} non-media domains: " +
              ", ".join(d['domain'] for d in rejected_editorial[:10]))

    # Topic-verify each candidate sample URL. _is_specific_article only checks
    # URL shape — it can't tell a real-but-off-topic article (Verge piece on a
    # golf cart) from an LLM hallucination (creativebloq.com/news/adobe-ai-...
    # that 403s us). Topic verification GETs the page, parses title + first
    # ~1500 chars of body, and confirms the brand or a category keyword
    # actually appears. Off-topic URLs get dropped; inaccessible URLs (403,
    # 5xx, timeouts) get transitive trust if the same domain has at least one
    # other verified URL.
    #
    # We only touch the surfaced `sample_urls` (via `dom['specific_urls']`).
    # Raw evidence (`all_responses[*].citations`, `raw_citation_domains[*].urls`)
    # stays intact for the JSON download.
    emit("analysis", "Topic-verifying displayed sample URLs...", 0, 1)
    brand_lower = (brand or '').lower().strip()
    category_kw = _category_keywords(category)

    # Collect candidate URLs to verify: top 10 specific_urls per surviving
    # domain across all three classifications. Capped per-domain so a single
    # ranking-heavy outlet can't blow the GET budget.
    candidate_urls = []
    for dom in (editorial_domains + institutional_domains + analyst_domains_found):
        for u in (dom.get('specific_urls') or [])[:10]:
            candidate_urls.append(u)
    candidate_urls = list(set(candidate_urls))  # dedup

    url_topic_status = {}  # url -> 'verified' | 'off_topic' | 'inaccessible'
    # Hard-cap candidates AND bound the overall pass — otherwise a single slow
    # URL whose connection stalls past the 4s requests timeout (DNS, SSL, etc.)
    # blocks the as_completed loop. Anything not finished inside the deadline
    # is marked 'inaccessible' (transitive-trust rules then apply).
    candidate_urls = candidate_urls[:60]
    topic_deadline_seconds = 45
    if candidate_urls:
        topic_executor = ThreadPoolExecutor(max_workers=20)
        try:
            futures = {topic_executor.submit(_check_url_topic, u, brand_lower, category_kw): u
                       for u in candidate_urls}
            try:
                for fut in as_completed(futures, timeout=topic_deadline_seconds):
                    try:
                        status = fut.result()
                    except Exception:
                        status = 'inaccessible'
                    url_topic_status[futures[fut]] = status
            except FuturesTimeoutError:
                # Deadline fired. Mark any unfinished futures inaccessible so
                # the downstream filter still sees them, then cancel + bail.
                unfinished = 0
                for f, u in futures.items():
                    if not f.done():
                        url_topic_status[u] = 'inaccessible'
                        unfinished += 1
                        f.cancel()
                print(f"[audit] topic-verify deadline hit ({topic_deadline_seconds}s); "
                      f"{unfinished} URLs marked inaccessible")
        finally:
            topic_executor.shutdown(wait=False, cancel_futures=True)

    # Domains earn transitive trust if AT LEAST ONE of their URLs verified —
    # the LLM probably wasn't fabricating that domain wholesale.
    def _domain_of(u):
        try:
            from urllib.parse import urlparse
            h = (urlparse(u).netloc or '').lower()
            return h[4:] if h.startswith('www.') else h
        except Exception:
            return ''

    verified_domains = set()
    for u, status in url_topic_status.items():
        if status == 'verified':
            host = _domain_of(u)
            if host:
                verified_domains.add(host)

    # Rebuild each domain's specific_urls. A URL survives iff:
    #   - it's 'verified' itself, OR
    #   - it's 'inaccessible' AND its host is in verified_domains, OR
    #   - it wasn't a candidate (beyond top-10 per domain — leave untouched).
    # 'off_topic' and 'confabulated' are always dropped — the latter is a
    # definitive 404/410 from the publisher (LLM hallucination fingerprint)
    # and must not earn transitive trust.
    for dom in ranked_domains:
        new_specific = []
        for u in (dom.get('specific_urls') or []):
            status = url_topic_status.get(u)
            if status == 'verified':
                new_specific.append(u)
            elif status in ('off_topic', 'confabulated'):
                continue  # always drop
            elif status == 'inaccessible':
                host = _domain_of(u)
                if host and host in verified_domains:
                    new_specific.append(u)
                # else drop — no transitive trust for this domain.
            else:
                # Not in the candidate set (beyond top-10 per domain). Leave alone.
                new_specific.append(u)
        dom['specific_urls'] = new_specific

    v_count = sum(1 for s in url_topic_status.values() if s == 'verified')
    o_count = sum(1 for s in url_topic_status.values() if s == 'off_topic')
    c_count = sum(1 for s in url_topic_status.values() if s == 'confabulated')
    i_count = sum(1 for s in url_topic_status.values() if s == 'inaccessible')
    print(f"[audit] topic-verify: {v_count} verified / {o_count} off-topic / "
          f"{c_count} confabulated (404/410) / {i_count} inaccessible "
          f"(transitive-trust eligible) of {len(candidate_urls)} candidates")

    emit("analysis", "Building your signal report...", 0, 1)

    def fmt_block(domains, limit):
        # Prefer specific-article URLs for the sample shown to Claude — keeps
        # homepages / tag-index pages out of the recommended sample_urls. If a
        # domain has no specific URLs (only homepages were cited), fall back
        # to the full urls list so Claude still sees evidence.
        def sample(d):
            specific = d.get('specific_urls') or []
            return specific[:3] if specific else (d.get('urls') or [])[:3]
        return "\n".join(
            f"  {i+1}. {d['domain']} — cited {d['count']}x by {', '.join(d['llms'])} — sample URLs: {', '.join(sample(d))}"
            for i, d in enumerate(domains[:limit])
        ) or "  (none)"

    editorial_block = fmt_block(editorial_domains, max(15, media_limit * 2))
    institutional_block = fmt_block(institutional_domains, max(10, institutional_limit * 2))
    analyst_block = fmt_block(analyst_domains_found, max(10, analyst_limit * 2))

    # Per-response excerpt size auto-tunes to fit a ~19K-token budget for the
    # responses block. Free tier (30 responses) gets ~2500 chars each (5x more
    # context than the old 500-char clip); paid tier (~500 responses) collapses
    # to a 500-char floor so the prompt stays inside Claude's 200K context.
    total_chars_budget = 75_000  # roughly 19K tokens for the responses block
    per_response_chars = max(500, total_chars_budget // max(1, len(all_responses)))
    per_response_chars = min(per_response_chars, 3000)  # don't exceed 3000 even on small batches

    responses_block = ""
    for i, r in enumerate(all_responses):
        citation_urls = [c['url'] for c in r.get('citations', [])]
        responses_block += f"\n--- Response {i+1} [{r['llm']}] ---\nPrompt: {r['prompt']}\nCitations found: {citation_urls}\n{r['response'][:per_response_chars]}\n"

    # Tier-aware token budget. The 4K default was truncating paid-tier responses
    # mid-JSON (25 media + 10 institutional + 10 analyst + competitors + exec
    # summary won't fit). 8K gives Claude headroom; the prompt also tells it
    # to self-budget so it ships concise text rather than truncated JSON.
    # Free tier raised 5000 -> 7000: verbose categories (enterprise software,
    # agentic AI — many competitors + long rationales) were truncating the JSON
    # at 5000, which then failed every parse strategy and dragged the self-
    # repair call into a timeout (observed: ServiceNow audit died twice). 7000
    # gives the JSON room to close; the prompt still tells the model to self-
    # budget so it ships concise text rather than running to the ceiling.
    analysis_max_tokens = 8000 if tier == "paid" else 7000
    output_budget_chars = analysis_max_tokens * 4  # rough char approximation

    # Per-call timeout override. The Anthropic client's 60s default was triggering
    # APITimeoutError on heavy paid-tier analysis (500 responses worth of context +
    # 8K output tokens). 240s gives the call real headroom on slow inference days.
    analysis_call_timeout = 240.0 if tier == "paid" else 120.0

    _analysis_prompt_content = f"""You are an AI citation intelligence analyst. You have actual citation data extracted from 30 AI-generated responses (10 prompts × 3 LLMs: Claude, ChatGPT, Gemini) about the category "{category}".

The client's brand is "{brand}". Their goal: "{problem_statement}"

DETERMINISTIC FACTS (pre-computed from raw response text — use these EXACTLY, do not re-estimate):
- Brand mention count: {brand_mention_count} of {len(all_responses)} responses mention "{brand}" (deterministic substring count over full response text — authoritative).
- PER-ASSISTANT VISIBILITY (how many of EACH AI's responses mention the brand): {_per_llm_facts}. Read: {_per_llm_read}. If this is lopsided (the brand surfaces on one or two assistants and is absent from the rest), that is a CRITICAL finding for the executive_summary — it means the brand is search-surfaced but not embedded in the models people use most.
- TOP COMPETITOR MENTION COUNTS (pre-computed deterministically — these are the AUTHORITATIVE counts for the `competitors` array. Do NOT re-estimate, do NOT add competitors not in this list, do NOT change the counts):
{top_competitor_block}

The citation data has been pre-classified into three groups:

EDITORIAL DOMAINS (news, trade press, magazines, B2B publications — outlets a PR pro can actually pitch):
{editorial_block}

INSTITUTIONAL / ASSOCIATION DOMAINS (.gov, .edu, .mil, and non-profit/trade-association .orgs — sources of authority that require partnership, certification, sponsorship, or research collaboration, NOT traditional pitching):
{institutional_block}

ANALYST FIRMS (Gartner, Forrester, IDC, and major consultancies — influence via analyst briefings, inclusion in evaluations like Magic Quadrants / Waves / MarketScapes, sponsored research, and client subscriptions, NOT pitching):
{analyst_block}

Total unique domains cited: {len(ranked_domains)}
Total citation instances: {total_citations}

RESPONSE SUMMARIES (with extracted citations):
{responses_block}

Produce THREE target lists:

1. EXACTLY {media_limit} MEDIA TARGETS — drawn ONLY from the EDITORIAL DOMAINS above. These are pitch-able publications. RETURN {media_limit} TARGETS unless the editorial-domains list above has fewer than {media_limit} entries (in which case return all of them). Do NOT artificially trim to "high-confidence" picks; even 1x-cited outlets are legitimate targets if they're real editorial publications. The reader needs to see the full pitch landscape, not just the obvious wins.
2. AUTHORITY & PARTNERSHIP TARGETS — drawn ONLY from the INSTITUTIONAL / ASSOCIATION DOMAINS above. Up to {institutional_limit}, only if meaningfully present. Universities, agencies, certification bodies, trade associations, advocacy non-profits, or government bodies.
3. ANALYST TARGETS — drawn ONLY from the ANALYST FIRMS above. Up to {analyst_limit}. ABSOLUTE RULE: if fewer than 2 distinct analyst firms appear in the ANALYST FIRMS list above (or none of them have 2+ citations), return an EMPTY array — `[]`. Do NOT invent analyst recommendations for categories that don't have analyst coverage (e.g. hospitality, fashion, consumer products, food & beverage, design, lifestyle). Influence is via analyst relations, not pitching or partnership.

If a category has nothing, return an empty array for it. NEVER write speculative language like "the absence of analyst citations suggests..." or "limited B2B influence requires analyst investment" — silence is correct when the data is empty.

For each target:
- Map domain to its proper name (e.g. allure.com → Allure, nih.gov → National Institutes of Health, harvard.edu → Harvard University, omri.org → OMRI, garden.org → National Gardening Association, gartner.com → Gartner, forrester.com → Forrester)
- Note competitors discussed alongside it
- Explain the influence strategy (editorial: how a placement helps; authority/partnership: what specific partnership/certification/sponsorship/research collab; analyst: which evaluation to target, briefing cadence, or research sponsorship that would shift citations)

CRITICAL: Only include entities that actually appear in the citation data above. Do NOT invent or guess.

NULL-VALUED FIELDS: `gap_insight` is the ONE media_targets field allowed to be JSON null. Use null (not "" and not a generic 'opportunity' string) when the data does not support a concrete gap claim. All other string fields should always be populated.

INSTITUTIONAL TYPE GUIDANCE — get the classification right:
- Visit Seattle, NYC Tourism, Brand USA, etc. → "Destination Marketing Organization (DMO)", NOT "Government Agency". DMOs are quasi-public marketing entities, not government bodies. Partnership plays differ accordingly (DMOs run press trips, co-marketing, partner pages; agencies don't).
- If the entity publishes news, opinion, or industry coverage REGULARLY (HospitalityNet, Skift, PhocusWire, etc.) it is EDITORIAL — do NOT classify as institutional and do NOT include it in institutional_targets. It belongs in media_targets. Specifically: HospitalityNet is a trade PUBLICATION (industry news + opinion), not a trade association.
- Trade associations have members, lobbying, certifications, conferences. If the entity primarily PUBLISHES CONTENT, it is editorial, not institutional.

SAMPLE_URLS RULES — these get surfaced to the client as "evidence" so they must be high-quality:
- PREFER URLs that point to specific articles (paths like /2025/03/some-headline/ or /article/title-slug/). The sample URLs shown above for each domain have already been filtered to favor article-style paths.
- AVOID homepage URLs (just the bare domain), tag/category index pages (/tag/ai/, /category/news/), search-result pages, author archive pages.
- If a domain only offered homepage-style URLs in the sample, include AT MOST 1 — and prefer leaving sample_urls empty over filling it with misleading evidence. An empty array is correct; a homepage URL passed off as "evidence" is not.

CRITICAL OUTPUT BUDGET: Your entire response must fit in {analysis_max_tokens} tokens (~{output_budget_chars} characters) or less. Be CONCISE — short rationales, short gap_insights, no filler words. Better to ship complete JSON with terse text than truncated JSON with verbose text. The JSON MUST be syntactically complete with all closing braces and brackets.

Respond with ONLY valid JSON:
{{
  "brand": "{brand}",
  "category": "{category}",
  "brand_mention_count": {brand_mention_count},
  "total_responses": {len(all_responses)},
  "total_citations_extracted": {total_citations},
  "competitors": [
    {{"name": "competitor name (USE THE NAMES + COUNTS FROM 'TOP COMPETITOR MENTION COUNTS' above EXACTLY — do not invent or omit; fill `cited_by` from response inspection)", "mention_count": <int from pre-computed list>, "cited_by": ["Claude", "ChatGPT", "Gemini"]}}
  ],
  "media_targets": [
    {{
      "rank": 1,
      "outlet": "Publication name derived from actual citation domain",
      "domain": "the actual domain from citation data",
      "reporter": "Named journalist if identifiable from response content, otherwise null",
      "citation_frequency": <actual count from citation data>,
      "sample_urls": ["up to 10 actual specific-article URLs that were cited — see SAMPLE_URLS RULES above; prefer empty over homepage/tag-page filler"],
      "cited_by_llms": ["which LLMs cited this outlet"],
      "competitors_citing": ["competitors discussed in responses that cited this outlet"],
      "rationale": "One sentence on why earned media here would move the needle for {brand}",
      "gap_insight": "DATA-GROUNDED RULE: This field describes a CONCRETE editorial gap at this outlet. Allowed ONLY when you have evidence from the responses_block to support a specific claim. Otherwise return null (JSON null — NOT an empty string, NOT a generic 'opportunity' filler). Specifically:\n        - If competitors_citing is non-empty AND the responses_block names a specific format/topic where they win: write that gap. Example: 'Spanx dominates this outlet's best-of roundup format while {brand} appears only in standalone reviews.'\n        - If competitors_citing is empty AND you have evidence from the responses_block that {brand} is missing from a specific identifiable section: write that. Example: 'Brand absent from {{outlet}}'s Hotels vertical where competitors haven't established presence either.'\n        - In ALL other cases — INCLUDING outlets where the brand IS cited but you have no evidence of a specific gap — return JSON null.\n        - NEVER invent 'whitespace opportunity', 'untapped potential', 'pure whitespace', or 'open editorial territory' framing. These are filler and embarrass the report. If the data doesn't support a specific gap claim, omit it (return null)."
    }}
  ],
  "institutional_targets": [
    {{
      "rank": 1,
      "institution": "Proper name of the institution",
      "domain": "the actual domain from citation data",
      "type": "Government Agency | Destination Marketing Organization (DMO) | Academic / Research Institute | Trade Association | Certification Body | Advocacy Non-profit | Standards Body",
      "citation_frequency": <actual count from citation data>,
      "sample_urls": ["up to 10 actual specific-article URLs that were cited — see SAMPLE_URLS RULES above; prefer empty over homepage/tag-page filler"],
      "cited_by_llms": ["which LLMs cited this institution"],
      "partnership_play": "One sentence on the specific partnership, certification, sponsorship, research collaboration, or expert engagement that would influence AI citations through this entity"
    }}
  ],
  "analyst_targets": [
    {{
      "rank": 1,
      "firm": "Proper firm name",
      "domain": "the actual domain from citation data",
      "type": "Industry Analyst | Management Consultancy | Boutique Research",
      "citation_frequency": <actual count from citation data>,
      "sample_urls": ["up to 10 actual specific-article URLs that were cited — see SAMPLE_URLS RULES above; prefer empty over homepage/tag-page filler"],
      "cited_by_llms": ["which LLMs cited this firm"],
      "evaluations_referenced": ["Specific reports/evaluations cited if identifiable from response text — e.g. 'Magic Quadrant for APM', 'Forrester Wave: Observability' — otherwise empty array"],
      "analyst_play": "One sentence on the specific analyst relations move: which evaluation to target, briefing cadence to establish, sponsored research, or client subscription that would shift citations"
    }}
  ],
  "executive_summary": "EXACTLY 3 sentences — this is the 'What we found' headline a comms director will paste into a CMO briefing. Each sentence must carry a specific, non-obvious finding tied to the actual numbers. Lead with the single most important insight, not a throat-clearing preamble. STRUCTURE:\n  Sentence 1 — THE POSITION: {brand}'s AI mindshare ({brand_mention_count} of {len(all_responses)} responses) framed against the top competitor's count. If {brand} >= top competitor, lead with strength ('{brand} leads/holds the AI conversation in [category]…'); if behind, lead with the gap. NEVER say 'lacks authority' if the brand out-mentions competitors.\n  Sentence 2 — THE SURPRISE: the single most non-obvious thing in the data. STRONGLY PREFER the PER-ASSISTANT VISIBILITY finding when it's lopsided — if the brand surfaces on only one or two of the five assistants and is absent from the rest, lead the surprise with that (e.g. 'almost all of {brand}'s visibility is Gemini; it's absent from ChatGPT, Claude, and Grok'), because it means the brand is search-surfaced but not embedded in the models people use most. Otherwise: a specific outlet where the brand is absent but a competitor owns it; analyst firms (Gartner/Forrester/etc.) dominating citations over editorial press; a competitor you'd expect to lead that doesn't. Name the specific entity + number.\n  Sentence 3 — THE STAKE (the strategic implication, NOT a tactic — the single highest-leverage ACTION is shown separately on the page as the report's #1 move, so do NOT duplicate it, name its outlet, or use its action verbs). State what the position MEANS or what is at risk: e.g. the gap is structural (evenly thin across assistants) rather than platform-specific; the brand is discoverable but not yet the default answer; a competitive/category dynamic that defines the challenge or the window to act. Frame it to the brand's real standing — if it LEADS the category overall, a position to protect; if it TRAILS the leader (lower overall mindshare) — even when it leads the few outlets that cite it — a gap that won't close on its own (never 'top-tier'/'maintain' for a trailing brand). Diagnose, don't prescribe.\n  RULES: No filler ('this audit reveals…', 'in today's landscape…'). No generic 'competitors dominate' unless the per-outlet data supports it (empty competitors_citing = open whitespace, not a bloodbath). Discuss analysts ONLY if analyst_targets has entries OR analyst firms appear heavily in the citation data; silence is fine for consumer categories. Write it so a CMO who reads ONLY these 3 sentences still walks away with the strategic takeaway."
}}"""

    # Retry once with reduced output budget on APITimeoutError. The most common
    # cause of timeout is Claude generating until it hits max_tokens; trimming
    # the budget forces it to ship concise JSON inside the deadline.
    def _call_analysis(max_tok):
        return anthropic.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=max_tok,
            timeout=analysis_call_timeout,
            messages=[{"role": "user", "content": _analysis_prompt_content}],
        )

    def _extract_text(resp):
        """Concatenate all text blocks, defensively. An empty string here means
        the model returned no text (observed: claude-sonnet-4-6 occasionally
        returns an empty response under load) — which is NOT an exception, so it
        must be detected and retried explicitly or it silently kills the audit
        at JSON-parse time ('Expecting value: line 1 column 1 (char 0)')."""
        try:
            return "".join(
                getattr(b, 'text', '') or '' for b in (resp.content or [])
                if getattr(b, 'type', None) == 'text'
            ).strip()
        except Exception:
            return ""

    # Retry on BOTH transient exceptions (timeout/connection) AND empty
    # responses. Each kills an 8-min audit otherwise; the analysis call is the
    # single most expensive step to lose.
    analysis_text = ""
    _tok = analysis_max_tokens
    for _attempt in range(3):
        try:
            analysis_response = _call_analysis(_tok)
        except Exception as _e:
            _err_name = type(_e).__name__
            if ('Timeout' in _err_name or 'Connection' in _err_name) and _attempt < 2:
                _tok = max(4000, _tok // 2)
                print(f"[analysis] {_err_name} attempt {_attempt+1}; retry at max_tokens={_tok}")
                continue
            raise
        analysis_text = _extract_text(analysis_response)
        if analysis_text:
            break
        _stop = getattr(analysis_response, 'stop_reason', None)
        print(f"[analysis] EMPTY response on attempt {_attempt+1} "
              f"(stop_reason={_stop}); retrying")
    if not analysis_text:
        raise AuditAnalysisError("Analysis call returned empty text after 3 attempts")
    # Multi-strategy parse with a Claude self-repair fallback. Raises
    # AuditAnalysisError (which the SSE worker catches and emails diagnostics
    # for) if every strategy fails.
    analysis, _parse_debug = _parse_analysis_json(analysis_text, retry_callback=_retry_analysis_json)

    # Hard-overwrite competitor counts with the deterministic pre-computed values.
    # Even if Claude drifts on the counts (or invents extra competitors), the
    # final report can't disagree with what a regex would produce. Preserve
    # cited_by lists Claude produced, by name, when we have them.
    cited_by_lookup = {}
    for c in analysis.get('competitors', []) or []:
        try:
            cited_by_lookup[(c.get('name') or '').strip().lower()] = c.get('cited_by') or []
        except Exception:
            pass
    analysis['competitors'] = [
        {
            "name": c["name"],
            "mention_count": c["mention_count"],
            "cited_by": cited_by_lookup.get(c["name"].lower(), []),
        }
        for c in competitor_counts
    ]
    # Brand's own alias/parent names that surfaced as "competitors" (e.g. Zendesk
    # for an Ultimate audit) — kept for context, never framed as rivals.
    analysis['related_brands'] = related_brands

    # Owned Signal Finder lens — same data, owned-media focus.
    try:
        brand_domain_hints = list(_resolve_brand_domains(brand, category))
        for _al in (brand_aliases or []):   # sub-brand owned sites (e.g. ishares.com)
            for _dd in _resolve_brand_domains(_al, category):
                if _dd not in brand_domain_hints:
                    brand_domain_hints.append(_dd)
        brand_domain_hints = _verify_brand_domains(brand, brand_aliases, brand_domain_hints, all_responses)
        analysis['brand_domains'] = brand_domain_hints
        analysis['owned'] = _compute_owned_analysis(
            brand, competitor_counts, related_brands, ranked_domains, all_responses,
            brand_domain_hints=brand_domain_hints, brand_aliases=brand_aliases)
        _enrich_owned_recommendations(brand, category, analysis['owned'])
    except Exception as _oe:
        print("owned analysis failed (continuing):", str(_oe)[:120])

    # Deterministically reconcile each media target's `cited_by_llms` with the
    # detected citing set (URL OR name). Claude estimates this field from the
    # response excerpts and can drift; the augmented ranked_domains carries the
    # authoritative per-outlet LLM set, so the "cited by …" chips match the data.
    _dom_llms = {
        (d.get('domain') or '').lower(): (d.get('llms') or [])
        for d in ranked_domains
    }
    for _t in (analysis.get('media_targets') or []):
        _dom = (_t.get('domain') or '').lower()
        if _dom in _dom_llms and _dom_llms[_dom]:
            _t['cited_by_llms'] = _dom_llms[_dom]

    analysis["prompts_used"] = prompts
    # Full ranked domains list (no cap). Each dict already carries:
    # {domain, count, llms, prompts, urls, specific_urls, diversity_score} — preserve all.
    analysis["raw_citation_domains"] = ranked_domains
    # Full raw per-response data: {llm, prompt, response, citations}. Lets users
    # re-analyze, audit, debug, or rerun analysis on a past report.
    analysis["all_responses"] = all_responses
    # Editorial-media share-of-voice analysis. For each top editorial outlet,
    # computes the brand's mention rate WITHIN responses citing that outlet
    # vs overall, and the same for each competitor. Labels each outlet as
    # 'strength' (defend), 'opportunity' (pitch — competitor over-indexes),
    # or 'neutral'. The UI + CSV surface this as a separate section.
    # IMPORTANT: scoped to editorial_domains only (not all ranked_domains).
    # PR opportunities/strengths only apply to pitchable media outlets, not
    # .gov pages, .edu pages, brand-own domains, trade associations, etc.
    # Relevance filter: drop off-audience trade/industrial/professional outlets
    # (kept in raw_citation_domains for the CSV) so they don't surface as
    # media targets or share-of-voice rows.
    _cat = analysis.get("category") or ""
    _cat_l = _cat.lower()
    _is_cons = _category_is_consumer(_cat)
    editorial_domains = _filter_relevant_editorial(editorial_domains, _cat)
    # Also gate Claude-named targets through the editorial classifier (the
    # editorial_domains list is already gated this way, but the analysis call can
    # name a wire service / retailer / analyst / vendor as a "media target").
    # Keeps fresh-audit media_targets consistent with the rerender path.
    analysis["media_targets"] = [
        t for t in (analysis.get("media_targets") or [])
        if not _is_irrelevant_outlet(t.get("domain"), t.get("outlet"), _is_cons, _cat_l)
        and (not t.get("domain") or classify_citation_domain(t.get("domain")) == "editorial")
    ]
    # Scope SoV to exactly the media-target outlets the report shows, so every
    # card carries its competitor breakdown and the verdicts aren't capped out
    # by obscure non-target outlets. Fall back to the full editorial list only
    # if no targets map to cited domains.
    try:
        _sov_eds = _editorial_dicts_for_targets(ranked_domains, analysis.get("media_targets")) or editorial_domains
        # SoV compares the brand against PEER BRANDS only — retailers and
        # marketplaces are distribution channels, not direct competitors. They
        # still show in the report's "Also surfaced" chip row, just not in the
        # per-outlet 'vs' math.
        _peer_competitors = [c for c in competitor_counts if c.get('type', 'brand_peer') == 'brand_peer']
        analysis["outlet_sov"] = _compute_outlet_share_of_voice(
            brand, _peer_competitors, all_responses, _sov_eds,
            max_outlets=max(10, len(_sov_eds)), brand_aliases=brand_aliases
        )
    except Exception as _sov_e:
        print("outlet share-of-voice computation failed (continuing without):", _sov_e)
        analysis["outlet_sov"] = []
    # Page-evidence layer: scrape the cited pages behind each SoV outlet and
    # count every tracked brand on them — in-answer SoV vs on-page SoV.
    if any(r.get("grounded") for r in all_responses):
        try:
            _attach_page_evidence(analysis.get("outlet_sov"), ranked_domains,
                                  brand, competitor_counts)
        except Exception as _pe_e:
            print("page-evidence failed (continuing without):", _pe_e)
    # Brand-coverage verification (BACKSTAGE ONLY): fetch each target's sample
    # article URL(s) and record whether the brand is actually named on the page.
    # Kept in the saved payload / CSV for diagnostics, but it no longer drives
    # the report — PR teams know their own clips, and a single page-fetch was
    # under-crediting marquee outlets (Fast Company, Allure) that block bots or
    # run list pages. Only meaningful when grounded (real URLs).
    if any(r.get("grounded") for r in all_responses):
        try:
            _verify_brand_coverage(brand, analysis.get("media_targets") or [], ranked_domains)
        except Exception as _bc_e:
            print("brand-coverage verification failed (continuing without):", _bc_e)
    # Reconcile coverage from the (more thorough) page-evidence layer so the
    # guard, headline, and card labels all agree on whether the brand is on the
    # cited pages. Must run AFTER both coverage passes, BEFORE the guard.
    try:
        _reconcile_coverage_from_page_evidence(analysis.get("media_targets"), analysis.get("outlet_sov"))
    except Exception as _rc_e:
        print("coverage reconciliation failed (continuing without):", _rc_e)
    # Coverage-guard the SoV verdicts: an outlet can only be 'strength' if the
    # brand is on the cited page. Otherwise it's 'opportunity'. Keeps Media
    # Targets agreeing with Media Landscape so 'strength' honestly means 'they
    # cover you and you lead.'
    try:
        _coverage_guard_verdicts(analysis.get("media_targets"), analysis.get("outlet_sov"), brand)
    except Exception as _cg_e:
        print("coverage-guard failed (continuing without):", _cg_e)
    try:
        _resort_sample_urls(analysis.get("media_targets"), brand)
    except Exception as _ss_e:
        print("sample-url resort failed (continuing without):", _ss_e)
    # Rank targets by how prominently the AI surfaces them (responses citing +
    # citation frequency) — relevance + share-of-voice, not coverage tiers.
    try:
        _sort_targets_by_prominence(analysis.get("media_targets"), analysis.get("outlet_sov"))
    except Exception as _st_e:
        print("target prominence sort failed (continuing):", _st_e)
    # Cap to top 8 pitch targets — enough to name a multi-outlet "build" play in
    # thin deep-tech categories (where everything is cited 2-3x), without a long
    # noisy list. The full ranked editorial set is still in raw_citation_domains.
    _mt = analysis.get("media_targets") or []
    if len(_mt) > 8:
        analysis["media_targets"] = _mt[:8]
        print(f"capped media_targets {len(_mt)} -> 8 (raw set kept in CSV)")
    # Single highest-priority action — gives every dashboard one unmistakable
    # next step even when it's all-strength or all-emerging.
    try:
        analysis["headline_move"] = _compute_headline_move(
            brand, analysis.get("outlet_sov"), analysis.get("media_targets"))
    except Exception as _hm_e:
        print("headline-move computation failed (continuing without):", _hm_e)
        analysis["headline_move"] = None
    # Per-assistant visibility — which of the 5 AIs actually surface the brand.
    analysis["brand_aliases"] = brand_aliases
    try:
        analysis["per_llm_visibility"] = _compute_per_llm_visibility(brand, all_responses, aliases=brand_aliases)
        analysis["per_llm_read"] = _llm_visibility_read(brand, analysis["per_llm_visibility"])
    except Exception as _pv_e:
        print("per-LLM visibility computation failed (continuing without):", _pv_e)
        analysis["per_llm_visibility"] = []
        analysis["per_llm_read"] = None
    # Media Landscape — the client's essential outlet list (consumer/trade/
    # business/advocacy) with each outlet's AI presence status (driving / open
    # lane / off radar). Answers "why isn't WWD here?" honestly. One Claude call.
    try:
        analysis["media_landscape"] = _compute_media_landscape(
            brand, analysis.get("category"), all_responses, ranked_domains)
    except Exception as _ml_e:
        print("media-landscape computation failed (continuing without):", _ml_e)
        analysis["media_landscape"] = {}
    # Regenerate the executive summary now that share-of-voice is known, so it
    # LEADS WITH STRENGTHS / OPPORTUNITIES (the main analysis call ran before SoV
    # was computed, so its summary couldn't see where the brand over- or under-
    # indexes per outlet). One extra Claude call (~$0.01).
    if analysis.get("outlet_sov"):
        try:
            _regen = _regenerate_executive_summary(analysis)
            if _regen:
                analysis["executive_summary"] = _regen
        except Exception as _es_e:
            print("SoV-aware summary regen failed (keeping original):", _es_e)
    # Surface to the frontend so it can offer a "Download raw data" affordance.
    analysis["full_responses_available"] = True
    # Per-provider error counts (consumed by the debug-email summary). Leading
    # underscore so a future cleanup pass can strip non-public keys easily.
    analysis["_per_provider_errors"] = dict(per_provider_errors)
    analysis["_per_provider_done"] = dict(per_provider_done)

    # Audit completion tracking. Surfaces partial-delivery in the UI + saved
    # payload so users (and the debug email) can see when the wall-clock
    # deadline cancelled the slow tail. `completion_rate` is a 0.0-1.0 ratio.
    analysis["responses_completed"] = len(all_responses)
    analysis["responses_planned"] = total_calls
    completion_rate = round(len(all_responses) / max(1, total_calls), 3)
    analysis["completion_rate"] = completion_rate

    # Belt-and-suspenders: set tier here so it's guaranteed-set on the saved
    # payload even if a caller forgets to mutate the dict before persisting.
    # (Adobe paid audit saved with tier=None — this prevents recurrence.)
    analysis["tier"] = tier

    print(f"[audit] tier={tier} completion={completion_rate:.1%} ({len(all_responses)}/{total_calls})")

    # Ground-truth QA pass — runs on EVERY completed audit so a numbers bug never
    # silently reaches a prospect. Persisted on the payload; deterministic, no
    # extra API cost. See _qa_audit's docstring + count_bug_fix_brief.md.
    try:
        analysis['qa'] = _qa_audit(analysis)
        if not analysis['qa']['client_ready']:
            print(f"[qa] NOT client_ready: {analysis['qa']['blocking_failures']}")
    except Exception as _qa_e:
        print("qa_audit failed (continuing without):", str(_qa_e)[:200])
        analysis['qa'] = None

    return analysis


# MVP branch: free-tier rate limit + client-IP helper.
# Free audits allowed per client IP per day — the primary cost control now that
# email is optional. Tunable via FREE_DAILY_CAP env var without a redeploy.
FREE_DAILY_CAP = max(1, int(os.environ.get("FREE_DAILY_CAP", "3") or "3"))

# ── Launch hardening: concurrency guard + global daily ceiling ──────────────
# Each free audit is ~50 grounded LLM calls fanned out over 20-30 threads and is
# memory-heavy; running several at once has OOM'd the single 2GB instance. Cap
# SIMULTANEOUS audits with an in-process semaphore (single gunicorn process, so no
# Redis needed); excess requests get a friendly "in line" 503 and the client auto-
# retries. GLOBAL_DAILY_CAP is a cost / rate-limit circuit breaker across ALL IPs
# (hundreds of audits = real provider spend). Both tunable via env for the window.
MAX_CONCURRENT_AUDITS = max(1, int(os.environ.get("MAX_CONCURRENT_AUDITS", "2") or "2"))
GLOBAL_DAILY_CAP = max(1, int(os.environ.get("GLOBAL_DAILY_CAP", "150") or "150"))
_AUDIT_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_AUDITS)
_audit_inflight_lock = threading.Lock()
_audit_inflight = 0


def _audit_slot_acquire():
    """Non-blocking. True if a concurrency slot was secured (caller MUST release)."""
    global _audit_inflight
    if _AUDIT_SEMAPHORE.acquire(blocking=False):
        with _audit_inflight_lock:
            _audit_inflight += 1
        return True
    return False


def _audit_slot_release():
    """Free a concurrency slot. Safe to call once per successful acquire."""
    global _audit_inflight
    try:
        _AUDIT_SEMAPHORE.release()
    except ValueError:
        pass  # guard against a double-release
    with _audit_inflight_lock:
        _audit_inflight = max(0, _audit_inflight - 1)


def _global_audits_today(day):
    """Total free audits recorded across ALL IPs today (for the global ceiling).
    Fails open (returns 0) so a DB hiccup never blocks a legitimate audit."""
    try:
        n = db.session.query(db.func.coalesce(db.func.sum(FreeAuditUse.count), 0)) \
            .filter(FreeAuditUse.day == day).scalar()
        return int(n or 0)
    except Exception as _ge:
        try:
            db.session.rollback()
        except Exception:
            pass
        print("global-audit-count query failed (allowing):", str(_ge)[:120])
        return 0


_GEO_CACHE = {}


def _coarse_geo(ip):
    """Best-effort coarse geo ('City, Region, Country') from an IP via a free
    lookup. Cached, short timeout, returns '' on any failure — it runs AFTER the
    audit result is sent, so it never blocks the user or breaks lead capture."""
    if not ip or ip in ('127.0.0.1', '::1', 'localhost'):
        return ''
    if ip in _GEO_CACHE:
        return _GEO_CACHE[ip]
    geo = ''
    try:
        import urllib.request
        url = f"http://ip-api.com/json/{ip}?fields=status,country,regionName,city"
        with urllib.request.urlopen(url, timeout=3) as r:
            d = json.loads(r.read().decode('utf-8', 'ignore'))
            if d.get('status') == 'success':
                geo = ", ".join(x for x in (d.get('city'), d.get('regionName'), d.get('country')) if x)[:160]
    except Exception:
        geo = ''
    _GEO_CACHE[ip] = geo
    return geo

# Comma-separated list of client IPs exempt from the per-day cap. Set on Render
# (Environment tab) when you want unlimited audits from a specific IP — your own
# residential / office IP, a co-worker, a demo machine. Look up the IP from any
# previous request in Render Logs (clientIP="..."). Whitespace + empty entries
# are ignored, so e.g. " 1.2.3.4 , 5.6.7.8" parses cleanly.
_FREE_AUDIT_BYPASS_IPS = set(
    ip.strip() for ip in (os.environ.get("FREE_AUDIT_BYPASS_IPS", "") or "").split(",")
    if ip.strip()
)


# Number of trusted reverse-proxy hops in front of the app. Render terminates at
# one load-balancer hop; if you front it with a CDN (Cloudflare etc.) raise this.
_TRUSTED_PROXY_HOPS = max(1, int(os.environ.get("TRUSTED_PROXY_HOPS", "1") or "1"))


import ipaddress as _ipaddress
# Cloudflare's published edge ranges. signal.innatec3.com is proxied through
# Cloudflare, so the immediate upstream Render sees is a Cloudflare edge — and
# X-Forwarded-For therefore yields a Cloudflare IP, not the visitor's. The real
# end-user IP is in CF-Connecting-IP (Cloudflare sets it authoritatively and
# strips any client-supplied copy). But we only trust that header when the
# request's VERIFIED upstream is Cloudflare — otherwise a direct hit to the
# onrender.com origin could forge CF-Connecting-IP to dodge the per-IP cap.
_CLOUDFLARE_CIDRS = [
    '173.245.48.0/20', '103.21.244.0/22', '103.22.200.0/22', '103.31.4.0/22',
    '141.101.64.0/18', '108.162.192.0/18', '190.93.240.0/20', '188.114.96.0/20',
    '197.234.240.0/22', '198.41.128.0/17', '162.158.0.0/15', '104.16.0.0/13',
    '104.24.0.0/14', '172.64.0.0/13', '131.0.72.0/22',
    '2400:cb00::/32', '2606:4700::/32', '2803:f800::/32', '2405:b500::/32',
    '2405:8100::/32', '2a06:98c0::/29', '2c0f:f248::/32',
]
try:
    _CLOUDFLARE_NETS = [_ipaddress.ip_network(c) for c in _CLOUDFLARE_CIDRS]
except Exception:
    _CLOUDFLARE_NETS = []


def _ip_is_cloudflare(ip):
    """True if ip is a Cloudflare edge address (per their published ranges)."""
    try:
        a = _ipaddress.ip_address(ip)
        return any(a in n for n in _CLOUDFLARE_NETS)
    except Exception:
        return False


def _client_ip():
    """Best-effort REAL client IP, resistant to X-Forwarded-For SPOOFING and
    unwrapping Cloudflare.

    X-Forwarded-For is `client, hop1, …, hopN`, where each proxy APPENDS the
    address it received from — so the entries OUR trusted proxies added are on
    the RIGHT, and everything to their left is attacker-controllable. We take the
    entry our own proxy chain appended: the (_TRUSTED_PROXY_HOPS)-th from the
    right. Behind Cloudflare that entry is a Cloudflare EDGE IP, so when the
    verified upstream is Cloudflare we unwrap the true visitor from
    CF-Connecting-IP (which fixes geo, the per-IP cap, and link-open tracking all
    collapsing onto Cloudflare data-center addresses). We require the verified
    upstream to be Cloudflare before trusting that header so a direct origin hit
    can't spoof it.
    """
    fwd = request.headers.get('X-Forwarded-For', '') or ''
    parts = [p.strip() for p in fwd.split(',') if p.strip()]
    edge = parts[-min(_TRUSTED_PROXY_HOPS, len(parts))] if parts else (request.remote_addr or 'unknown')
    cf = (request.headers.get('CF-Connecting-IP', '') or '').strip()
    if cf and _ip_is_cloudflare(edge):
        return cf
    return edge


def _ip_is_exempt_from_cap(ip):
    """True if this IP should bypass FREE_DAILY_CAP enforcement (allowlisted
    via FREE_AUDIT_BYPASS_IPS env var)."""
    return bool(ip) and ip in _FREE_AUDIT_BYPASS_IPS


_IP_HOSTING_CACHE = {}


def _ip_is_hosting(ip):
    """True if ip is a data-center / hosting / proxy address rather than a
    residential or mobile end-user. Used to demote link 'opens' that are actually
    an automated fetch presenting a real browser UA — LinkedIn's own preview/
    safety infrastructure and corporate URL scanners fetch from data-center IPs;
    a genuine human read comes from a residential or mobile IP. Cached per-IP;
    best-effort; FAILS OPEN (returns False) on any error or timeout so a real
    open is never wrongly hidden. Requires _client_ip's CF-Connecting-IP unwrap
    to have run first, or every IP looks like Cloudflare's data center."""
    if not ip or ip in ('127.0.0.1', '::1', 'localhost', 'unknown'):
        return False
    if ip in _IP_HOSTING_CACHE:
        return _IP_HOSTING_CACHE[ip]
    hosting = False
    try:
        import urllib.request
        url = f"http://ip-api.com/json/{ip}?fields=status,hosting,proxy"
        with urllib.request.urlopen(url, timeout=2) as r:
            d = json.loads(r.read().decode('utf-8', 'ignore'))
            if d.get('status') == 'success':
                hosting = bool(d.get('hosting') or d.get('proxy'))
    except Exception:
        hosting = False
    _IP_HOSTING_CACHE[ip] = hosting
    return hosting


def _log_page_visit(kind, slug=None):
    """Best-effort first-party traffic log for the operator /traffic view. Skips
    bots/HEAD and operator-self IPs. NEVER raises — a logging failure must never
    break a page render. Referrer host is parsed for the referral roll-up;
    internal (innatec3.com) referrers are treated as direct."""
    try:
        if _is_link_preview_bot(request.headers.get('User-Agent', ''), request.method):
            return
        ip = _client_ip()
        if _ip_is_exempt_from_cap(ip):        # don't count operator self-traffic
            return
        ref = (request.referrer or '')[:400]
        ref_host = None
        if ref:
            try:
                from urllib.parse import urlparse
                h = (urlparse(ref).hostname or '').lower()
                if h.startswith('www.'):
                    h = h[4:]
                ref_host = None if (not h or 'innatec3.com' in h) else h[:160]
            except Exception:
                ref_host = None
        # New-unique-visitor check BEFORE inserting this row.
        seen_before = bool(kind == 'report' and slug and PageVisit.query.filter_by(
            kind='report', slug=slug, ip=ip).count())
        db.session.add(PageVisit(
            kind=kind, path=(request.path or '')[:200], slug=slug, ip=ip,
            referrer=ref or None, ref_host=ref_host,
            utm_source=(request.args.get('utm_source') or '')[:120] or None,
            utm_medium=(request.args.get('utm_medium') or '')[:120] or None,
            utm_campaign=(request.args.get('utm_campaign') or '')[:120] or None,
        ))
        db.session.commit()

        # Email on a NEW unique visitor to a report — unless they arrived via a
        # tracked /r/ link moments ago (that path already emails via
        # _send_click_email; without this dedup one click would ping twice).
        if kind == 'report' and slug and not seen_before:
            tokens = [t.token for t in TrackedLink.query.filter_by(slug=slug).all()]
            recent_click = bool(tokens and LinkClick.query.filter(
                LinkClick.token.in_(tokens), LinkClick.is_bot.is_(False),
                LinkClick.ip == ip,
                LinkClick.clicked_at >= datetime.utcnow() - timedelta(minutes=3)).count())
            # Same datacenter-IP filter the /r/ click path uses — LinkedIn/
            # corporate URL scanners fetch with browser UAs from hosting IPs
            # and would otherwise email as "new visitors". Checked here (not at
            # log time) so the ip-api lookup only runs on would-notify events.
            if not recent_click and not _ip_is_hosting(ip):
                from sqlalchemy import distinct as _distinct
                visitor_no = (db.session.query(db.func.count(_distinct(PageVisit.ip)))
                              .filter(PageVisit.kind == 'report', PageVisit.slug == slug)
                              .scalar() or 1)
                brand = None
                try:
                    rec = (InboundAudit.query.filter_by(slug=slug)
                           .filter(InboundAudit.brand.isnot(None))
                           .order_by(InboundAudit.created_at.desc()).first())
                    brand = rec.brand if rec else None
                except Exception:
                    pass
                report_url = request.url_root.rstrip('/') + url_for('view_signal_report', slug=slug)
                threading.Thread(
                    target=_send_dashboard_visitor_email,
                    args=(slug, brand, ip, request.headers.get('User-Agent', ''),
                          _fmt_et(datetime.utcnow()), visitor_no, report_url, ref_host),
                    daemon=True,
                ).start()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass


# Per-email hard cap on free audits. Each audit costs real money in LLM calls;
# uncapped self-serve becomes a billing risk on any viral share. Default 1 free
# audit per email; raise via EMAIL_AUDIT_CAP env var.
_EMAIL_AUDIT_CAP = max(1, int(os.environ.get("EMAIL_AUDIT_CAP", "1") or "1"))
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")


def _normalize_email(s):
    """Strip + lowercase + bounds-check. Returns None if obviously invalid."""
    s = (s or "").strip().lower()
    if not s or len(s) > 254 or not _EMAIL_RE.match(s):
        return None
    return s


def _is_operator_email(e):
    """True for biz-dev's own batch/operator runs. The batch runner posts from a
    per-run ops+<key>@innatec3.com address, so any ops+*@innatec3.com (or bare
    ops@innatec3.com) is us, not a real DIY lead — used to hide our runs from the
    default /inbound view."""
    e = (e or "").strip().lower()
    return e == "ops@innatec3.com" or (e.startswith("ops+") and e.endswith("@innatec3.com"))


def _session_inbound_attr():
    """Attribution captured on the landing GET (UTM + external referrer), read
    from the session so the worker thread — which has no request context — can
    persist it. Safe to call anywhere with a request/session available."""
    a = session.get("inbound_attr") or {}
    return {
        "utm_source": (a.get("utm_source") or "")[:120] or None,
        "utm_medium": (a.get("utm_medium") or "")[:120] or None,
        "utm_campaign": (a.get("utm_campaign") or "")[:120] or None,
        "referrer": (a.get("referrer") or "")[:400] or None,
    }


def _capture_inbound_lead(status, problem_statement, ip, email, is_operator, attr,
                          geo=None, slug=None, brand=None, category=None):
    """Best-effort insert of one InboundAudit lead row. Returns the new row id (or
    None on failure). NEVER raises — a capture failure must not affect the audit.
    Used both at audit START (status='started', later flipped to completed/errored)
    and at the rate-limit gates (status='rate_limited')."""
    try:
        with app.app_context():
            rec = InboundAudit(
                status=status, slug=slug, brand=brand, category=category,
                problem_statement=(problem_statement or "")[:2000],
                ip=ip, geo=geo or None, email=email or None,
                is_operator=bool(is_operator),
                utm_source=attr.get("utm_source"), utm_medium=attr.get("utm_medium"),
                utm_campaign=attr.get("utm_campaign"), referrer=attr.get("referrer"),
            )
            db.session.add(rec)
            db.session.commit()
            return rec.id
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        print("inbound-lead capture failed (continuing):", str(e)[:120])
        return None


@app.route('/healthz')
def healthz():
    """Lightweight liveness probe — no DB, no LLM. Surfaces audit concurrency so
    Render alerts and manual checks can watch the launch spike."""
    return jsonify({
        "status": "ok",
        "audits_inflight": _audit_inflight,
        "max_concurrent": MAX_CONCURRENT_AUDITS,
        "slots_free": max(0, MAX_CONCURRENT_AUDITS - _audit_inflight),
    }), 200


@app.route('/admin/test-email')
def admin_test_email():
    """Operator: attempt a real alert-email send SYNCHRONOUSLY and report the
    outcome. The alert helpers (_send_click_email etc.) are fire-and-forget
    threads that swallow errors into logs, so 'no email ever arrived' is
    undiagnosable from outside without this. Returns whether the key is set,
    the SendGrid HTTP status, or the exact exception."""
    if not _operator_ok():
        abort(404)
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    info = {
        "resend_key_set": bool(os.environ.get("RESEND_API_KEY")),
        "sendgrid_key_set": bool(os.environ.get("SENDGRID_API_KEY")),
        "provider": ("resend" if os.environ.get("RESEND_API_KEY")
                     else "sendgrid" if os.environ.get("SENDGRID_API_KEY") else None),
        "to": to,
    }
    if not info["provider"]:
        info["result"] = ("NO email provider configured — set RESEND_API_KEY (preferred) "
                          "or SENDGRID_API_KEY on Render; until then every alert email "
                          "(first-open, new-visitor, audit-error, lead reports) is "
                          "silently skipped")
        return jsonify(info)
    try:
        from sendgrid.helpers.mail import Mail
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to],
            subject="✅ Signal Finder alert-email test",
            plain_text_content="Test send from /admin/test-email — if you're reading "
                               "this, alert emails are working end to end.",
        )
        resp = _send_mail_object(msg)
        info["result"] = f"accepted by {info['provider']} (HTTP {resp.status_code})"
    except Exception as e:
        info["result"] = f"SEND FAILED — {type(e).__name__}: {str(e)[:400]}"
    return jsonify(info)


@app.route('/whoami')
def whoami_diagnostic():
    """Operator diagnostic: how does THIS request's IP resolve + classify? Used to
    verify the Cloudflare unwrap and the hosting/bot detection from real devices —
    hit it from a phone/desktop and compare. Operator-gated."""
    if not _operator_ok():
        abort(404)
    xff = request.headers.get('X-Forwarded-For', '') or ''
    parts = [p.strip() for p in xff.split(',') if p.strip()]
    edge = parts[-min(_TRUSTED_PROXY_HOPS, len(parts))] if parts else (request.remote_addr or 'unknown')
    ua = request.headers.get('User-Agent', '') or ''
    ip = _client_ip()
    return jsonify({
        "x_forwarded_for": xff,
        "cf_connecting_ip": request.headers.get('CF-Connecting-IP', ''),
        "trusted_edge": edge,
        "edge_is_cloudflare": _ip_is_cloudflare(edge),
        "resolved_client_ip": ip,
        "ip_is_hosting": _ip_is_hosting(ip),
        "ip_is_exempt_from_cap": _ip_is_exempt_from_cap(ip),
        "ua_is_bot": _is_link_preview_bot(ua, request.method),
        "would_count_open_as_human": not (_is_link_preview_bot(ua, request.method) or _ip_is_hosting(ip)),
        "geo": _coarse_geo(ip),
        "user_agent": ua,
    }), 200


@app.route('/citation-audit', methods=['GET', 'POST'])
def citation_audit():
    # MVP branch: audits are always anonymous + free tier. We still pass
    # signal_user / signal_credits through to the template (the JS globals
    # reference them) but they're always falsy.
    user = current_signal_user()
    credits = current_user_credits(user)

    if request.method == 'GET':
        # Stash attribution for the eventual audit POST (operator inbound view).
        # The POST is a same-origin fetch, so its Referer is our own page; the
        # only place to see the external referrer (e.g. linkedin.com) + the UTM
        # query is this landing GET. Persist in the session; read it on POST.
        try:
            _utm = {k: (request.args.get(k) or '')[:120]
                    for k in ('utm_source', 'utm_medium', 'utm_campaign')}
            _ext_ref = (request.referrer or '')[:400]
            if any(_utm.values()) or _ext_ref:
                session['inbound_attr'] = {
                    'utm_source': _utm['utm_source'],
                    'utm_medium': _utm['utm_medium'],
                    'utm_campaign': _utm['utm_campaign'],
                    'referrer': _ext_ref,
                }
        except Exception:
            pass
        _log_page_visit('home')
        return render_template(
            'citation_audit.html',
            ga_measurement_id=GA_MEASUREMENT_ID,
            signal_user=user,
            signal_credits=credits,
        )

    problem_statement = request.form.get('problem_statement', '').strip()
    if not problem_statement:
        return jsonify({"error": "Please describe the problem you want your brand to own."}), 400

    # MVP branch: paid tier disabled — see feat/pr-signal-finder for full paid flow.
    # Force-coerce any incoming tier to "free".
    tier = 'free'
    credit_charged = False

    # Accept optional user-curated prompts from the review/revise step. The
    # frontend posts a JSON-encoded list of 10 strings in the 'prompts' form
    # field. Defensive parse: must be list of 5-20 non-empty strings; bad
    # input falls back to auto-generation so a buggy client never breaks the
    # audit, only loses the editor benefit.
    prompts_override = None
    raw_prompts = (request.form.get('prompts') or '').strip()
    if raw_prompts:
        try:
            parsed = json.loads(raw_prompts)
            if isinstance(parsed, list):
                cleaned = [p.strip() for p in parsed if isinstance(p, str) and p.strip()]
                if 5 <= len(cleaned) <= 20:
                    # Trim per-prompt length to a sane ceiling so a malicious
                    # client can't blow Claude's input budget.
                    prompts_override = [p[:500] for p in cleaned]
        except Exception:
            prompts_override = None

    # Cost control on free audits (email is OPTIONAL — not required to run):
    #   1. Per-IP per-day cap (primary): FREE_DAILY_CAP audits per IP per day —
    #      the main control against runaway LLM spend now that email isn't gated.
    #   2. Per-email lifetime cap (only when an email is voluntarily provided):
    #      EMAIL_AUDIT_CAP (default 1), and we capture the lead.
    # IPs in FREE_AUDIT_BYPASS_IPS skip BOTH caps — operator self-testing +
    # close-partner walkthroughs don't pollute the lead table.
    today = date.today()
    ip = _client_ip()
    lead_email = None
    if _ip_is_exempt_from_cap(ip):
        print(f"[audit] FREE_AUDIT_BYPASS_IPS hit: {ip} (unlimited)")
    else:
        # Email is optional. If one is provided (e.g. a future opt-in field or
        # the demo flow), capture it as a lead and honor the per-email lifetime
        # cap. With no email, the per-IP/day cap below is the sole cost control.
        lead_email = _normalize_email(request.form.get('email', ''))
        if lead_email:
            lead = AuditLead.query.filter_by(email=lead_email).first()
            if lead and lead.audit_count >= _EMAIL_AUDIT_CAP:
                # High-intent lead: used their free audit and came back for more.
                # Capture the blocked attempt (status='rate_limited') so biz-dev can
                # follow up with a bespoke offer. Best-effort; never blocks the 429.
                # (No pending DB write at this point, so the capture's commit is clean.)
                _capture_inbound_lead(
                    'rate_limited', problem_statement, ip, lead_email,
                    (_is_operator_email(lead_email) or _operator_ok()),
                    _session_inbound_attr())
                return jsonify({
                    "error": ("You've already used your free audit. For more, talk to us about a bespoke "
                              "audit — we'll cover your category in depth and walk through the findings live."),
                    "code": "email_rate_limited",
                }), 429
            if lead:
                lead.audit_count += 1
                lead.last_seen = datetime.utcnow()
                lead.last_ip = ip
                lead.last_problem_statement = problem_statement[:1000]
            else:
                lead = AuditLead(
                    email=lead_email, audit_count=1,
                    first_seen=datetime.utcnow(), last_seen=datetime.utcnow(),
                    last_ip=ip, last_problem_statement=problem_statement[:1000],
                )
                db.session.add(lead)
        # Per-IP per-day abuse cap — the primary cost control now email is optional.
        # CHECK only here; the per-IP increment is deferred until a concurrency slot
        # is secured below, so at-capacity retries don't burn the daily quota.
        use = FreeAuditUse.query.filter_by(ip=ip, day=today).first()
        if use and use.count >= FREE_DAILY_CAP:
            return jsonify({
                "error": f"You've used your {FREE_DAILY_CAP} free audits for today. Talk to us about a bespoke audit for unlimited access.",
                "code": "rate_limited",
            }), 429
        # Global daily ceiling across ALL IPs — cost / rate-limit circuit breaker.
        if _global_audits_today(today) >= GLOBAL_DAILY_CAP:
            return jsonify({
                "error": "The free audit has hit its daily limit — lots of interest today! "
                         "Talk to us for a full bespoke audit, or check back tomorrow.",
                "code": "global_limit",
            }), 429
        db.session.commit()  # persist any lead upsert from above (increment deferred)
        print(f"[audit] {'lead capture: ' + lead_email if lead_email else 'anonymous audit'} ip={ip}")

    user_id = user.id if user else None
    cfg = TIER_CONFIG[tier]

    # ── Concurrency guard ── cap simultaneous audits so a traffic spike can't OOM
    # the single 2GB instance. Excess requests get a friendly "in line" 503; the
    # template auto-retries, so the user sees a queue state, never a 500 or a hang.
    if not _audit_slot_acquire():
        return jsonify({
            "code": "at_capacity",
            "error": "We're at capacity right now — a couple of audits are already running. "
                     "You're in line; it usually clears within a few minutes and will start automatically.",
            "retry_after": 45,
        }), 503
    # Slot secured — record per-IP usage now (deferred from the cap check above so
    # at-capacity retries don't burn the daily quota). Exempt IPs are never metered.
    if not _ip_is_exempt_from_cap(ip):
        try:
            _use = FreeAuditUse.query.filter_by(ip=ip, day=today).first()
            if _use:
                _use.count += 1
            else:
                db.session.add(FreeAuditUse(ip=ip, day=today, count=1))
            db.session.commit()
        except Exception as _ue:
            try:
                db.session.rollback()
            except Exception:
                pass
            print("usage increment failed (continuing):", str(_ue)[:120])

    # Attribution for the inbound-lead pipeline. Captured HERE (request context) so
    # the worker thread — which has no request/session — can persist it after save.
    _inbound_attr = _session_inbound_attr()
    # Flag our own runs (operator key, exempt IP, or an ops+*@innatec3.com email)
    # so the default /inbound view shows real DIY demand, not biz-dev batch runs.
    _is_operator = bool(_operator_ok() or _ip_is_exempt_from_cap(ip) or _is_operator_email(lead_email))

    q = queue.Queue()

    def on_progress(step, detail, current, total):
        q.put(json.dumps({"type": "progress", "step": step, "detail": detail, "current": current, "total": total}))

    def worker():
        start_dt = datetime.utcnow()
        # Record the lead the MOMENT the audit starts (status='started'), so an
        # abandon / server crash / OOM before the report renders still leaves a
        # lead row. Flipped to 'completed' (with slug) on success, or 'errored'
        # in the except below. Best-effort — never blocks or fails the audit.
        _inbound_id = _capture_inbound_lead(
            'started', problem_statement, ip, lead_email, _is_operator,
            _inbound_attr, geo=_coarse_geo(ip))
        try:
            result = run_citation_audit(
                problem_statement,
                on_progress=on_progress,
                tier=tier,
                prompts_override=prompts_override,
            )
            slug = uuid.uuid4().hex[:10]
            # Strip the debug-only keys (leading underscore) before the
            # payload is persisted to SharedResult / sent to the client.
            per_provider_errors = result.pop("_per_provider_errors", {}) or {}
            per_provider_done = result.pop("_per_provider_done", {}) or {}
            # Set slug + tier BEFORE serializing to SharedResult so the saved
            # payload carries them. (Previously these were assigned after
            # json.dumps — causing every saved payload to have slug=None and
            # tier=None when re-loaded for a shared report.)
            result["slug"] = slug
            result["tier"] = tier
            with app.app_context():
                shared = SharedResult(slug=slug, payload=json.dumps(result))
                db.session.add(shared)
                db.session.add(AuditRun(
                    user_id=user_id,
                    slug=slug,
                    tier=tier,
                    prompt_count=cfg["prompt_count"],
                    llm_count=len(cfg["llms"]),
                    credits_consumed=(1 if credit_charged else 0),
                    problem_statement=problem_statement,
                ))
                # Stamp the slug back on the lead row so we can match emails to
                # the reports they ran (lead follow-up). Failure here is fine —
                # the audit itself already succeeded.
                if lead_email:
                    try:
                        _lead = AuditLead.query.filter_by(email=lead_email).first()
                        if _lead:
                            _lead.last_slug = slug
                    except Exception as _le:
                        print("lead slug stamp failed (continuing):", str(_le)[:120])
                db.session.commit()
            q.put(json.dumps({"type": "result", "data": result}))
            # Flip the start-row to completed + backfill slug/brand/category IN
            # PLACE (no duplicate row). If the start-row failed to insert earlier,
            # fall back to a fresh completed row so a real completion is never lost.
            try:
                with app.app_context():
                    _rec = db.session.get(InboundAudit, _inbound_id) if _inbound_id else None
                    if _rec is not None:
                        _rec.status = 'completed'
                        _rec.slug = slug
                        _rec.brand = result.get('brand')
                        _rec.category = result.get('category')
                        db.session.commit()
                    else:
                        _capture_inbound_lead(
                            'completed', problem_statement, ip, lead_email, _is_operator,
                            _inbound_attr, geo=_coarse_geo(ip), slug=slug,
                            brand=result.get('brand'), category=result.get('category'))
            except Exception as _ib_e:
                print("inbound-audit completion update failed (continuing):", str(_ib_e)[:120])
            # Fire the owner debug email AFTER the result is pushed to the
            # queue so the user-visible response never waits on email send.
            duration_seconds = (datetime.utcnow() - start_dt).total_seconds()
            try:
                _send_audit_debug_email(
                    result,
                    duration_seconds,
                    per_provider_done,
                    per_provider_errors,
                    tier,
                    problem_statement=problem_statement,
                )
            except Exception as email_err:
                print("debug-email dispatch error:", email_err)
        except Exception as e:
            # Mark the start-row errored so the incomplete lead stays visible.
            try:
                if _inbound_id:
                    with app.app_context():
                        _erec = db.session.get(InboundAudit, _inbound_id)
                        if _erec is not None and _erec.status != 'completed':
                            _erec.status = 'errored'
                            db.session.commit()
            except Exception:
                pass
            # Diagnostic email to owner BEFORE the refund, so even if the
            # refund somehow blows up the owner still hears about the failure.
            try:
                _send_audit_failure_email(e, problem_statement, tier)
            except Exception as email_err:
                print("Failure-email dispatch error:", email_err)
            if credit_charged and user_id is not None:
                try:
                    with app.app_context():
                        bal_inner = CreditBalance.query.filter_by(user_id=user_id).first()
                        if bal_inner:
                            bal_inner.credits_remaining = (bal_inner.credits_remaining or 0) + 1
                            db.session.commit()
                except Exception as refund_err:
                    print("Credit refund failed:", refund_err)
            # User-facing message stays generic; technical details go to the
            # owner via _send_audit_failure_email above. Server log still has
            # the full exception for grep.
            print(f"run_citation_audit failed: {type(e).__name__}: {e}")
            if credit_charged:
                user_message = "Analysis step failed unexpectedly. Your credit has been refunded. Nathan has been notified to investigate."
            else:
                user_message = "Analysis step failed unexpectedly. Nathan has been notified to investigate — please try again in a moment."
            q.put(json.dumps({"type": "error", "error": user_message}))
        finally:
            _audit_slot_release()

    t = threading.Thread(target=worker)
    t.start()

    def generate():
        # Heartbeat: the analysis/verification phases (page-evidence, topic-verify,
        # final report LLM call) can run >100s without emitting progress. Cloudflare
        # kills a streaming connection after ~100s of silence (524), so the browser
        # times out even though the worker finishes and saves the report. Emit an SSE
        # keepalive comment on idle so the connection never goes silent that long.
        while True:
            try:
                msg = q.get(timeout=20)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield f"data: {msg}\n\n"
            parsed = json.loads(msg)
            if parsed["type"] in ("result", "error"):
                break

    return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/signal/audit/prompts', methods=['POST'])
def signal_generate_prompts():
    """Generate the 10 prompts that the audit would run, without actually
    running the LLM batch. The frontend uses this to offer a "review &
    revise prompts" step before the user commits to the (slow + rate-limited)
    full audit.

    MVP / free-tier behavior: anonymous, but ABUSE-RATE-LIMITED by the same
    daily IP cap that governs full audits — generating prompts counts the
    same as running an audit against FREE_DAILY_CAP. This prevents bots from
    burning Claude tokens by hammering the endpoint without ever running an
    actual audit. IPs on FREE_AUDIT_BYPASS_IPS skip the cap.
    """
    body = request.get_json(silent=True) or {}
    problem_statement = (body.get('problem_statement') or '').strip()
    if not problem_statement:
        return jsonify({"error": "Please describe the problem you want your brand to own."}), 400

    # MVP: tier always free. Paid-tier code path preserved on feat/pr-signal-finder.
    tier = 'free'

    # Same daily IP cap as /citation-audit so bots can't burn tokens by
    # spamming this endpoint without ever running an audit. We do NOT
    # increment here — only the actual audit increments. We just check.
    ip = _client_ip()
    if not _ip_is_exempt_from_cap(ip):
        today = date.today()
        use = FreeAuditUse.query.filter_by(ip=ip, day=today).first()
        if use and use.count >= FREE_DAILY_CAP:
            return jsonify({
                "error": f"You've used your {FREE_DAILY_CAP} free audits for today. Talk to us about a bespoke audit for unlimited access.",
                "code": "rate_limited",
            }), 429

    try:
        gen = _generate_audit_prompts(problem_statement, tier=tier)
    except Exception as e:
        print("Prompt generation failed:", e)
        return jsonify({"error": "Could not generate prompts. Try again."}), 500

    return jsonify({
        "brand": gen["brand"],
        "category": gen["category"],
        "prompts": gen["prompts"],
    })


def _load_signal_report(slug):
    """Return parsed PR Signal Finder report dict for a slug, or None."""
    rec = SharedResult.query.filter_by(slug=slug).first()
    if not rec:
        return None
    try:
        data = json.loads(rec.payload)
    except Exception:
        return None
    if not isinstance(data, dict) or "media_targets" not in data:
        return None
    data["slug"] = slug
    return data


def _regenerate_executive_summary(data):
    """Re-run ONLY the executive-summary Claude call against the re-analyzed
    cached data. ONE Claude call (~$0.01, ~10s) — lets the operator refresh
    the 'What we found' summary on an existing audit with the CURRENT
    summary-prompt wording, without a fresh 50-LLM batch.

    Returns the new summary string, or the existing one on any failure.
    """
    try:
        brand = data.get('brand') or 'the brand'
        category = data.get('category') or 'this category'
        total = data.get('total_responses') or 0
        brand_mentions = data.get('brand_mention_count') or 0
        mindshare = round(brand_mentions / total * 100) if total else 0
        competitors = data.get('competitors') or []
        related = data.get('related_brands') or []

        # If a name inside the brand string surfaced as an "entity" (its parent/
        # acquirer/alias, e.g. Zendesk for 'Ultimate (Zendesk Ultimate)'), tell the
        # model NOT to frame it as a competitor the brand is losing to, and to
        # reframe a ~0% result as brand architecture rather than a deficit.
        related_note = ""
        if related:
            rnames = ", ".join(c.get('name') for c in related if c.get('name'))
            related_note = (
                f"\nCRITICAL CONTEXT: {rnames} appears INSIDE the brand's own name — it is "
                f"{brand}'s parent / acquirer / alias, NOT a competitor. NEVER say {brand} "
                f"'trails', 'is losing to', or has a 'deficit' vs {rnames}. If {brand} "
                f"registers little on its own, explain its AI presence is likely captured "
                f"UNDER {rnames} (which leads the category), and frame the takeaway as a brand-"
                f"architecture choice — build a distinct identity vs operate under {rnames} — "
                f"not a head-to-head loss. Sentence 1 must reflect this, not a deficit number.\n")

        def _pct(n):
            return round((n or 0) / total * 100) if total else 0

        comp_block = "\n".join(
            f"  - {c.get('name')}: cited in {c.get('mention_count')} of {total} responses "
            f"({_pct(c.get('mention_count'))}%)"
            for c in competitors[:8]
        ) or "  (none surfaced)"

        # Precompute the position math — models reliably garble count-vs-percent
        # arithmetic (observed: "trails by 5" when the real gap was 32 points),
        # so sentence 1's facts are handed over verbatim, not derived.
        position_facts = "(no competitor data)"
        if competitors:
            top = max(competitors, key=lambda c: c.get('mention_count') or 0)
            tn, tp = top.get('mention_count') or 0, _pct(top.get('mention_count'))
            delta = mindshare - tp
            rel = "LEADS" if delta > 0 else ("TRAILS" if delta < 0 else "TIES")
            position_facts = (
                f"{brand}: {mindshare}% ({brand_mentions} of {total} responses). "
                f"Top competitor {top.get('name')}: {tp}% ({tn} of {total}). "
                f"{brand} {rel} {top.get('name')} by {abs(delta)} percentage points."
            )

        sov = data.get('outlet_sov') or []

        def _sov_line(r):
            tc = r.get('top_competitor_at_outlet')
            tcs = (f"; top competitor here {tc['name']} at {int((tc.get('sov_at_outlet') or 0)*100)}%"
                   if tc else "")
            n = r.get('responses_citing') or 0
            return (f"  - {r.get('domain')}: {brand} in {r.get('brand_mentions_at_outlet')}/{n} "
                    f"responses citing it ({int((r.get('brand_sov_at_outlet') or 0)*100)}%){tcs}")

        strengths = [r for r in sov if r.get('verdict') == 'strength']
        opps = [r for r in sov if r.get('verdict') == 'opportunity']
        strengths_block = "\n".join(_sov_line(r) for r in strengths[:6]) or "  (none)"
        opps_block = "\n".join(_sov_line(r) for r in opps[:6]) or "  (none)"

        raw = data.get('raw_citation_domains') or []
        raw_block = "\n".join(f"  - {d.get('domain')}: {d.get('count')}x" for d in raw[:12]) or "  (none)"

        per_llm = data.get('per_llm_visibility') or []
        per_llm_block = "\n".join(
            f"  - {x.get('llm')}: {x.get('mentions')}/{x.get('total')} responses"
            f"{' (search-grounded)' if x.get('grounded') else ''}"
            for x in per_llm
        ) or "  (not available)"
        per_llm_read = data.get('per_llm_read') or ""
        hm = data.get('headline_move') or {}
        hm_block = (f"{hm.get('outlet') or ''} — {hm.get('text') or ''}".strip(" —")
                    or "(none — build the move around the strongest opportunity above)")

        prompt = f"""You are an AI citation intelligence analyst writing the headline finding of a brand's AI Mindshare Briefing. The reader is a comms director who will paste your 3 sentences into a CMO briefing.

Brand: {brand}
Category: {category}
{brand}'s AI mindshare: {mindshare}% (cited in {brand_mentions} of {total} AI responses)
{related_note}
POSITION FACTS (pre-computed — use these EXACT numbers in Sentence 1; do NOT derive your own deltas or percentages):
{position_facts}

PER-ASSISTANT VISIBILITY (how many of EACH AI's responses mention {brand}):
{per_llm_block}
Read: {per_llm_read}

TOP COMPETITORS (by mentions across all responses):
{comp_block}

STRENGTHS — outlets where {brand} over-indexes (when the AI cites this outlet, it names {brand} more than its overall rate — defend / extend):
{strengths_block}

OPPORTUNITIES — outlets where a competitor out-cites {brand} (when the AI cites this outlet, it names the competitor, not {brand} — pitch whitespace):
{opps_block}

MOST-CITED SOURCE DOMAINS (all types — note if analyst firms like Gartner/Forrester/McKinsey/IDC dominate over editorial press):
{raw_block}

THE #1 MOVE (computed from this data and shown as its OWN separate card on the page — your summary must stay directionally CONSISTENT with it but must NOT duplicate it: do not reuse its outlet, its action verb, or its tactical steps. The summary DIAGNOSES the position; the card PRESCRIBES the action):
{hm_block}

Write EXACTLY 3 sentences. Lead with the single most important insight, no preamble.
  Sentence 1 — THE POSITION: {brand}'s mindshare framed against the top competitor's count. If {brand} >= top competitor, lead with strength; if behind, lead with the gap. NEVER say 'lacks authority' if {brand} out-mentions competitors.
  Sentence 2 — THE SURPRISE: the single most non-obvious thing in the data. STRONGLY PREFER the per-assistant concentration when it's lopsided — e.g. "{brand}'s visibility is almost entirely one assistant (Gemini 9/10) and absent from ChatGPT, Claude, and Grok". Otherwise surface the sharpest share-of-voice gap — name a specific OUTLET and the competitor out-citing {brand} there (but NOT the #1-move outlet — save that for Sentence 3).
  Sentence 3 — THE STAKE: the strategic implication of the position — what it MEANS or what is at risk — NOT a tactic (the tactic is shown separately as the #1 move card, so DON'T restate it). Good Sentence 3s: the gap is structural (e.g. evenly thin across all assistants) rather than platform-specific; the brand is discoverable but not yet the default answer; a category/competitive dynamic that defines the challenge or the window to act. Stay consistent with the brand's real standing — if it LEADS overall, a position to protect; if it TRAILS, a gap that won't close on its own (never imply 'top-tier'/'maintain' for a trailing brand) — but DIAGNOSE, don't prescribe: do NOT name the #1 move's outlet and do NOT reuse its action verbs (pitch/defend/build/hold/widen/cultivate).
FRAMING RULE (critical): these are AI-citation SHARE-OF-VOICE signals — how often each AI names {brand} vs competitors when it cites an outlet — NOT press clips. Never say an outlet "covers", "features", or "wrote about" {brand}; say {brand} "over-indexes at", "is out-cited at", or "is absent from" the outlet.
BANNED: filler like 'this audit reveals', 'in today's landscape'. Discuss analysts only if they actually dominate the source domains above. Respond with ONLY the 3 sentences — no preamble, no JSON, no labels."""

        resp = anthropic.messages.create(
            model=CLAUDE_SONNET,
            max_tokens=400,
            timeout=60.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.content[0].text or "").strip()
        return text or data.get('executive_summary')
    except Exception as e:
        print(f"[rerender] exec-summary regen failed (keeping original): {e}")
        return data.get('executive_summary')


def _rerender_from_cached_responses(data, regenerate_summary=False):
    """Re-run the post-LLM analysis pipeline on a saved audit's cached
    all_responses, applying the CURRENT codebase logic for editorial
    classification, brand/competitor-domain filtering, and share-of-voice
    computation. Returns a NEW dict — does not mutate or persist.

    Use case: iterate on filter / SoV / UI changes without re-running the
    full 50-LLM batch (which is the slow + expensive part of an audit).
    Costs: zero API calls for the pure-Python pass. If regenerate_summary
    is True, ONE additional Claude call refreshes the 'What we found'
    executive summary with the current prompt wording (~$0.01, ~10s).

    Limitation: cannot regenerate per-target rationale or gap_insight
    (those come from the full Claude analysis call). So the rerender can
    only PRUNE saved media_targets — drop any whose domain is no longer
    classified as editorial — not ADD newly-surfaced editorial domains
    that didn't make the original cut. For changes to the per-target
    rationale wording, run a fresh audit.
    """
    all_responses = data.get('all_responses') or []
    brand = data.get('brand') or ''
    if not all_responses or not brand:
        return data

    # Drop errored responses (same rule as the fresh-audit path) so a rerender
    # repairs reports that were saved with a failed provider's empty responses
    # — recomputing mindshare/per-LLM/SoV over what actually came back. Lets us
    # fix a credit-exhausted-Grok report without re-running the 50-call batch.
    _errored = [r for r in all_responses if r.get('error')]
    if _errored:
        all_responses = [r for r in all_responses if not r.get('error')]
        print(f"[rerender] dropped {len(_errored)} errored responses; metrics "
              f"now over {len(all_responses)} delivered responses")

    out = dict(data)
    out['all_responses'] = all_responses  # persist the cleaned set
    ranked_domains = aggregate_citations(all_responses)
    ranked_domains = _augment_citations_with_named_outlets(ranked_domains, all_responses)

    editorial_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'editorial']
    editorial_domains = [d for d in editorial_domains if not _is_brand_own_domain(d['domain'], brand)]

    # Sub-brand aliases (e.g. iShares for BlackRock) — resolve once, reuse if stored.
    if 'brand_aliases' in data:
        brand_aliases = data.get('brand_aliases') or []
    else:
        brand_aliases = _resolve_brand_aliases(brand)
    # Re-verify even STORED aliases on every render: an older payload can carry
    # hallucinated (BMS's "Cellgene") or generic (GE's "GE"/"Innova"/"Discovery")
    # aliases saved before this guard existed, and reusing them blindly here would
    # keep re-inflating the count on every future refresh.
    brand_aliases = _verify_brand_aliases(brand_aliases, all_responses)
    out['brand_aliases'] = brand_aliases
    # Recount the brand's own mention total (alias-aware, so iShares counts toward
    # BlackRock) so the headline reflects current guardrails + the brand's sub-brands.
    if all_responses and brand:
        _bforms = _brand_match_forms(brand, brand_aliases)
        new_brand_count = sum(1 for r in all_responses if _brand_present_in_text(_bforms, _response_text(r)))
        if new_brand_count != (data.get('brand_mention_count') or 0):
            print(f"[rerender] brand_mention_count (alias-aware) recount: "
                  f"{data.get('brand_mention_count')} -> {new_brand_count}")
        out['brand_mention_count'] = new_brand_count
    # Sync the denominator to the cleaned response set so mindshare % is over
    # what was actually delivered (a dropped-error rerender shrinks this from
    # e.g. 50 -> 40, correcting ServiceNow's "34%" to ~43% over 4 live LLMs).
    if len(all_responses) != (data.get('total_responses') or 0):
        print(f"[rerender] total_responses: {data.get('total_responses')} -> {len(all_responses)}")
    out['total_responses'] = len(all_responses)
    out['responses_completed'] = len(all_responses)

    competitor_counts = data.get('competitors') or []
    # Recount competitor mention totals against the cached responses so the
    # current code's guardrails (common-word, first-word) apply. Names come
    # from the cached pick; counts are recomputed. Drops anything that
    # recounts to 0 (e.g. a name that was only counted via a fixed bug).
    if competitor_counts and all_responses:
        before = {c.get('name'): c.get('mention_count') for c in competitor_counts}
        rebuilt = []
        for c in competitor_counts:
            name = (c.get('name') or '').strip()
            if not name:
                continue
            cnt = _count_brand_mentions(name, all_responses)
            if cnt > 0:
                cc = dict(c)
                cc['mention_count'] = cnt
                rebuilt.append(cc)
        rebuilt.sort(key=lambda c: c['mention_count'], reverse=True)
        deltas = [(c['name'], before.get(c['name'], 0), c['mention_count'])
                  for c in rebuilt if before.get(c['name'], 0) != c['mention_count']]
        if deltas:
            print(f"[rerender] competitor recount deltas: " +
                  ", ".join(f"{n}: {a}->{b}" for n, a, b in deltas[:10]))
        competitor_counts = rebuilt
        out['competitors'] = competitor_counts
    # Merge same-entity duplicates (e.g. 'JPMorgan' + 'J.P. Morgan Asset Management')
    # so existing audits get the dedupe on refresh too.
    competitor_counts = _dedupe_competitors(competitor_counts, all_responses)
    out['competitors'] = competitor_counts
    # Re-classify competitor types so existing audits (saved before the
    # classifier existed) also get the brand_peer / retailer split applied.
    # Skip if every competitor is already classified — saves a Claude call.
    if competitor_counts and not all(c.get('type') for c in competitor_counts):
        competitor_counts = _classify_competitor_types(
            brand, data.get('category'), competitor_counts)
        out['competitors'] = competitor_counts
    # Pull the brand's own alias/parent out of the competitive set (e.g. 'Zendesk'
    # when the brand is 'Ultimate (Zendesk Ultimate)') so the report never frames
    # the brand as losing to its own parent. Done AFTER the recount so the parent's
    # mention total is known (passed to the summary as context, not as a rival).
    competitor_counts, _related = _split_self_referential_competitors(brand, competitor_counts)
    competitor_counts = _merge_competitor_subbrands(competitor_counts, all_responses)  # SPDR -> State Street
    competitor_counts = _drop_supplier_noncompetitors(brand, data.get('category'), competitor_counts)
    out['competitors'] = competitor_counts
    out['related_brands'] = _related
    if _related:
        print(f"[rerender] moved {len(_related)} self/parent name(s) out of competitors: "
              f"{', '.join(c.get('name') for c in _related)}")
    # Owned Signal Finder lens — same data, owned-media focus.
    try:
        if out.get('brand_domains'):
            brand_domain_hints = out['brand_domains']
        else:
            brand_domain_hints = list(_resolve_brand_domains(brand, data.get('category')))
            for _al in (brand_aliases or []):   # sub-brand owned sites (e.g. ishares.com)
                for _dd in _resolve_brand_domains(_al, data.get('category')):
                    if _dd not in brand_domain_hints:
                        brand_domain_hints.append(_dd)
        # Re-verify on EVERY render (even the reused-from-storage branch) so a
        # stale hallucinated domain saved before this guard existed is dropped.
        brand_domain_hints = _verify_brand_domains(brand, brand_aliases, brand_domain_hints, all_responses)
        out['brand_domains'] = brand_domain_hints
        out['owned'] = _compute_owned_analysis(brand, competitor_counts, _related, ranked_domains,
                                               all_responses, brand_domain_hints=brand_domain_hints,
                                               brand_aliases=brand_aliases)
        if regenerate_summary:
            _enrich_owned_recommendations(brand, out.get('category') or data.get('category'), out['owned'])
    except Exception as _oe:
        print(f"[rerender] owned analysis failed (continuing): {_oe}")
    competitor_stems = _competitor_domain_stems(competitor_counts)
    if competitor_stems:
        editorial_domains = [d for d in editorial_domains if not _is_competitor_owned_domain(d['domain'], competitor_stems)]

    # Relevance filter: drop off-audience trade/industrial/professional outlets.
    # editorial_set (built below from editorial_domains) prunes them from
    # media_targets too, so SoV + cards stay consistent.
    editorial_domains = _filter_relevant_editorial(editorial_domains, data.get('category'), log_prefix="[rerender] ")

    out['raw_citation_domains'] = ranked_domains
    # Per-assistant visibility — pure-python over cached responses, so the
    # ?fresh/?refresh rerender surfaces it on existing audits too.
    try:
        out['per_llm_visibility'] = _compute_per_llm_visibility(brand, all_responses, aliases=brand_aliases)
        out['per_llm_read'] = _llm_visibility_read(brand, out['per_llm_visibility'])
    except Exception:
        out['per_llm_visibility'] = []
        out['per_llm_read'] = None
    # Media Landscape (watchlist × AI-presence status). One Claude call; only on
    # ?refresh (the summary-regen path) to keep ?fresh a zero-API pure rerender.
    if regenerate_summary:
        try:
            out['media_landscape'] = _compute_media_landscape(
                brand, data.get('category'), all_responses, ranked_domains)
        except Exception:
            out['media_landscape'] = out.get('media_landscape') or {}

    # Rebuild media_targets from the CURRENT editorial ranking so the refresh
    # view can both PRUNE (drop domains no longer classified editorial) AND ADD
    # outlets newly surfaced by name-aware detection (WWD, Forbes, TechCrunch…)
    # that the original Claude pass never saw because their URLs had been pruned.
    # Saved targets keep their bespoke Claude rationale; newly-surfaced outlets
    # get a deterministic verdict-based rationale (a fresh audit regenerates the
    # full per-target copy). Ordered by citation strength, capped at 10.
    editorial_set = {d['domain'].lower() for d in editorial_domains}
    saved_by_dom = {(t.get('domain') or '').lower(): dict(t)
                    for t in (data.get('media_targets') or [])}
    outlet_name_by_dom = {d: _outlet_patterns()[d]['name'] for d in EDITORIAL_OUTLETS}

    def _synth_rationale(verdict):
        if verdict == 'strength':
            return f"{brand} already over-indexes here — defend and extend this relationship."
        if verdict == 'opportunity':
            return f"Competitors over-index here while {brand} is light — a priority pitch to close the gap."
        if verdict == 'neutral':
            return f"{brand} holds roughly even share here — steady-state coverage to maintain."
        return "Named as a citation source in your category by AI assistants — an emerging outlet worth a relationship."

    new_media = []
    for d in editorial_domains:
        dom = (d.get('domain') or '').lower()
        if dom not in editorial_set:
            continue
        if dom in saved_by_dom:
            t = saved_by_dom[dom]
            # Reconcile the LLM chips with the detected citing set.
            if d.get('llms'):
                t['cited_by_llms'] = d['llms']
        elif dom in EDITORIAL_OUTLETS:
            # Auto-ADD a card ONLY for high-confidence allowlisted outlets that
            # the original Claude pass never saw (their URLs had been pruned).
            # We deliberately do NOT auto-add arbitrary URL-classified editorial
            # domains Claude omitted by judgement (obscure blogs, brand sites
            # that slip past the competitor filter) — those still require a fresh
            # audit to be vetted. The registry is the precision gate.
            t = {
                'outlet': d.get('outlet_name') or outlet_name_by_dom.get(d.get('domain')) or d.get('domain'),
                'domain': d.get('domain'),
                'reporter': None,
                'citation_frequency': d.get('count', 0),
                'cited_by_llms': d.get('llms') or [],
                'sample_urls': (d.get('specific_urls') or [])[:5],
                'rationale': _synth_rationale(None),  # patched once SoV is known
                'gap_insight': None,
                '_synthesized': True,
            }
        else:
            # Neither a saved target nor an allowlisted outlet — skip it.
            # (Without this, `t` keeps the PREVIOUS iteration's value and gets
            # appended again, duplicating the prior card.)
            continue
        new_media.append(t)
        if len(new_media) >= 8:
            break
    for i, t in enumerate(new_media):
        t['rank'] = i + 1
    out['media_targets'] = new_media

    # Compute SoV for EXACTLY the media-target outlets the report shows, so every
    # card gets its competitor breakdown and the verdicts aren't capped out by
    # obscure non-target outlets.
    try:
        _sov_eds = _editorial_dicts_for_targets(ranked_domains, new_media) or editorial_domains
        # SoV: compare only against brand peers (retailers aren't direct
        # competitors; they sell the brand's products alongside others).
        _peer_competitors = [c for c in competitor_counts if c.get('type', 'brand_peer') == 'brand_peer']
        new_sov = _compute_outlet_share_of_voice(
            brand, _peer_competitors, all_responses, _sov_eds,
            max_outlets=max(10, len(_sov_eds)), brand_aliases=brand_aliases)
    except Exception as e:
        print(f"[rerender] SoV failed: {e}")
        new_sov = data.get('outlet_sov') or []
    out['outlet_sov'] = new_sov

    # Page-evidence layer — same as the fresh path; cache makes reruns cheap.
    if any(r.get("grounded") for r in all_responses):
        try:
            _attach_page_evidence(new_sov, ranked_domains, brand, competitor_counts)
        except Exception as _pe_e:
            print(f"[rerender] page-evidence failed (continuing): {_pe_e}")

    # Patch synthesized rationales now that each outlet's verdict is known.
    sov_by_dom = {(r.get('domain') or '').lower(): r for r in new_sov}
    for t in new_media:
        if t.get('_synthesized'):
            t['rationale'] = _synth_rationale(
                (sov_by_dom.get((t.get('domain') or '').lower()) or {}).get('verdict'))

    # Brand-coverage verification (BACKSTAGE ONLY — saved for the CSV, not shown
    # as a coverage tier). Only meaningful when grounded (real URLs).
    if any(r.get("grounded") for r in all_responses):
        try:
            _verify_brand_coverage(brand, out.get("media_targets") or [], ranked_domains)
        except Exception:
            pass
    # Reconcile coverage from the more-thorough page-evidence layer so guard +
    # headline + card labels agree (mirrors the fresh-audit path).
    try:
        _reconcile_coverage_from_page_evidence(out.get("media_targets"), new_sov)
    except Exception:
        pass
    # Coverage-guard: 'strength' requires on-page coverage; otherwise -> opportunity.
    try:
        _coverage_guard_verdicts(out.get("media_targets"), new_sov, brand)
    except Exception:
        pass
    try:
        _resort_sample_urls(out.get("media_targets"), brand)
    except Exception:
        pass
    # Rank by AI prominence (responses citing + citation frequency), not coverage.
    try:
        _sort_targets_by_prominence(out.get("media_targets"), new_sov)
    except Exception:
        pass

    # Headline move LAST — it must read post-guard verdicts, or a guarded
    # 'strength' fires a "Defend X" headline while X's card shows opportunity.
    # (Matches the fresh-audit path: guard -> sort -> headline.)
    try:
        out['headline_move'] = _compute_headline_move(brand, new_sov, new_media)
    except Exception:
        out['headline_move'] = None

    # Optionally refresh the 'What we found' executive summary with the
    # current prompt wording — the one piece the pure-Python pass can't
    # regenerate. One Claude call.
    if regenerate_summary:
        out['executive_summary'] = _regenerate_executive_summary(out)

    # Mark this view so the UI / operator knows it's a rerender, not the
    # original saved analysis.
    out['_rerendered'] = True

    # Re-run the ground-truth QA pass so a re-render reflects the CURRENT fixes
    # (an older payload may have been saved before a guard existed).
    try:
        out['qa'] = _qa_audit(out)
    except Exception as _qa_e:
        print("qa_audit (rerender) failed (continuing without):", str(_qa_e)[:200])
        out['qa'] = out.get('qa')

    return out


def _apply_display_editorial_filter(data):
    """Drop vendor / competitor / non-pitchable domains from the DISPLAYED outlet
    lists (outlet_sov, media_targets), in place. A report's saved data can predate
    a NON_EDITORIAL_DOMAINS addition, or simply never have had this quality bar
    applied at generation time; this cheap pure-Python filter keeps the cards +
    the on-the-fly earned_wins clean without re-running the audit. Raw
    all_responses and citations are left intact (CSV/JSON export still shows
    everything). Shared by the report view AND /admin/qa/<slug>, so the QA check
    always reflects what a viewer actually sees, not stale stored junk."""
    try:
        # _LOWQ_MEDIA_DOMAINS / _CORP_CONTENT_SUFFIXES are the module-level
        # constants shared with _qa_audit (defined above, near run_citation_audit)
        # — kept in one place so the QA check and the live display never disagree.
        _comp_labels = {re.sub(r'[^a-z0-9]', '', (c.get('name') or '').lower())
                        for c in (data.get('competitors') or []) if c.get('name')}

        def _ed_ok(o):
            dom = (o.get('domain') or '')
            dom_l = dom.lower()
            if classify_citation_domain(dom) in ('non_editorial', 'retail', 'defunct'):
                return False
            # Corporate content-marketing (a vendor's own "<Brand> Blog" / resource
            # center) is not a pitchable earned-media outlet — drop it unless the
            # domain is a known editorial publication. Catches spend-mgmt vendor
            # blogs (Fyle Blog, Precoro Blog, ApprovalMax Blog) that pass domain
            # classification but shouldn't surface as pitch targets.
            nm = (o.get('outlet') or o.get('name') or '').strip().lower()
            if nm.endswith(_CORP_CONTENT_SUFFIXES) and dom_l not in EDITORIAL_OUTLETS:
                return False
            if dom_l in _LOWQ_MEDIA_DOMAINS:
                return False
            # A direct COMPETITOR's own site is not an earned-media pitch target —
            # e.g. Kyndryl's report surfacing ensono.com (a rival managed-services
            # firm) as somewhere to "pitch." Registrable-label match, exact only.
            reg = re.sub(r'[^a-z0-9]', '', dom_l.split('.')[0]) if dom_l else ''
            if reg and reg in _comp_labels and dom_l not in EDITORIAL_OUTLETS:
                return False
            return True
        if isinstance(data.get('outlet_sov'), list):
            data['outlet_sov'] = [o for o in data['outlet_sov'] if _ed_ok(o)]
        if isinstance(data.get('media_targets'), list):
            data['media_targets'] = [o for o in data['media_targets'] if _ed_ok(o)]
    except Exception as _fe:
        print("display editorial-filter skipped:", str(_fe)[:120])
    return data


@app.route('/signal/<slug>')
def view_signal_report(slug):
    """Render a shared PR Signal Finder report.

    Operator-only refresh flags (both gated to FREE_AUDIT_BYPASS_IPS so
    public viewers always see the original saved report):
      ?fresh=1     — re-run the pure-Python pipeline (editorial filtering +
                     SoV) on cached responses. Zero API calls, instant.
      ?refresh=1   — same, PLUS regenerate the 'What we found' executive
                     summary with the current prompt wording. One Claude
                     call (~$0.01, ~10s). Use this to see the full current
                     report — summary included — on an existing audit
                     without a fresh 50-LLM batch.
    """
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))

    # First-party traffic log — count a real report view, not an operator
    # re-render/persist hit (fresh/refresh/save). Bots + operator IPs are
    # skipped inside _log_page_visit.
    if not (request.args.get('fresh') or request.args.get('refresh') or request.args.get('save')):
        _log_page_visit('report', slug=slug)

    data = _apply_display_editorial_filter(data)

    want_fresh = request.args.get('fresh') == '1'
    want_refresh = request.args.get('refresh') == '1'
    # Either the legacy IP allowlist OR the operator key can trigger a rerender —
    # the operator key is the one that actually works remotely (Render's proxy IP
    # isn't the caller's IP), so QA re-renders don't require an IP-allowlist edit.
    if (want_fresh or want_refresh) and (_ip_is_exempt_from_cap(_client_ip()) or _operator_ok()):
        try:
            data = _rerender_from_cached_responses(data, regenerate_summary=want_refresh)
            print(f"[rerender] applied current logic to slug={slug} "
                  f"(operator, summary_regen={want_refresh})")
            # &save=1 persists the correction back to the SharedResult row —
            # without it, a rerender is EPHEMERAL (this request's render only),
            # matching the original "preview current logic" behavior. Persisting
            # is gated strictly on the operator key (not the IP-exempt path),
            # since it's a write to production data, not just a preview.
            if request.args.get('save') == '1' and _operator_ok():
                rec = SharedResult.query.filter_by(slug=slug).first()
                if rec:
                    rec.payload = json.dumps(data, default=str)
                    db.session.commit()
                    print(f"[rerender] persisted correction to SharedResult slug={slug}")
        except Exception as e:
            print(f"[rerender] failed for slug={slug}: {e}")

    # 'What's working' lead section — computed on the fly so every report (incl.
    # frozen/older ones) shows it without a re-render. Deterministic, no API.
    try:
        data['earned_wins'] = _compute_earned_wins(
            data.get('brand') or '', data.get('outlet_sov') or [], data.get('all_responses') or [],
            brand_aliases=data.get('brand_aliases'))
        _attach_wins_recency(data['earned_wins'])   # dates + recency bias from CitedPage
    except Exception as _we:
        print("earned wins compute failed (continuing):", str(_we)[:120])

    # Announcement-Anchored mode — launch-landing metrics atop the report (only when
    # the audit was run as an announcement audit, i.e. it carries a product_name).
    if data.get('product_name') or data.get('mode') == 'announcement':
        try:
            data['launch_landing'] = _compute_launch_landing(
                data.get('brand') or '', data.get('product_name') or '',
                data.get('all_responses') or [], data.get('competitors') or [],
                brand_aliases=data.get('brand_aliases'))
        except Exception as _lle:
            print("launch landing compute failed (continuing):", str(_lle)[:120])

    share_url = request.url_root.rstrip('/') + url_for('view_signal_report', slug=slug)
    pdf_url = request.url_root.rstrip('/') + url_for('signal_report_pdf', slug=slug)
    csv_url = request.url_root.rstrip('/') + url_for('signal_report_csv', slug=slug)
    json_url = request.url_root.rstrip('/') + url_for('signal_report_json', slug=slug)
    user = current_signal_user()
    return render_template(
        'citation_audit.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        shared_data=data,
        share_url=share_url,
        pdf_url=pdf_url,
        csv_url=csv_url,
        json_url=json_url,
        signal_user=user,
        signal_credits=current_user_credits(user),
        operator_view=_operator_ok(),
    )


@app.route('/admin/qa/<slug>')
def admin_qa(slug):
    """Operator-only: run the ground-truth QA pass on any existing audit and
    return the full checks table + corrected numbers as JSON. Applies the same
    display-time media filter the live report uses, so this reflects what a
    viewer actually sees. Does not mutate the saved payload — use
    /admin/persist-rerender/<slug> to persist a correction."""
    if not _operator_ok():
        abort(404)
    data = _load_signal_report(slug)
    if not data:
        return jsonify({"error": f"no report for slug '{slug}'"}), 404
    data = _apply_display_editorial_filter(data)
    result = _qa_audit(data)
    return jsonify({
        "slug": slug, "brand": data.get('brand'), "category": data.get('category'),
        "client_ready": result['client_ready'], "blocking_failures": result['blocking_failures'],
        "checks": result['checks'], "corrected": result['corrected'],
    })


@app.route('/admin/persist-rerender/<slug>', methods=['POST'])
def admin_persist_rerender(slug):
    """Operator-only: run _rerender_from_cached_responses and WRITE the result
    back to the SharedResult row (unlike the report view's ?refresh=1, which is
    ephemeral / preview-only and never touches the DB). Isolated from the full
    page-render pipeline (no earned_wins/launch_landing/page-evidence side
    effects) so a failure here surfaces its real exception instead of being
    silently swallowed by an unrelated try/except elsewhere. ?refresh=1 also
    regenerates the executive summary (one Claude call)."""
    if not _operator_ok():
        abort(404)
    rec = SharedResult.query.filter_by(slug=slug).first()
    if not rec:
        return jsonify({"error": f"no report for slug '{slug}'"}), 404
    try:
        data = json.loads(rec.payload)
        data = _rerender_from_cached_responses(data, regenerate_summary=(request.args.get('refresh') == '1'))
        rec.payload = json.dumps(data, default=str)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-3000:]}), 500
    qa = data.get('qa') or {}
    return jsonify({
        "slug": slug, "saved": True, "brand": data.get('brand'),
        "brand_mention_count": data.get('brand_mention_count'),
        "brand_aliases": data.get('brand_aliases'), "brand_domains": data.get('brand_domains'),
        "client_ready": qa.get('client_ready'), "blocking_failures": qa.get('blocking_failures'),
    })


@app.route('/signal/<slug>/owned')
def view_signal_report_owned(slug):
    """Owned Signal Finder — the owned-media lens on the SAME shared audit as the
    PR Signal Finder report. No new audit: re-lenses the saved payload's cached
    citations toward the brand's vs competitors' OWN content + content gaps."""
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))
    owned = data.get('owned')
    if not owned:
        # Audit saved before the owned lens existed — compute on the fly from the
        # cached payload (pure-Python, no API), so every report is dual-lens.
        try:
            owned = _compute_owned_analysis(
                data.get('brand') or '',
                data.get('competitors') or [],
                data.get('related_brands') or [],
                data.get('raw_citation_domains') or [],
                data.get('all_responses') or [],
                brand_domain_hints=data.get('brand_domains') or [])
        except Exception as e:
            print(f"[owned] on-the-fly compute failed for {slug}: {e}")
            owned = {'owned_sov': {'rows': [], 'owned_total': 0, 'brand_owned': 0,
                                   'brand_share': 0.0, 'parent_folded': []},
                     'mix': {}, 'content_gaps': [], 'headline_move': None}
    # Top owned pages AI cites — the per-URL detail behind the owned share.
    # Computed on the fly so every report (incl. older payloads whose stored
    # `owned` predates this key) shows it, no re-render. Same pattern as earned_wins.
    if not owned.get('top_owned_urls'):
        try:
            _top = _compute_top_owned_urls(
                data.get('brand') or '',
                data.get('related_brands') or [],
                data.get('all_responses') or [],
                brand_domain_hints=data.get('brand_domains') or [])
            owned['top_owned_urls'] = _top['urls']
            owned['owned_page_mix'] = {
                'homepage': _top['homepage_citations'],
                'deep': _top['total_citations'] - _top['homepage_citations']}
        except Exception as e:
            print(f"[owned] top-URLs compute failed for {slug}: {e}")
            owned['top_owned_urls'] = []
    # Brand-level citation mix — same on-the-fly backfill for older payloads.
    if not owned.get('brand_mix'):
        try:
            owned['brand_mix'] = _compute_brand_citation_mix(
                data.get('brand') or '',
                data.get('competitors') or [],
                data.get('related_brands') or [],
                data.get('all_responses') or [],
                brand_domain_hints=data.get('brand_domains') or [],
                brand_aliases=data.get('brand_aliases') or [])
        except Exception as e:
            print(f"[owned] brand-mix compute failed for {slug}: {e}")
    return render_template(
        'citation_audit_owned.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        brand=data.get('brand') or 'this brand',
        category=data.get('category') or '',
        slug=slug,
        owned=owned,
        mix_bench=_current_owned_mix_bench(),
    )


# ---------------------------------------------------------------------------
# Tracked pitch links: /r/<token> — log genuine human opens of a shared report
# and email the operator the first time a named recipient opens theirs.
# ---------------------------------------------------------------------------

# Lowercase User-Agent substrings that identify link-preview crawlers and
# automated fetchers — NOT a human opening the report. LinkedIn/Slack/iMessage/
# Outlook all fetch a URL to build the preview card before any human clicks; if
# we counted those, every shared link would look "opened" instantly. Anything
# matching here (plus HEAD requests and empty UAs) is logged as a bot hit and
# never triggers the email alert.
_LINK_PREVIEW_BOT_UA = (
    'bot', 'crawler', 'spider', 'preview', 'linkpreview', 'link-preview',
    'linkedinbot', 'slackbot', 'slack-imgproxy', 'facebookexternalhit',
    'facebot', 'whatsapp', 'telegram', 'twitterbot', 'discordbot',
    'pinterest', 'redditbot', 'embedly', 'quora link', 'skypeuripreview',
    'vkshare', 'w3c_validator', 'baiduspider', 'yandex', 'duckduckbot',
    'bingbot', 'googlebot', 'google-read-aloud', 'applebot', 'petalbot',
    'semrush', 'ahrefs', 'mj12', 'dotbot', 'bytespider', 'gptbot',
    'oai-searchbot', 'chatgpt-user', 'perplexitybot', 'amazonbot',
    'iframely', 'metainspector', 'snapchat', 'tumblr', 'flipboard',
    'ms-office', 'microsoft office', 'outlook', 'safelinks',
    'curl', 'wget', 'python-requests', 'go-http-client', 'java/',
    'okhttp', 'headlesschrome', 'phantomjs', 'lighthouse', 'apache-httpclient',
    'axios', 'node-fetch', 'httpx', 'urllib', 'libwww', 'scrapy', 'zgrab',
)


def _is_link_preview_bot(ua, method):
    """True if this hit is an automated preview/crawler rather than a human
    opening the link. Conservative: HEAD requests and empty UAs count as bots."""
    if (method or 'GET').upper() == 'HEAD':
        return True
    ua = (ua or '').strip().lower()
    if not ua:
        return True
    return any(sig in ua for sig in _LINK_PREVIEW_BOT_UA)


def _fmt_et(dt):
    """Format a UTC datetime as US Eastern for the operator (who's in ET)."""
    if not dt:
        return '—'
    try:
        from zoneinfo import ZoneInfo
        et = dt.replace(tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo('America/New_York'))
        return et.strftime('%b %-d, %Y %-I:%M %p ET')
    except Exception:
        return dt.strftime('%b %d, %Y %H:%M UTC')


def _mint_tracked_link(slug, recipient=None, campaign=None, token=None):
    """Create + persist a unique /r/<token> link for a report slug. If `token`
    is supplied and still free, use it (stable seed links); else generate one."""
    if token and not TrackedLink.query.filter_by(token=token).first():
        chosen = token
    else:
        chosen = secrets.token_hex(4)  # 8 hex chars
        for _ in range(8):
            if not TrackedLink.query.filter_by(token=chosen).first():
                break
            chosen = secrets.token_hex(4)
    link = TrackedLink(token=chosen, slug=slug, recipient=recipient, campaign=campaign)
    db.session.add(link)
    db.session.commit()
    return link


def _send_click_email(recipient, slug, token, ip, ua, when_et, report_url, stats_url,
                      visitor_no=1):
    """Fire-and-forget alert when a tracked link gets a NEW unique visitor
    (a human open from an IP that hasn't opened this link before). visitor_no=1
    is the classic first-open ping; 2+ means a new device/network on the same
    link — often the link being forwarded internally, a stronger buying signal.
    Skipped silently if SENDGRID_API_KEY or the recipient address unset."""
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not to or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        who = recipient or f"Someone (link {token})"
        if visitor_no <= 1:
            subject = f"📬 {who} opened your Signal Finder report"
            lead = f"{who} just opened the report you shared."
            tail = f"This is their first open. Re-opens and the full list live at {stats_url}"
        else:
            subject = f"👀 New visitor #{visitor_no} on {who}'s link"
            lead = (f"{who}'s link was just opened from a new device/network — "
                    f"unique visitor #{visitor_no} on this link. If you only sent it to "
                    f"one person, it's likely being shared internally.")
            tail = f"Full open history at {stats_url}"
        text_body = (
            f"{lead}\n\n"
            f"Report:  {slug}  ({report_url})\n"
            f"When:    {when_et}\n"
            f"Link:    /r/{token}\n"
            f"IP:      {ip}\n"
            f"Device:  {ua[:300]}\n\n"
            f"{tail}"
        )
        html_body = (
            f'<p style="font-size:15px;margin:0 0 10px">{html.escape(lead).replace(html.escape(who), "<strong>" + html.escape(who) + "</strong>", 1)}</p>'
            f'<table style="font-size:13px;border-collapse:collapse">'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">Report</td>'
            f'<td><a href="{html.escape(report_url)}">{html.escape(slug)}</a></td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">When</td>'
            f'<td>{html.escape(when_et)}</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">Device</td>'
            f'<td>{html.escape(ua[:300])}</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">IP</td>'
            f'<td>{html.escape(ip or "")}</td></tr>'
            f'</table>'
            f'<p style="font-size:12px;color:#999;margin-top:12px">'
            f'{"First open. " if visitor_no <= 1 else f"Unique visitor #{visitor_no} on this link. "}'
            f'Full history: <a href="{html.escape(stats_url)}">{html.escape(stats_url)}</a></p>'
        )
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to],
            subject=subject,
            plain_text_content=text_body,
            html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("click-email send failed:", e)


def _tracked_link_target(slug):
    """Resolve a TrackedLink slug to its redirect path. Three forms:
    a report slug (default), '__home__' → homepage, and 'file:<name>' → a
    hosted PDF at static/reports/<name>.pdf (for tracked proposal links).
    The file name is sanitized to [a-z0-9-] — anything else (traversal
    attempts, dots, slashes) falls back to the homepage, same as a dead
    token. Serving is confined to static/reports/ by construction."""
    if slug == '__home__':
        return url_for('citation_audit')
    if (slug or '').startswith('file:'):
        name = slug[5:]
        if not re.fullmatch(r'[a-z0-9-]+', name or ''):
            return url_for('citation_audit')
        return url_for('static', filename=f'reports/{name}.pdf')
    return url_for('view_signal_report', slug=slug)


def _send_dashboard_visitor_email(slug, brand, ip, ua, when_et, visitor_no, report_url, ref_host):
    """Fire-and-forget alert when a report dashboard gets a NEW unique visitor
    who did NOT arrive via a tracked /r/ link (those are covered by
    _send_click_email — the dedup lives in _log_page_visit). Someone typed,
    bookmarked, or was forwarded the raw report URL."""
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not to or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        label = f"{brand} ({slug})" if brand else slug
        subject = f"👀 New visitor on the {label} dashboard"
        via = f"via {ref_host}" if ref_host else "direct (no referrer)"
        text_body = (
            f"A new unique visitor (#{visitor_no}) just viewed the {label} dashboard — "
            f"not through a tracked link.\n\n"
            f"Report:  {report_url}\n"
            f"When:    {when_et}\n"
            f"Source:  {via}\n"
            f"IP:      {ip}\n"
            f"Device:  {(ua or '')[:300]}\n\n"
            f"Unique-visitor counts live at /traffic."
        )
        html_body = (
            f'<p style="font-size:15px;margin:0 0 10px">A new unique visitor '
            f'(<strong>#{visitor_no}</strong>) just viewed the '
            f'<a href="{html.escape(report_url)}">{html.escape(label)}</a> dashboard — '
            f'not through a tracked link.</p>'
            f'<table style="font-size:13px;border-collapse:collapse">'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">When</td><td>{html.escape(when_et)}</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">Source</td><td>{html.escape(via)}</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">IP</td><td>{html.escape(ip or "")}</td></tr>'
            f'<tr><td style="padding:2px 12px 2px 0;color:#777">Device</td><td>{html.escape((ua or "")[:300])}</td></tr>'
            f'</table>'
        )
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to],
            subject=subject,
            plain_text_content=text_body,
            html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("dashboard-visitor email send failed:", e)


@app.route('/r/<token>')
def track_and_redirect(token):
    """Log a hit on a tracked pitch link, then 302 to the real target — a
    report, the homepage, or a hosted PDF (file: slugs). Bot/preview hits are
    recorded but never email; the first genuine human open pings the operator."""
    link = TrackedLink.query.filter_by(token=token).first()
    if not link:
        return redirect(url_for('citation_audit'))

    ua = request.headers.get('User-Agent', '') or ''
    ip = _client_ip()
    ref = (request.headers.get('Referer', '') or '')[:500]
    # A hit is a preview/scan (not a human read) if it has a bot UA OR comes from
    # a data-center IP. The IP check is what catches LinkedIn's own link-preview /
    # safety fetchers and corporate URL scanners — they fetch with real browser
    # UAs from hosting IPs, which is why near-instant "opens" fire in send order
    # for every prospect. (_ip_is_hosting is cached + fails open, and relies on
    # _client_ip having unwrapped the true visitor IP from Cloudflare.)
    is_bot = _is_link_preview_bot(ua, request.method) or _ip_is_hosting(ip)

    # Notify on every NEW unique visitor: a human open from an IP that hasn't
    # opened THIS link before (checked BEFORE inserting this row). The first
    # open is visitor #1; later new IPs usually mean the link was forwarded.
    # Repeat opens from a known IP stay silent.
    new_ip = (not is_bot) and (
        LinkClick.query.filter_by(token=token, is_bot=False, ip=ip).count() == 0
    )
    try:
        db.session.add(LinkClick(
            token=token, is_bot=is_bot, ip=ip,
            user_agent=ua[:1000], referer=ref,
        ))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("link-click log failed:", e)

    # A genuine human open advances the prospect's CRM status to 'opened'
    # (without clobbering replied/call/won/passed).
    if not is_bot:
        try:
            o = Outreach.query.filter_by(link_token=token).first()
            if o and o.status in ('queued', 'sent'):
                o.status = 'opened'
                o.last_activity_at = datetime.utcnow()
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("outreach open-sync failed:", e)

    if new_ip:
        from sqlalchemy import distinct as _distinct
        visitor_no = (db.session.query(db.func.count(_distinct(LinkClick.ip)))
                      .filter(LinkClick.token == token, LinkClick.is_bot.is_(False))
                      .scalar() or 1)   # AFTER insert, so this open is included
        report_url = request.url_root.rstrip('/') + _tracked_link_target(link.slug)
        stats_url = request.url_root.rstrip('/') + '/r-stats'
        when_et = _fmt_et(datetime.utcnow())
        threading.Thread(
            target=_send_click_email,
            args=(link.recipient, link.slug, token, ip, ua, when_et, report_url, stats_url),
            kwargs={'visitor_no': visitor_no},
            daemon=True,
        ).start()

    return redirect(_tracked_link_target(link.slug), code=302)


def _operator_ok():
    """Gate for operator-only routes that expose client data (the outreach board,
    link stats). Requires the LINK_STATS_KEY secret, supplied once via ?key=,
    which then sets a signed session cookie so the board's links and POST forms
    keep working without it. FAILS CLOSED: if no secret is configured these
    routes 404 for everyone — they must never default to open.

    Deliberately does NOT trust the client-IP allowlist. Behind Render's proxy,
    _client_ip() can collapse to a shared value for all external callers, so
    IP-gating these surfaces (the prior behavior) let everyone in once that
    shared IP was allowlisted — which publicly exposed the board."""
    secret = os.environ.get('LINK_STATS_KEY') or os.environ.get('OPERATOR_KEY')
    if not secret:
        return False
    if session.get('op_ok') is True:
        return True
    supplied = request.args.get('key') or request.headers.get('X-Operator-Key')
    if supplied and hmac.compare_digest(str(supplied), str(secret)):
        session['op_ok'] = True
        session.permanent = True
        return True
    return False


@app.route('/inbound')
def inbound_view():
    """Operator-gated warm-lead pipeline: every self-serve audit run, newest first.
    The problem_statement IS the lead (brand + what they want AI to surface them
    for). Attribution + geo shown per row. Never public — gated on the operator key
    (/inbound?key=<OPERATOR_KEY>)."""
    if not _operator_ok():
        abort(404)
    from datetime import timedelta
    now = datetime.utcnow()
    _INCOMPLETE = ['started', 'errored', 'rate_limited']
    show_ops = request.args.get('ops') == '1'          # include biz-dev's own runs
    status_filter = request.args.get('status', 'all')  # all | completed | incomplete
    # Default view = real DIY demand only (operator/batch runs hidden). is_operator
    # IS NOT TRUE matches both False and legacy NULL rows.
    _base = InboundAudit.query if show_ops else InboundAudit.query.filter(InboundAudit.is_operator.isnot(True))
    total = _base.count()
    n24 = _base.filter(InboundAudit.created_at >= now - timedelta(hours=24)).count()
    n7d = _base.filter(InboundAudit.created_at >= now - timedelta(days=7)).count()
    n_completed = _base.filter(InboundAudit.status == 'completed').count()
    n_incomplete = _base.filter(InboundAudit.status.in_(_INCOMPLETE)).count()
    n_operator = InboundAudit.query.filter(InboundAudit.is_operator.is_(True)).count()
    # Apply the status chip to the table rows.
    _rows_q = _base
    if status_filter == 'completed':
        _rows_q = _rows_q.filter(InboundAudit.status == 'completed')
    elif status_filter == 'incomplete':
        _rows_q = _rows_q.filter(InboundAudit.status.in_(_INCOMPLETE))
    rows = _rows_q.order_by(InboundAudit.created_at.desc()).limit(300).all()
    # Source roll-up over the same (real-lead) base — LinkedIn vs direct vs other.
    try:
        _sq = db.session.query(InboundAudit.utm_source, db.func.count(InboundAudit.id))
        if not show_ops:
            _sq = _sq.filter(InboundAudit.is_operator.isnot(True))
        src_rows = _sq.group_by(InboundAudit.utm_source).all()
        sources = sorted(((s or 'direct', c) for s, c in src_rows), key=lambda x: -x[1])[:6]
    except Exception:
        sources = []
    _k = request.args.get('key')
    import urllib.parse

    def _url(status=None, ops=None):
        p = {}
        if _k:
            p['key'] = _k
        _ops = show_ops if ops is None else ops
        if _ops:
            p['ops'] = '1'
        _st = status_filter if status is None else status
        if _st and _st != 'all':
            p['status'] = _st
        return '/inbound' + ('?' + urllib.parse.urlencode(p) if p else '')
    _csv_p = {}
    if _k:
        _csv_p['key'] = _k
    if show_ops:
        _csv_p['ops'] = '1'
    urls = {
        'all': _url(status='all'), 'completed': _url(status='completed'),
        'incomplete': _url(status='incomplete'), 'ops_toggle': _url(ops=(not show_ops)),
        'csv': '/inbound.csv' + ('?' + urllib.parse.urlencode(_csv_p) if _csv_p else ''),
    }
    return render_template('inbound.html', rows=rows, n24=n24, n7d=n7d, total=total,
                           sources=sources, status_filter=status_filter, show_ops=show_ops,
                           n_completed=n_completed, n_incomplete=n_incomplete,
                           n_operator=n_operator, urls=urls)


@app.route('/inbound.csv')
def inbound_csv():
    """Operator: export the inbound pipeline as CSV to work in a spreadsheet.
    Defaults to real DIY leads (operator/batch runs excluded); add ?ops=1 to
    include them."""
    if not _operator_ok():
        abort(404)
    import csv
    import io
    _q = InboundAudit.query
    if request.args.get('ops') != '1':
        _q = _q.filter(InboundAudit.is_operator.isnot(True))
    rows = _q.order_by(InboundAudit.created_at.desc()).limit(5000).all()
    root = os.environ.get("PUBLIC_BASE_URL", "https://signal.innatec3.com").rstrip('/')
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['created_utc', 'status', 'email', 'brand', 'category', 'problem_statement',
                'utm_source', 'utm_medium', 'utm_campaign', 'referrer', 'geo', 'ip',
                'is_operator', 'slug', 'report_url'])
    for r in rows:
        w.writerow([
            r.created_at.strftime('%Y-%m-%d %H:%M:%S') if r.created_at else '',
            r.status or '', r.email or '',
            r.brand or '', r.category or '', (r.problem_statement or '')[:1000],
            r.utm_source or '', r.utm_medium or '', r.utm_campaign or '',
            r.referrer or '', r.geo or '', r.ip or '',
            ('yes' if r.is_operator else 'no'), r.slug or '',
            f"{root}/signal/{r.slug}" if r.slug else '',
        ])
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=inbound_audits.csv'})


@app.route('/r-new')
def mint_tracked_link_route():
    """Operator: mint a tracked link. /r-new?slug=swarovski&to=Jennifer+McGuire
    Also mints links to hosted PDFs: ?slug=file:citi-proposal points /r/<token>
    at static/reports/citi-proposal.pdf (file must exist on the deploy).
    Optional &token= requests a stable vanity token (falls back to random if
    taken — check the returned URL)."""
    if not _operator_ok():
        abort(404)
    slug = (request.args.get('slug') or '').strip()
    if not slug:
        return Response("Provide ?slug=<report-slug>&to=<recipient>  "
                        "(or ?slug=file:<name> for a PDF at static/reports/<name>.pdf; "
                        "optional &token=<vanity-token>)\n", mimetype='text/plain')
    if slug.startswith('file:'):
        name = slug[5:]
        if not re.fullmatch(r'[a-z0-9-]+', name or ''):
            return Response("file: names may only contain a-z, 0-9 and dashes.\n",
                            status=400, mimetype='text/plain')
        if len(slug) > 32:
            return Response(f"slug '{slug}' is {len(slug)} chars — the column caps at 32. "
                            f"Use a shorter file name.\n", status=400, mimetype='text/plain')
        if not os.path.isfile(os.path.join(app.static_folder, 'reports', name + '.pdf')):
            return Response(f"No file at static/reports/{name}.pdf on this deploy.\n",
                            status=404, mimetype='text/plain')
    elif not _load_signal_report(slug):
        return Response(f"No report found for slug '{slug}'.\n", status=404, mimetype='text/plain')
    recipient = (request.args.get('to') or '').strip()[:160] or None
    campaign = (request.args.get('campaign') or '').strip()[:80] or None
    vanity = (request.args.get('token') or '').strip() or None
    if vanity and not re.fullmatch(r'[A-Za-z0-9-]{1,64}', vanity):
        return Response("Vanity tokens: letters, digits and dashes only, max 64 chars.\n",
                        status=400, mimetype='text/plain')
    link = _mint_tracked_link(slug, recipient, campaign, token=vanity)
    url = request.url_root.rstrip('/') + '/r/' + link.token
    return Response(url + "\n", mimetype='text/plain')


@app.route('/r-del')
def delete_tracked_link_route():
    """Operator: delete a mis-minted tracked link (/r-del?token=xyz). SAFETY:
    refuses if the link has ANY human opens — real engagement history is never
    deletable; this exists only for typo'd/test mints, which otherwise clutter
    /r-stats forever (there's no other delete path short of a DB shell). Bot
    hits on the deleted token are removed with it."""
    if not _operator_ok():
        abort(404)
    token = (request.args.get('token') or '').strip()
    if not token:
        return Response("Provide ?token=<token-to-delete>\n", mimetype='text/plain')
    link = TrackedLink.query.filter_by(token=token).first()
    if not link:
        return Response(f"No tracked link with token '{token}'.\n", status=404, mimetype='text/plain')
    humans = LinkClick.query.filter_by(token=token, is_bot=False).count()
    if humans:
        return Response(f"Refusing: /r/{token} has {humans} human open(s) — engagement "
                        f"history is not deletable.\n", status=409, mimetype='text/plain')
    bots = LinkClick.query.filter_by(token=token).delete()
    db.session.delete(link)
    db.session.commit()
    return Response(f"Deleted /r/{token} (slug {link.slug}, {bots} bot hits removed).\n",
                    mimetype='text/plain')


def _link_stats_rows(as_json=False):
    """Per-link open stats, sorted most-recently-opened first."""
    links = TrackedLink.query.order_by(TrackedLink.created_at.desc()).all()
    out = []
    for lk in links:
        humans = (LinkClick.query
                  .filter_by(token=lk.token, is_bot=False)
                  .order_by(LinkClick.clicked_at.asc()).all())
        bots = LinkClick.query.filter_by(token=lk.token, is_bot=True).count()
        out.append({
            'token': lk.token, 'slug': lk.slug,
            'recipient': lk.recipient, 'campaign': lk.campaign,
            'created_at': lk.created_at,
            'human_opens': len(humans), 'bot_hits': bots,
            # Distinct devices/networks behind the opens: 1 IP re-opening five
            # times is one engaged person; 4 IPs on one token means the link
            # was forwarded — a different (stronger) buying signal.
            'unique_ips': len({h.ip for h in humans if h.ip}),
            'first_open': humans[0].clicked_at if humans else None,
            'last_open': humans[-1].clicked_at if humans else None,
        })
    out.sort(key=lambda r: (r['last_open'] or datetime.min), reverse=True)
    if as_json:
        for r in out:
            for k in ('created_at', 'first_open', 'last_open'):
                r[k] = r[k].isoformat() if r[k] else None
    return out


@app.route('/r-stats.json')
def link_stats_json():
    if not _operator_ok():
        abort(404)
    return jsonify(_link_stats_rows(as_json=True))


def _operator_nav(active=''):
    """Shared top nav linking the operator-only pages, preserving ?key=."""
    k = request.args.get('key')
    q = f"?key={k}" if k else ""
    items = [('outreach', 'Outreach board', '/outreach'),
             ('traffic', 'Traffic', '/traffic'),
             ('inbound', 'Inbound', '/inbound'),
             ('dashboards', 'All dashboards', '/dashboards'),
             ('rstats', 'Link opens', '/r-stats')]
    links = []
    for key_, label, path in items:
        if key_ == active:
            links.append(f'<span style="font-weight:700;color:#1a1a1a">{label}</span>')
        else:
            links.append(f'<a href="{path}{q}">{label}</a>')
    return ('<div style="font-size:13px;margin:0 0 16px;padding-bottom:10px;'
            'border-bottom:1px solid #eee">' + ' &nbsp;·&nbsp; '.join(links) + '</div>')


@app.route('/traffic')
def traffic_view():
    """Operator-only first-party traffic dashboard: unique visitors, pageviews,
    top referrers, source breakdown, a 14-day trend, top-viewed reports, and the
    visits -> audits-run -> outreach-opens funnel. Built entirely from the
    PageVisit / InboundAudit / LinkClick tables — no third-party analytics."""
    if not _operator_ok():
        abort(404)
    from datetime import timedelta
    now = datetime.utcnow()

    def since(d):
        return now - timedelta(days=d)

    # Window stats: visits, unique IPs, home vs report split.
    windows = []
    for label, d in (('24 hours', 1), ('7 days', 7), ('30 days', 30)):
        base = PageVisit.query.filter(PageVisit.created_at >= since(d))
        windows.append({
            'label': label,
            'visits': base.count(),
            'uniques': base.with_entities(PageVisit.ip).distinct().count(),
            'home': base.filter(PageVisit.kind == 'home').count(),
            'report': base.filter(PageVisit.kind == 'report').count(),
        })
    total_visits = PageVisit.query.count()
    total_uniques = db.session.query(PageVisit.ip).distinct().count()

    # Top referrers (30d) — external hosts only (internal navigation is nulled at log time).
    referrers = (db.session.query(PageVisit.ref_host, db.func.count(PageVisit.id))
                 .filter(PageVisit.created_at >= since(30), PageVisit.ref_host.isnot(None))
                 .group_by(PageVisit.ref_host)
                 .order_by(db.func.count(PageVisit.id).desc()).limit(12).all())

    # Source breakdown (30d): LinkedIn / other-UTM / Referral / Direct.
    src_counts = {}
    for utm, rh in (PageVisit.query.with_entities(PageVisit.utm_source, PageVisit.ref_host)
                    .filter(PageVisit.created_at >= since(30)).all()):
        s = (utm or '').lower()
        if s == 'linkedin':
            key = 'LinkedIn'
        elif s:
            key = utm
        elif rh:
            key = 'Referral'
        else:
            key = 'Direct'
        src_counts[key] = src_counts.get(key, 0) + 1
    sources = sorted(src_counts.items(), key=lambda x: -x[1])

    # 14-day daily trend (visits + uniques).
    trend = []
    for i in range(13, -1, -1):
        d0 = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        d1 = d0 + timedelta(days=1)
        dq = PageVisit.query.filter(PageVisit.created_at >= d0, PageVisit.created_at < d1)
        trend.append({'day': d0.strftime('%-m/%-d'),
                      'visits': dq.count(),
                      'uniques': dq.with_entities(PageVisit.ip).distinct().count()})
    trend_max = max((t['visits'] for t in trend), default=0) or 1

    # Top-viewed reports (30d).
    from sqlalchemy import distinct as _distinct
    _rep = (db.session.query(PageVisit.slug,
                             db.func.count(_distinct(PageVisit.ip)),
                             db.func.count(PageVisit.id))
            .filter(PageVisit.kind == 'report', PageVisit.slug.isnot(None))
            .group_by(PageVisit.slug)
            .order_by(db.func.count(_distinct(PageVisit.ip)).desc(),
                      db.func.count(PageVisit.id).desc()).limit(20).all())
    # Map slug -> brand (cheap: InboundAudit is small) so hex slugs are readable;
    # fall back to the slug (seed prospects like 'lumen'/'swarovski' ARE the brand).
    _brand_by_slug = {}
    try:
        for _ia in InboundAudit.query.filter(InboundAudit.slug.isnot(None),
                                             InboundAudit.brand.isnot(None)).all():
            _brand_by_slug.setdefault(_ia.slug, _ia.brand)
    except Exception:
        pass
    top_reports = [(_slug, _uniq, _views, _brand_by_slug.get(_slug) or _slug)
                   for _slug, _uniq, _views in _rep]

    # Funnel (30d): homepage visits -> audits run -> genuine outreach opens.
    home_30 = PageVisit.query.filter(PageVisit.created_at >= since(30), PageVisit.kind == 'home').count()
    audits_30 = InboundAudit.query.filter(InboundAudit.created_at >= since(30)).count()

    # Outreach opens: UNIQUE = distinct tokens with >=1 human open in the window
    # (token is the person-level key — one minted per recipient — so this reads
    # "how many prospects opened"; distinct IP would double-count phone+laptop).
    # Raw event counts stay alongside so repeat-opens remain visible.
    def _opens(days=None):
        q = LinkClick.query.filter(LinkClick.is_bot.is_(False))
        if days is not None:
            q = q.filter(LinkClick.clicked_at >= since(days))
        return (q.with_entities(LinkClick.token).distinct().count(), q.count())

    opens_30, opens_30_total = _opens(30)

    # ---- Retroactive first-party history (predates the PageVisit table): every
    # self-serve audit-run carries referrer + source since launch (InboundAudit);
    # outreach opens (LinkClick) and total dashboards built (SharedResult) round
    # out the picture. Backfills the funnel + referral view immediately. ----
    audits_all = InboundAudit.query.count()
    audits_7 = InboundAudit.query.filter(InboundAudit.created_at >= since(7)).count()
    dashboards_built = SharedResult.query.count()
    opens_all, opens_all_total = _opens()
    opens_7, opens_7_total = _opens(7)

    # Daily audit-run trend (last 21 days).
    audit_trend = []
    for i in range(20, -1, -1):
        d0 = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        d1 = d0 + timedelta(days=1)
        audit_trend.append({'day': d0.strftime('%-m/%-d'),
                            'n': InboundAudit.query.filter(InboundAudit.created_at >= d0,
                                                           InboundAudit.created_at < d1).count()})
    audit_trend_max = max((t['n'] for t in audit_trend), default=0) or 1

    # What drove those audits, all-time: source breakdown + top referrer hosts.
    def _ref_host(u):
        try:
            from urllib.parse import urlparse
            h = (urlparse(u).hostname or u or '').lower()
            return h[4:] if h.startswith('www.') else h
        except Exception:
            return (u or '').lower()
    a_src, a_ref = {}, {}
    for utm, ref in InboundAudit.query.with_entities(InboundAudit.utm_source, InboundAudit.referrer).all():
        s = (utm or '').lower()
        key = 'LinkedIn' if s == 'linkedin' else (utm if s else ('Referral' if ref else 'Direct'))
        a_src[key] = a_src.get(key, 0) + 1
        if ref:
            h = _ref_host(ref)
            if h and 'innatec3.com' not in h:
                a_ref[h] = a_ref.get(h, 0) + 1
    audit_sources = sorted(a_src.items(), key=lambda x: -x[1])
    audit_referrers = sorted(a_ref.items(), key=lambda x: -x[1])[:10]

    return render_template('traffic.html',
                           nav=_operator_nav('traffic'), windows=windows,
                           total_visits=total_visits, total_uniques=total_uniques,
                           referrers=referrers, sources=sources, trend=trend, trend_max=trend_max,
                           top_reports=top_reports, home_30=home_30, audits_30=audits_30,
                           opens_30=opens_30, opens_30_total=opens_30_total,
                           audits_all=audits_all, audits_7=audits_7, dashboards_built=dashboards_built,
                           opens_all=opens_all, opens_7=opens_7,
                           opens_all_total=opens_all_total, opens_7_total=opens_7_total,
                           audit_trend=audit_trend,
                           audit_trend_max=audit_trend_max, audit_sources=audit_sources,
                           audit_referrers=audit_referrers,
                           keyq=(f"?key={request.args.get('key')}" if request.args.get('key') else ""))


@app.route('/r-stats')
def link_stats():
    if not _operator_ok():
        abort(404)
    rows = _link_stats_rows()
    root = request.url_root.rstrip('/')
    opened = sum(1 for r in rows if r['human_opens'])
    trs = []
    for r in rows:
        dot = '#1db954' if r['human_opens'] else '#ccc'
        track_url = f"{root}/r/{r['token']}"
        report_url = root + _tracked_link_target(r['slug'])
        recip = html.escape(r['recipient'] or '—')
        opens = r['human_opens']
        opens_cell = (f'<strong>{opens}</strong>'
                      f'<span style="color:#999;font-size:11px"> · {r["unique_ips"]} IP{"s" if r["unique_ips"] != 1 else ""}</span>'
                      if opens else '<span style="color:#bbb">0</span>')
        trs.append(
            '<tr>'
            f'<td><span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{dot};margin-right:7px"></span>{recip}</td>'
            f'<td><a href="{report_url}">{html.escape(r["slug"])}</a></td>'
            f'<td style="text-align:center">{opens_cell}</td>'
            f'<td>{html.escape(_fmt_et(r["first_open"]))}</td>'
            f'<td>{html.escape(_fmt_et(r["last_open"]))}</td>'
            f'<td style="text-align:center;color:#aaa">{r["bot_hits"]}</td>'
            f'<td><code style="font-size:11px;color:#888">{html.escape(track_url)}</code></td>'
            '</tr>'
        )
    body = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="robots" content="noindex"><title>Pitch link opens</title>'
        '<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        'margin:32px;color:#1a1a1a}h1{font-size:20px;margin:0 0 4px}'
        '.sub{color:#777;font-size:13px;margin:0 0 18px}'
        'table{border-collapse:collapse;width:100%;font-size:13px}'
        'th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #eee;vertical-align:top}'
        'th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#999;'
        'border-bottom:2px solid #ddd}'
        'tr:hover td{background:#fafafa}a{color:#2356c7;text-decoration:none}'
        'code{background:#f5f5f5;padding:1px 4px;border-radius:3px}</style></head><body>'
        + _operator_nav('rstats') +
        '<h1>Pitch link opens</h1>'
        f'<p class="sub">{opened} of {len(rows)} links opened by a human · bot/preview hits '
        'shown separately · times in ET · IP counts distinguish one person re-reading from '
        'a forwarded link (opens before the Jul 8 Cloudflare-IP fix log edge IPs, so older '
        'IP counts overstate)</p>'
        '<table><thead><tr><th>Recipient</th><th>Report</th><th>Human opens</th>'
        '<th>First open</th><th>Last open</th><th>Bot hits</th><th>Tracked link</th>'
        '</tr></thead><tbody>'
        + (''.join(trs) or '<tr><td colspan="7" style="color:#999">No tracked links yet. '
           'Mint one with <code>/r-new?slug=&lt;slug&gt;&amp;to=&lt;name&gt;</code>.</td></tr>')
        + '</tbody></table></body></html>'
    )
    return Response(body, mimetype='text/html')


# ---------------------------------------------------------------------------
# Outreach follow-up tracker (CRM-lite). Operator-only. Builds on tracked links:
# a recipient opening their /r/ link auto-advances status to 'opened', a daily
# digest reminds the operator who's due for follow-up (each with ready-to-paste
# proposed text seeded from that prospect's report insight), and the /outreach
# board gives one-click status controls.
# ---------------------------------------------------------------------------

_OUTREACH_STATUS_LABELS = {
    'queued': 'Queued', 'sent': 'Sent', 'opened': 'Opened',
    'followed_up': 'Followed up',
    'replied': 'Replied', 'call_scheduled': 'Call scheduled',
    'won': 'Won', 'cold': 'Cold', 'passed': 'Passed',
}
_OUTREACH_STATUS_COLORS = {
    'queued': '#9aa0a6', 'sent': '#2356c7', 'opened': '#1db954',
    'followed_up': '#e8820c',
    'replied': '#0a8a6f', 'call_scheduled': '#7a3ff2',
    'won': '#0a8a0a', 'cold': '#b00020', 'passed': '#9aa0a6',
}
# Statuses still "in play" for follow-up reminders.
_OUTREACH_ACTIVE = ('sent', 'opened', 'followed_up')


def _cadence_days(cadence):
    out = [int(p.strip()) for p in (cadence or '').split(',') if p.strip().isdigit()]
    return out or [5, 14]


def _outreach_first_name(o):
    raw = (o.prospect_name or '').split('—')[0].strip()
    return (raw.split(' ')[0] if raw else '') or 'there'


def _outreach_compute_due(o):
    """Next reminder date = sent_at + cadence[followup_count], or None if the
    cadence is exhausted."""
    if not o.sent_at:
        return None
    steps = _cadence_days(o.cadence)
    if o.followup_count >= len(steps):
        return None
    return (o.sent_at + timedelta(days=steps[o.followup_count])).date()


def _clean_relationship(rel):
    """Normalize a relationship hook for display/use. Operators sometimes type a
    non-hook ('none', 'skip', 'n/a', '-') to mean 'no shared connection' — treat
    those as empty so the opener stays clean."""
    rel = (rel or '').strip()
    if rel.lower() in ('none', 'n/a', 'na', 'skip', '-', '—'):
        return ''
    return rel


def _followup_text(o):
    """Proposed follow-up copy for this prospect's current step — warmer if they
    opened the link, soft break-up on the final step, and opens with the shared-
    connection hook (UO alum, ex-colleague, mutual contact) when one is set."""
    first = _outreach_first_name(o)
    rel = _clean_relationship(o.relationship)
    if o.status == 'opened':
        # Soft, relationship-first note for prospects who opened but haven't replied:
        # thank them, no heavy insight, frame it as an illustrative preview, invite a chat.
        lead = ""
        body = "thanks for taking a look. Hope it piqued your interest."
        close = ("It's purely an illustrative preview of the kind of insights consultative AI citation "
                 "analysis can surface. I'd love to chat about how you're thinking about AI visibility "
                 "from a comms standpoint.")
    else:
        insight = (o.insight or '').strip()
        lead = f"{insight} " if insight else ""
        steps = _cadence_days(o.cadence)
        is_last = o.followup_count >= len(steps) - 1
        body = "floating this back up in case it got buried."
        if is_last:
            close = "I'll leave it here either way, happy to send the full one-pager if AI visibility ever climbs the priority list."
        else:
            close = ("Quick note on why this isn't just another AI dashboard: I verify the actual pages behind "
                     "each citation rather than counting mentions, which is how I separate the coverage AI is "
                     "actually leaning on from the noise and spot the real PR openings. Happy to walk through yours.")
    # Lead with the shared-connection hook, EXCEPT on an opened follow-up where the
    # hook is a long sentence — there it just restates the original message's opener.
    if rel and not (o.status == 'opened' and len(rel) > 55):
        # relationship is its own warm clause; capitalize the body that follows.
        sep = '' if rel[-1] in '.!?' else '.'
        opener = f"Hi {first}, {rel}{sep} {body[:1].upper()}{body[1:]}"
    else:
        opener = f"Hi {first}, {body}"
    return f"{opener} {lead}{close}".strip()


def _outreach_mark(o, action):
    """Apply an operator status action in place."""
    now = datetime.utcnow()
    o.last_activity_at = now
    if action == 'sent':
        o.status = 'sent'
        o.sent_at = o.sent_at or now
        o.followup_count = 0
        o.next_followup_due = _outreach_compute_due(o)
    elif action == 'followup':
        o.status = 'followed_up'
        o.followup_count = (o.followup_count or 0) + 1
        o.next_followup_due = _outreach_compute_due(o)
    elif action == 'replied':
        o.status = 'replied'; o.next_followup_due = None
    elif action == 'call':
        o.status = 'call_scheduled'; o.next_followup_due = None
    elif action == 'won':
        o.status = 'won'; o.next_followup_due = None
    elif action == 'cold':
        o.status = 'cold'; o.next_followup_due = None
    elif action == 'passed':
        o.status = 'passed'; o.next_followup_due = None
    elif action == 'reopen':
        o.status = 'sent' if o.sent_at else 'queued'
        o.next_followup_due = _outreach_compute_due(o)


def _outreach_apply_status(o, status):
    """Set an arbitrary status directly (the board's manual status dropdown),
    keeping the derived fields consistent: 'sent'/'opened' stamp sent_at (once)
    and recompute the next reminder; terminal/closed states clear the reminder;
    'queued' resets to not-yet-sent. Unknown statuses are ignored."""
    if status not in _OUTREACH_STATUS_LABELS:
        return
    now = datetime.utcnow()
    o.last_activity_at = now
    o.status = status
    if status in ('sent', 'opened', 'followed_up'):
        o.sent_at = o.sent_at or now
        o.next_followup_due = _outreach_compute_due(o)
    elif status == 'queued':
        o.next_followup_due = None
    else:  # replied | call_scheduled | won | cold | passed
        o.next_followup_due = None


def _outreach_upsert(name, slug, title=None, company=None, channel='linkedin',
                     insight=None, cadence='5,14', token=None, relationship=None):
    """Idempotent create-or-update of a prospect. On first create, mints the
    prospect's tracked link (using `token` if given). Safe against concurrent
    gunicorn workers via the (prospect_name, slug) unique constraint."""
    o = Outreach.query.filter_by(prospect_name=name, slug=slug).first()
    if not o:
        recip = name + (f" — {company}" if company else "")
        link = _mint_tracked_link(slug, recip, "outreach", token=token)
        o = Outreach(prospect_name=name, slug=slug, link_token=link.token,
                     channel=channel or 'linkedin', cadence=cadence or '5,14')
        db.session.add(o)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            o = Outreach.query.filter_by(prospect_name=name, slug=slug).first()
            if o is None:
                raise
    changed = False
    if title is not None and o.prospect_title != title:
        o.prospect_title = title; changed = True
    if company is not None and o.company != company:
        o.company = company; changed = True
    if channel and o.channel != channel:
        o.channel = channel; changed = True
    if insight is not None and o.insight != insight:
        o.insight = insight; changed = True
    if relationship is not None and o.relationship != relationship:
        o.relationship = relationship; changed = True
    if cadence and o.cadence != cadence:
        o.cadence = cadence; changed = True
    if changed:
        db.session.commit()
    return o


def _send_outreach_digest_email(due):
    """Daily 'follow-ups due' digest with ready-to-paste proposed text."""
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not to or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        root = os.environ.get("PUBLIC_BASE_URL", "https://signal.innatec3.com").rstrip('/')
        n = len(due)
        subject = f"⏰ {n} Signal Finder follow-up{'s' if n != 1 else ''} due"
        txt, htm = [], []
        for o in due:
            days = (datetime.utcnow() - o.sent_at).days if o.sent_at else '?'
            opened = 'opened the link' if o.status == 'opened' else 'no open yet'
            text = _followup_text(o)
            txt.append(
                f"{o.prospect_name} ({o.company or o.slug}) — sent {days}d ago, {opened}\n"
                f"Report: {root}/signal/{o.slug}\n"
                f"Proposed: {text}\n"
            )
            htm.append(
                '<div style="margin:0 0 16px;padding:12px 14px;border:1px solid #eee;border-radius:8px">'
                f'<div style="font-size:14px"><strong>{html.escape(o.prospect_name)}</strong> '
                f'<span style="color:#888">· {html.escape(o.company or o.slug)} · sent {days}d ago · {opened}</span></div>'
                f'<div style="margin:8px 0;font-size:13px;color:#222;background:#f7f7f7;padding:10px;border-radius:6px">{html.escape(text)}</div>'
                f'<a href="{root}/signal/{html.escape(o.slug)}" style="font-size:12px">view report →</a></div>'
            )
        board = f"{root}/outreach"
        text_body = f"{n} follow-up(s) due.\n\n" + "\n".join(txt) + f"\nManage: {board}"
        html_body = (
            f'<p style="font-size:15px">{n} follow-up{"s" if n != 1 else ""} due today:</p>'
            + "".join(htm)
            + f'<p style="font-size:12px"><a href="{board}">Open the outreach board →</a></p>'
        )
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to], subject=subject,
            plain_text_content=text_body, html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("outreach digest email failed:", e)


def _run_daily_traffic_digest():
    """Daily roll-up of prospect report opens in the last 24h, emailed to the
    operator. Human opens only (link-preview bots excluded). Stays quiet on a
    zero-open day so the inbox only lights up when something happened."""
    with app.app_context():
        since = datetime.utcnow() - timedelta(hours=24)
        clicks = (LinkClick.query
                  .filter(LinkClick.clicked_at >= since, LinkClick.is_bot.is_(False))
                  .order_by(LinkClick.clicked_at.asc()).all())
        if not clicks:
            return
        by_token = {}
        for c in clicks:
            by_token.setdefault(c.token, []).append(c)
        rows = []
        for token, cs in by_token.items():
            o = Outreach.query.filter_by(link_token=token).first()
            tl = TrackedLink.query.filter_by(token=token).first()
            who = (o.prospect_name if o else None) or (tl.recipient if tl else None) or f"link {token}"
            slug = (o.slug if o else None) or (tl.slug if tl else '')
            company = (o.company if o else None) or slug
            # A genuine human open before this window => "repeat", else a first-time opener.
            prior = (LinkClick.query
                     .filter(LinkClick.token == token, LinkClick.is_bot.is_(False),
                             LinkClick.clicked_at < since).first())
            rows.append({
                'who': who, 'company': company, 'slug': slug,
                'opens': len(cs), 'last': cs[-1].clicked_at, 'is_new': prior is None,
            })
        rows.sort(key=lambda r: (-int(r['is_new']), -r['opens']))
        _send_daily_traffic_email(rows, len(clicks))


def _send_daily_traffic_email(rows, total_opens):
    """'Who opened your reports in the last 24h' digest. Skipped silently if
    SENDGRID_API_KEY or the operator address is unset."""
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not to or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        root = os.environ.get("PUBLIC_BASE_URL", "https://signal.innatec3.com").rstrip('/')
        n = len(rows)
        new_n = sum(1 for r in rows if r['is_new'])
        subject = f"📈 {n} prospect{'s' if n != 1 else ''} opened a report in the last 24h"
        txt, htm = [], []
        for r in rows:
            tag = 'NEW opener' if r['is_new'] else 'repeat'
            txt.append(
                f"{r['who']} ({r['company']}) — {r['opens']} open{'s' if r['opens'] != 1 else ''} "
                f"· {tag} · last {_fmt_et(r['last'])}\n  {root}/signal/{r['slug']}")
            htm.append(
                '<tr>'
                f'<td style="padding:6px 14px 6px 0"><strong>{html.escape(r["who"])}</strong>'
                f'<span style="color:#888"> · {html.escape(r["company"])}</span></td>'
                f'<td style="padding:6px 14px;text-align:center">{r["opens"]}</td>'
                f'<td style="padding:6px 14px;color:{"#1a7f37" if r["is_new"] else "#888"};font-weight:'
                f'{"600" if r["is_new"] else "400"}">{tag}</td>'
                f'<td style="padding:6px 0;color:#666">{html.escape(_fmt_et(r["last"]))}</td></tr>'
            )
        board = f"{root}/outreach"
        text_body = (
            f"{total_opens} human open{'s' if total_opens != 1 else ''} across {n} "
            f"prospect{'s' if n != 1 else ''} in the last 24h ({new_n} first-time).\n\n"
            + "\n".join(txt) + f"\n\nBoard: {board}")
        html_body = (
            f'<p style="font-size:15px"><strong>{total_opens}</strong> human open'
            f'{"s" if total_opens != 1 else ""} across <strong>{n}</strong> '
            f'prospect{"s" if n != 1 else ""} in the last 24h — {new_n} opening for the first time.</p>'
            '<table style="font-size:13px;border-collapse:collapse">'
            '<tr style="color:#999;text-align:left;font-size:11px;text-transform:uppercase">'
            '<td style="padding:0 14px 4px 0">Prospect</td>'
            '<td style="padding:0 14px 4px;text-align:center">Opens</td>'
            '<td style="padding:0 14px 4px">First/repeat</td>'
            '<td style="padding:0 0 4px">Last open</td></tr>'
            + "".join(htm) + '</table>'
            f'<p style="font-size:12px;margin-top:14px"><a href="{board}">Open the outreach board →</a></p>')
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to], subject=subject,
            plain_text_content=text_body, html_content=html_body,
        )
        _send_mail_object(msg)
    except Exception as e:
        print("daily traffic digest email failed:", e)


def _run_daily_inbound_digest():
    """Daily roll-up of self-serve audits run in the last 24h, emailed to the
    operator — the warm inbound pipeline. The problem_statement is the lead.
    Stays quiet on a zero-audit day so the inbox only lights up when it matters."""
    with app.app_context():
        since = datetime.utcnow() - timedelta(hours=24)
        items = (InboundAudit.query
                 .filter(InboundAudit.created_at >= since)
                 .order_by(InboundAudit.created_at.desc()).all())
        if not items:
            return
        _send_daily_inbound_email(items)


def _send_daily_inbound_email(items):
    """'Who ran the free audit in the last 24h' digest. Skipped silently if
    SENDGRID_API_KEY or the operator address is unset."""
    to = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not to or not sg_key:
        return
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        root = os.environ.get("PUBLIC_BASE_URL", "https://signal.innatec3.com").rstrip('/')
        n = len(items)
        li_n = sum(1 for r in items if (r.utm_source or '').lower() == 'linkedin')
        subject = (f"🧲 {n} inbound audit{'s' if n != 1 else ''} in the last 24h"
                   + (f" ({li_n} from LinkedIn)" if li_n else ""))
        txt, htm = [], []
        for r in items:
            src = r.utm_source or 'direct'
            geo = r.geo or '—'
            txt.append(
                f"{r.brand or '—'} — {(r.category or '')[:60]} · {src} · {geo} · {_fmt_et(r.created_at)}\n"
                f'  "{(r.problem_statement or "")[:160]}"\n  {root}/signal/{r.slug}')
            is_li = (r.utm_source or '').lower() == 'linkedin'
            htm.append(
                '<tr>'
                f'<td style="padding:6px 14px 6px 0;vertical-align:top"><strong>{html.escape(r.brand or "—")}</strong>'
                f'<div style="color:#888;font-size:12px">{html.escape((r.category or "")[:70])}</div>'
                f'<div style="color:#555;font-size:12px;max-width:340px">{html.escape((r.problem_statement or "")[:180])}</div></td>'
                f'<td style="padding:6px 14px;vertical-align:top;color:{"#1a7f37" if is_li else "#666"};'
                f'font-weight:{"600" if is_li else "400"}">{html.escape(src)}</td>'
                f'<td style="padding:6px 14px;vertical-align:top;color:#666">{html.escape(geo)}</td>'
                f'<td style="padding:6px 0;vertical-align:top;color:#666;white-space:nowrap">{html.escape(_fmt_et(r.created_at))}</td></tr>'
            )
        board = f"{root}/inbound"
        text_body = (
            f"{n} self-serve audit{'s' if n != 1 else ''} in the last 24h ({li_n} from LinkedIn).\n\n"
            + "\n\n".join(txt) + f"\n\nInbound view: {board}")
        html_body = (
            f'<p style="font-size:15px"><strong>{n}</strong> self-serve audit'
            f'{"s" if n != 1 else ""} in the last 24h'
            f'{f" — <strong>{li_n}</strong> from LinkedIn" if li_n else ""}.</p>'
            '<table style="font-size:13px;border-collapse:collapse">'
            '<tr style="color:#999;text-align:left;font-size:11px;text-transform:uppercase">'
            '<td style="padding:0 14px 4px 0">Brand / intent (the lead)</td>'
            '<td style="padding:0 14px 4px">Source</td>'
            '<td style="padding:0 14px 4px">Geo</td>'
            '<td style="padding:0 0 4px">When</td></tr>'
            + "".join(htm) + '</table>'
            f'<p style="font-size:12px;margin-top:14px"><a href="{board}">Open the inbound view →</a></p>')
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[to], subject=subject,
            plain_text_content=text_body, html_content=html_body,
        )
        _send_mail_object(msg)
        print(f"[inbound-digest] sent ({n} audits)")
    except Exception as e:
        print("daily inbound digest email failed:", str(e)[:160])


def _run_outreach_digest():
    """Auto-cold stale prospects, then email the operator everyone due today."""
    with app.app_context():
        today = datetime.utcnow().date()
        # Auto-cold: still active, cadence exhausted, quiet for 7+ days.
        stale = Outreach.query.filter(
            Outreach.status.in_(_OUTREACH_ACTIVE),
            Outreach.next_followup_due.is_(None),
        ).all()
        for o in stale:
            last = o.last_activity_at or o.sent_at
            if last and (datetime.utcnow() - last).days >= 7:
                o.status = 'cold'
        due = (Outreach.query
               .filter(Outreach.status.in_(_OUTREACH_ACTIVE),
                       Outreach.next_followup_due.isnot(None),
                       Outreach.next_followup_due <= today)
               .order_by(Outreach.next_followup_due.asc()).all())
        db.session.commit()
        if due:
            _send_outreach_digest_email(due)


# --- Seed: the five live LinkedIn-outreach prospects -----------------------
_OUTREACH_SEED = [
    dict(name="Bill Chandler", title="SVP, Global Communications & PR", company="Lululemon",
         slug="lululemon", channel="linkedin", token="7a7483ef",
         insight="Vuori — not Alo — is dead even with you at Women's Health and Business Insider, the two outlets that most shape how AI describes the category.",
         message=(
             "Hi Bill — I'd been in talks with some of your digital marketing colleagues a while back, "
             "but wanted to bring you something built specifically for communicators. Everyone's drowning "
             "in GEO dashboards right now; this is the opposite — a simple read that answers two questions "
             "about how AI describes your category: which outlets punch above their weight on your "
             "visibility, and which under-index you while quietly surfacing competitors instead.\n\n"
             "For lululemon, the standout: you're tied for #1 in AI mindshare — but it's Vuori, not Alo, "
             "that's dead even with you, matching you 19-for-19 at Women's Health and Business Insider, the "
             "two outlets with the most influence on how AI describes the category. Those are the "
             "relationships to defend before Vuori turns parity into a lead.\n\n"
             "Happy to send the full one-pager if useful — no pitch, just a lens I think is about to "
             "matter for earned media.")),
    dict(name="Kate Wolfe", title="VP, Marketing & Brand Strategy", company="SpotOn",
         slug="spoton", channel="linkedin", token="91e0f327",
         insight="At the outlets AI leans on most — TechRadar, Forbes Advisor, Business News Daily — Toast is named in nearly every response and SpotOn in fewer than half.",
         message=(
             "Hi Kate — can't believe it's been a decade-plus since our Edelman days! Quick one in that "
             "spirit: instead of another GEO dashboard, I built the opposite — a simple read that answers "
             "two questions about how AI describes \"best restaurant POS\": which outlets shape the answer, "
             "and which under-index you while surfacing competitors instead.\n\n"
             "For SpotOn, the pattern is consistent and actionable: at TechRadar, Forbes Advisor, Business "
             "News Daily and NerdWallet — the outlets AI leans on most — Toast is named in nearly every "
             "response and SpotOn in fewer than half. Those are your highest-leverage earned-media targets. "
             "(AI also skips SpotOn entirely on ChatGPT today — a specific, fixable gap, not a brand-wide "
             "weakness.)\n\n"
             "Happy to send the full one-pager if useful — no pitch, just a lens I think is about to "
             "matter for SMB discovery.")),
    dict(name="Carolyn Bos", title="VP, Corporate Marketing", company="Motive",
         slug="motive", channel="linkedin", token="11418dcd",
         insight="AI's category leader isn't Samsara — it's Geotab (42 responses vs your 32), and the trade press (CCJ, TechRadar) is where Geotab out-indexes you.",
         message=(
             "Hi Carolyn — quick one. Rather than another GEO dashboard, I built the opposite: a simple "
             "read for communicators that answers two questions about how AI describes \"best fleet "
             "management platform\" — which outlets shape the answer, and which under-index you while "
             "surfacing competitors instead.\n\n"
             "Two things stood out for Motive. First, AI's category leader isn't Samsara — it's Geotab "
             "(named in 42 responses vs. Motive's 32; Samsara, 33, is dead even with you). Second, the "
             "trade press is where to focus: at Commercial Carrier Journal, TechRadar, and Work Truck "
             "Online, Geotab out-indexes you — those are your earned-media targets — while FreightWaves "
             "and Forbes already punch above their weight for you.\n\n"
             "Happy to send the full one-pager if useful — no pitch, just a lens I think is about to "
             "matter for fleet-buyer discovery.")),
    dict(name="Sydney Williams", title="VP, Global Brand Marketing", company="ServiceNow",
         slug="servicenow", channel="linkedin", token="2e2af1ad",
         insight="You're genuinely top-tier (58% mindshare, neck-and-neck with Microsoft/IBM/Google) — the move is at IT Pro and Forbes, where IBM and Salesforce out-cite you.",
         message=(
             "Hi Sydney — can't believe it's been a decade-plus since our GE days! Quick one: everyone's "
             "pushing complex GEO dashboards right now; I built the opposite — a simple read for "
             "communicators that answers two questions about how AI describes your category: which outlets "
             "punch above their weight on your visibility, and which under-index you while quietly "
             "surfacing competitors instead.\n\n"
             "For ServiceNow, the read is encouraging and specific: when people ask the five major "
             "assistants about agentic AI platforms for the enterprise, you're genuinely top-tier — 58% "
             "mindshare, neck-and-neck with Microsoft, IBM, and Google, just behind Salesforce. The move "
             "is at the outlets AI leans on most: at IT Pro and Forbes, IBM and Salesforce out-cite "
             "ServiceNow, and at MarkTechPost AI names you alongside Microsoft but the cited pages only "
             "mention you in passing. Those few outlets are where earned media moves the needle.\n\n"
             "Happy to send the full one-pager if useful — no pitch, just a lens I think is about to "
             "matter for how buyers discover enterprise software.")),
    dict(name="Jennifer McGuire", title="Head of Brand & PR", company="Swarovski",
         slug="swarovski", channel="linkedin", token="e8e86b05",
         insight="Forbes is your single highest-return target: it's cited 15× across the audit and Mejuri owns it 10-to-3.",
         message=(
             "Hi Jennifer — still nostalgic about the good old Tiffany days; our exec-comms collaboration "
             "never quite came together, so I thought I'd come back with something new. Everyone's selling "
             "complex GEO dashboards right now; I built the opposite — a simple read for communicators "
             "that answers two questions about how AI describes your category: which outlets punch above "
             "their weight on your visibility, and which under-index you while quietly surfacing "
             "competitors instead.\n\n"
             "For Swarovski, the finding worth knowing: when people ask the five major assistants about "
             "crystal and accessible-luxury jewelry, the digital-native brands have caught you — Mejuri "
             "leads by a wide margin, and Missoma is now dead even with Swarovski, with Monica Vinader, "
             "Adina Eden, and BaubleBar just behind. More actionably: you under-index at the exact "
             "fashion-and-shopping outlets AI leans on most — Forbes, Elle, Who What Wear, AOL — where "
             "Mejuri is named far more often than you. Forbes is the single highest-return target: it's "
             "cited 15 times across the audit, and Mejuri owns it 10-to-3.\n\n"
             "Happy to send the full one-pager if useful — no pitch, just a lens I think is about to "
             "matter for how shoppers discover jewelry.")),
]


def _ensure_outreach_columns():
    """Add columns introduced AFTER the outreach table was first created.
    db.create_all() only creates missing tables, never alters existing ones, so
    a new field on a table that already exists in prod needs an explicit,
    idempotent ALTER. Postgres honors IF NOT EXISTS; on a fresh SQLite dev DB
    the column already exists (created by create_all) and the duplicate-column
    error is caught and ignored."""
    migrations = [
        "ALTER TABLE outreach ADD COLUMN IF NOT EXISTS relationship VARCHAR(240)",
        "ALTER TABLE outreach ADD COLUMN IF NOT EXISTS message TEXT",
        "ALTER TABLE outreach ADD COLUMN IF NOT EXISTS notes TEXT",
    ]
    try:
        with app.app_context():
            from sqlalchemy import text
            for stmt in migrations:
                try:
                    db.session.execute(text(stmt))
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print("outreach column migrate skipped:", str(e)[:120])
    except Exception as e:
        print("outreach column migrate error:", e)


def _ensure_inbound_columns():
    """Add the lead-capture columns introduced after inbound_audits was created:
    status (lifecycle), email (opt-in contact), is_operator (batch/operator flag).
    Idempotent ALTER — Postgres honors IF NOT EXISTS; a fresh SQLite dev DB already
    has them from create_all and the duplicate-column error is caught and ignored.
    Backfills existing rows: they all have a slug (completed) and no operator flag."""
    migrations = [
        "ALTER TABLE inbound_audits ADD COLUMN IF NOT EXISTS status VARCHAR(16)",
        "ALTER TABLE inbound_audits ADD COLUMN IF NOT EXISTS email VARCHAR(254)",
        "ALTER TABLE inbound_audits ADD COLUMN IF NOT EXISTS is_operator BOOLEAN",
        # Every pre-existing row was written only on completion, so mark them done.
        "UPDATE inbound_audits SET status='completed' WHERE status IS NULL AND slug IS NOT NULL",
        "UPDATE inbound_audits SET is_operator=FALSE WHERE is_operator IS NULL",
        # Widen ALL token-bearing columns 16 -> 64 so vanity tokens fit (a
        # 17-char token 500'd the mint, and link_clicks inserts failed silently
        # at the old width, dropping open-tracking for long tokens). Widening
        # is lossless; SQLite doesn't enforce VARCHAR length and errors on
        # these ALTERs — caught and skipped below.
        "ALTER TABLE tracked_links ALTER COLUMN token TYPE VARCHAR(64)",
        "ALTER TABLE link_clicks ALTER COLUMN token TYPE VARCHAR(64)",
        "ALTER TABLE outreach ALTER COLUMN link_token TYPE VARCHAR(64)",
    ]
    try:
        with app.app_context():
            from sqlalchemy import text
            for stmt in migrations:
                try:
                    db.session.execute(text(stmt))
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print("inbound column migrate skipped:", str(e)[:120])
    except Exception as e:
        print("inbound column migrate error:", e)


def _backfill_inbound_operator_flags():
    """One-time retroactive flag of pre-existing batch/operator InboundAudit rows
    (they predate is_operator, so the column migrate above left them FALSE and they
    would keep polluting the default /inbound view). Two reliable signals: the row
    ran from an exempt IP (operator/self testing), or its slug matches an
    ops+*@innatec3.com AuditLead (biz-dev's batch runner posts from those). Won't
    false-flag a real DIY lead. Idempotent — safe on every boot."""
    try:
        with app.app_context():
            changed = 0
            if _FREE_AUDIT_BYPASS_IPS:
                for r in InboundAudit.query.filter(
                        InboundAudit.ip.in_(list(_FREE_AUDIT_BYPASS_IPS)),
                        InboundAudit.is_operator.isnot(True)).all():
                    r.is_operator = True
                    changed += 1
            ops_slugs = [l.last_slug for l in
                         AuditLead.query.filter(AuditLead.last_slug.isnot(None)).all()
                         if _is_operator_email(l.email)]
            if ops_slugs:
                for r in InboundAudit.query.filter(
                        InboundAudit.slug.in_(ops_slugs),
                        InboundAudit.is_operator.isnot(True)).all():
                    r.is_operator = True
                    changed += 1
            if changed:
                db.session.commit()
                print(f"[inbound] backfilled is_operator on {changed} pre-existing batch/operator rows")
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        print("inbound operator backfill skipped:", str(e)[:120])


def _ensure_outreach_seed():
    """Idempotently ensure the five seed prospects exist (mints their links on
    first run, using their fixed tokens). Backfills the suggested message once
    when empty — never clobbers later board edits. Safe to call on every boot."""
    try:
        with app.app_context():
            for s in _OUTREACH_SEED:
                o = _outreach_upsert(s["name"], s["slug"], title=s["title"],
                                     company=s["company"], channel=s["channel"],
                                     insight=s["insight"], cadence="5,14",
                                     token=s["token"])
                if s.get("message") and o is not None and not (o.message or "").strip():
                    o.message = s["message"]
                    db.session.commit()
    except Exception as e:
        print("outreach seed error:", e)


def _outreach_keyq():
    """Preserve ?key= across redirects/forms so off-network access keeps working."""
    k = request.args.get('key')
    return f"?key={k}" if k else ""


def _outreach_redirect(oid=None, extra=None):
    """Redirect back to the board, preserving ?key=, optionally tacking on extra
    query params and a #card-<id> fragment so the browser lands on the row the
    operator just acted on (survives client-side re-sorting/filtering)."""
    url = url_for('outreach_board') + _outreach_keyq()
    if extra:
        url += ('&' if '?' in url else '?') + extra
    if oid is not None:
        url += f'#card-{oid}'
    return redirect(url)


def _qa_gate_for_outreach(data, force):
    """Block pointing a NEW outreach card at a report that fails the ground-truth
    QA pass — this is the actual mechanism behind "a bad report never reaches a
    prospect" (blocking the live report itself would break existing shared
    links; blocking card CREATION stops the bad number from ever being pitched).
    Pass force=True (the route's &force=1) to override once you've manually
    reviewed a report that's fine to send despite a non-blocking/edge-case flag.
    Returns (ok, error_message)."""
    if force:
        return True, None
    try:
        r = _qa_audit(_apply_display_editorial_filter(data))
        if not r['client_ready']:
            return False, ("Report failed QA (" + ", ".join(r['blocking_failures'])
                           + f"). Corrected brand count: {r['corrected'].get('brand_mention_count')}. "
                             "Fix + re-render (?refresh=1) before carding, or add &force=1 to override.")
    except Exception as e:
        print("qa gate check failed (allowing through):", str(e)[:150])
    return True, None


@app.route('/outreach/new')
def outreach_new():
    if not _operator_ok():
        abort(404)
    slug = (request.args.get('slug') or '').strip()
    name = (request.args.get('name') or '').strip()
    if not slug or not name:
        return Response("Provide ?name=&slug= (optional &title=&company=&channel=&insight=&cadence=)\n",
                        mimetype='text/plain')
    _report = _load_signal_report(slug)
    if not _report:
        return Response(f"No report found for slug '{slug}'.\n", status=404, mimetype='text/plain')
    _ok, _err = _qa_gate_for_outreach(_report, request.args.get('force') == '1')
    if not _ok:
        return Response(_err + "\n", status=409, mimetype='text/plain')
    o = _outreach_upsert(name, slug,
                     title=(request.args.get('title') or None),
                     company=(request.args.get('company') or None),
                     channel=(request.args.get('channel') or 'linkedin'),
                     insight=(request.args.get('insight') or None),
                     relationship=(request.args.get('relationship') or None),
                     cadence=(request.args.get('cadence') or '5,14'),
                     token=(request.args.get('token') or None))
    msg = request.args.get('message')
    if msg is not None and o is not None:
        o.message = (msg.strip() or None)
        db.session.commit()
    return redirect(url_for('outreach_board') + _outreach_keyq())


@app.route('/outreach/<int:oid>/set', methods=['POST'])
def outreach_set(oid):
    """Inline edit of fields from the board: free-text (relationship hook,
    message, insight, notes) plus a manual status override. Used both by full
    form posts and by the board's debounced autosave (fetch), which sends
    ajax=1 and expects an empty 204 back instead of a full board re-render."""
    if not _operator_ok():
        abort(404)
    o = Outreach.query.get(oid)
    if o:
        if 'relationship' in request.form:
            o.relationship = (request.form.get('relationship') or '').strip()[:240] or None
        if 'message' in request.form:
            o.message = (request.form.get('message') or '').strip() or None
        if 'insight' in request.form:
            o.insight = (request.form.get('insight') or '').strip() or None
        if 'notes' in request.form:
            o.notes = (request.form.get('notes') or '').strip() or None
        if request.form.get('status'):
            _outreach_apply_status(o, request.form.get('status'))
        db.session.commit()
    if request.form.get('ajax') or request.headers.get('X-Requested-With') == 'fetch':
        return ('', 204)
    return _outreach_redirect(oid)


@app.route('/outreach/add', methods=['POST'])
def outreach_add():
    """Quick-add a prospect from the board UI (vs. hand-typing /outreach/new?...).
    Validates the report slug exists, then upserts and lands you on the new card."""
    if not _operator_ok():
        abort(404)
    from urllib.parse import quote_plus
    name = (request.form.get('name') or '').strip()
    slug = (request.form.get('slug') or '').strip()
    if not name or not slug:
        return _outreach_redirect(extra='err=' + quote_plus('Name and report slug are required.'))
    _report = _load_signal_report(slug)
    if not _report:
        return _outreach_redirect(extra='err=' + quote_plus(f"No report found for slug '{slug}'."))
    _ok, _err = _qa_gate_for_outreach(_report, request.form.get('force') == '1')
    if not _ok:
        return _outreach_redirect(extra='err=' + quote_plus(_err))
    o = _outreach_upsert(
        name, slug,
        title=(request.form.get('title') or '').strip() or None,
        company=(request.form.get('company') or '').strip() or None,
        channel=(request.form.get('channel') or 'linkedin'),
        insight=(request.form.get('insight') or '').strip() or None,
        relationship=(request.form.get('relationship') or '').strip() or None,
        cadence=(request.form.get('cadence') or '5,14'))
    msg = (request.form.get('message') or '').strip()
    if msg and o is not None:
        o.message = msg
        db.session.commit()
    return _outreach_redirect(o.id if o else None, extra='added=' + quote_plus(name))


@app.route('/outreach/<int:oid>/<action>', methods=['POST'])
def outreach_action(oid, action):
    if not _operator_ok():
        abort(404)
    o = Outreach.query.get(oid)
    if o and action == 'delete':
        # hard-delete the prospect, its tracked link, and that link's click history
        tok = o.link_token
        if tok:
            LinkClick.query.filter_by(token=tok).delete()
            tl = TrackedLink.query.filter_by(token=tok).first()
            if tl:
                db.session.delete(tl)
        db.session.delete(o)
        db.session.commit()
        return redirect(url_for('outreach_board') + _outreach_keyq())
    if o and action in ('sent', 'followup', 'replied', 'call', 'won', 'cold', 'passed', 'reopen'):
        _outreach_mark(o, action)
        db.session.commit()
    return _outreach_redirect(oid)


# Action buttons offered per current status.
_OUTREACH_ACTIONS = {
    'queued': [('sent', 'Mark sent')],
    'sent': [('followup', 'Logged follow-up'), ('replied', 'Replied'),
             ('call', 'Call set'), ('cold', 'Cold'), ('passed', 'Pass')],
    'opened': [('followup', 'Logged follow-up'), ('replied', 'Replied'),
               ('call', 'Call set'), ('cold', 'Cold'), ('passed', 'Pass')],
    'followed_up': [('followup', 'Logged another follow-up'), ('replied', 'Replied'),
                    ('call', 'Call set'), ('cold', 'Cold'), ('passed', 'Pass')],
    'replied': [('call', 'Call set'), ('won', 'Won'), ('passed', 'Pass')],
    'call_scheduled': [('won', 'Won'), ('passed', 'Pass')],
    'won': [('reopen', 'Reopen')],
    'cold': [('reopen', 'Reopen')],
    'passed': [('reopen', 'Reopen')],
}

# Board styles + client behavior, kept out of the route body. The JS is a raw
# string so JS regex/\u escapes survive Python unscathed. It powers: debounced
# field autosave (fetch → 204), manual status set, the live hook→opener preview
# + one-click apply, status filters + name search + overdue sort (sessionStorage-
# persisted so they survive the POST→redirect reloads), and scroll restoration.
_OUTREACH_BOARD_CSS = (
    '<style>'
    ':root{--blue:#2356c7}'
    'body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:28px auto;max-width:820px;color:#1a1a1a;padding:0 16px}'
    'h1{font-size:20px;margin:0 0 2px}.sub{color:#777;font-size:13px;margin:0 0 16px}'
    'a{color:var(--blue);text-decoration:none}'
    'code{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:11px;color:#888}'
    'button{cursor:pointer;font-family:inherit}'
    '.toolbar{position:sticky;top:0;background:#fff;padding:10px 0 12px;z-index:5;border-bottom:1px solid #eee;margin-bottom:14px}'
    '.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px}'
    '.chip{font-size:12px;padding:4px 11px;border:1px solid #ddd;border-radius:20px;background:#fff;color:#444}'
    '.chip:hover{background:#f5f5f5}.chip.on{background:var(--blue);color:#fff;border-color:var(--blue)}'
    '.chip .cct{opacity:.65;font-weight:600;margin-left:4px}.chip.on .cct{opacity:.9}'
    '.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center}'
    '.search{flex:1;min-width:150px;font-size:13px;padding:6px 10px;border:1px solid #ddd;border-radius:6px}'
    '.tools button{font-size:12px;padding:6px 11px;border:1px solid #ccc;border-radius:6px;background:#fff}'
    '.tools button.prim{background:var(--blue);color:#fff;border-color:var(--blue);font-weight:600}'
    '.banner{font-size:13px;padding:9px 12px;border-radius:7px;margin-bottom:12px}'
    '.banner.err{background:#fdecea;color:#b00020;border:1px solid #f5c6cb}'
    '.banner.ok{background:#e9f8ef;color:#0a8a0a;border:1px solid #b8e6c8}'
    '.addbox{border:1px solid #e0e0e0;border-radius:10px;padding:14px;margin-bottom:14px;background:#fafbff}'
    '.addform input,.addform select{font-size:13px;padding:7px 9px;border:1px solid #ddd;border-radius:6px;width:100%;box-sizing:border-box;margin-bottom:8px}'
    '.addform .arow{display:flex;gap:8px}.addform .arow button{width:auto;margin:0}'
    '.card{border:1px solid #e7e7e7;border-radius:10px;padding:14px 16px;margin:0 0 12px;scroll-margin-top:130px}'
    '.chead{display:flex;justify-content:space-between;align-items:baseline;gap:10px}'
    '.pname{font-size:15px;font-weight:600}.ptitle{color:#888;font-size:13px}'
    '.hookbadge{font-size:11px;color:#7a3ff2;background:#f3eefe;padding:1px 8px;border-radius:10px;margin-left:6px}'
    '.pill{font-size:11px;font-weight:700;color:#fff;padding:2px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}'
    '.meta{font-size:12px;color:#777;margin-top:5px}'
    '.dashrow{margin-top:6px;font-size:12px}.dashrow a{font-weight:600}.muted{color:#aaa;font-size:11px}'
    '.fld{margin-top:10px}.flab{font-size:11px;color:#999;margin-bottom:4px}'
    '.autosave{width:100%;box-sizing:border-box;font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#222;padding:8px 11px;border:1px solid #ddd;border-radius:7px}'
    'textarea.autosave{resize:vertical}.hookin{padding:6px 9px}'
    '.sv{font-size:11px;margin-left:7px;font-weight:600}'
    '.hp{margin-top:6px}'
    '.hp-snip{font-size:12px;color:#444;background:#f3eefe;border:1px solid #e6dcfa;padding:7px 9px;border-radius:6px;font-style:italic}'
    '.hp-apply{margin-top:5px;font-size:11px;padding:3px 10px;border:1px solid #7a3ff2;color:#7a3ff2;background:#fff;border-radius:5px}'
    '.btns{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px}'
    '.btns button{font-size:12px;padding:5px 10px;border:1px solid #ccc;border-radius:5px;background:#fff}'
    '.btns button.prim{border-color:var(--blue);color:#fff;background:var(--blue);font-weight:600}'
    '.prop{margin-top:8px;font-size:12px;color:#333;background:#f7f7f7;padding:9px 11px;border-radius:6px}.prop .lbl{color:#999}'
    '.actions{margin-top:12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}'
    '.actions form{display:inline}.actions .setlbl{font-size:11px;color:#999}'
    '.statussel{font-size:12px;padding:4px 7px;border:1px solid #ccc;border-radius:5px;background:#fff}'
    '.act{font-size:12px;padding:4px 9px;border:1px solid #ccc;border-radius:5px;background:#fff}.act:hover{background:#f0f0f0}'
    '.delform{margin-left:auto}.act.del{color:#b00020;border-color:#f0c0c0}.act.del:hover{background:#fdecea}'
    '.emptymsg{color:#999;font-size:13px}'
    '</style>'
)

_OUTREACH_BOARD_JS = r'''<script>
function debounce(fn,ms){var t;return function(){var a=arguments,c=this;clearTimeout(t);t=setTimeout(function(){fn.apply(c,a);},ms);};}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function flash(b){var o=b.textContent;b.textContent='Copied ✓';setTimeout(function(){b.textContent=o;},1400);}
function cpEl(b){var t=document.getElementById(b.dataset.t);navigator.clipboard.writeText(t.value).then(function(){flash(b);});}
function cpV(b){navigator.clipboard.writeText(b.dataset.v).then(function(){flash(b);});}
function saveField(el){
  var fd=new FormData();fd.append(el.dataset.field,el.value);fd.append('ajax','1');
  var ind=document.getElementById('sv-'+el.dataset.field+'-'+el.dataset.oid);
  if(ind){ind.textContent='saving…';ind.style.color='#bbb';}
  fetch(el.dataset.action,{method:'POST',headers:{'X-Requested-With':'fetch'},body:fd})
   .then(function(r){if(ind){ind.textContent=r.ok?'saved ✓':'save failed';ind.style.color=r.ok?'#1db954':'#b00020';if(r.ok){setTimeout(function(){ind.textContent='';},1600);}}})
   .catch(function(){if(ind){ind.textContent='save failed';ind.style.color='#b00020';}});
}
var saveDebounced=debounce(function(el){saveField(el);},900);
function setStatus(sel){
  saveScroll();
  var f=document.createElement('form');f.method='POST';f.action=sel.dataset.action;
  var i=document.createElement('input');i.name='status';i.value=sel.value;f.appendChild(i);
  document.body.appendChild(f);f.submit();
}
var SKIP={'none':1,'n/a':1,'na':1,'skip':1,'-':1,'—':1};
function hookClean(h){h=(h||'').trim();return SKIP[h.toLowerCase()]?'':h;}
function hookOpener(first,hook,msg){
  hook=hookClean(hook).replace(/[.!?]+$/,'');
  if(!hook){return msg;}
  var clause=hook+'. ';
  var m=msg.match(/^(\s*Hi\s+[^\n—-]+\s*[—-]\s*)/);
  if(m){
    var rest=msg.slice(m[0].length);
    if(rest.toLowerCase().indexOf(hook.toLowerCase())===0){return msg;}
    return m[1]+clause+(rest?rest.charAt(0).toUpperCase()+rest.slice(1):'');
  }
  return 'Hi '+first+' — '+clause+(msg?msg.charAt(0).toUpperCase()+msg.slice(1):'');
}
function hookPreview(id){
  var hk=document.getElementById('hook-'+id),mg=document.getElementById('msg-'+id),pv=document.getElementById('hp-'+id);
  if(!hk||!mg||!pv){return;}
  var hook=hookClean(hk.value);
  if(!hook){pv.innerHTML='';return;}
  var opener=hookOpener(hk.dataset.first,hook,mg.value).split('\n')[0];
  if(opener.length>180){opener=opener.slice(0,180)+'…';}
  pv.innerHTML='<div class="hp-snip">'+esc(opener)+'</div><button type="button" class="hp-apply" onclick="applyHook('+id+')">Use this opener</button>';
}
function applyHook(id){
  var hk=document.getElementById('hook-'+id),mg=document.getElementById('msg-'+id);
  if(!hk||!mg){return;}
  mg.value=hookOpener(hk.dataset.first,hk.value,mg.value);
  saveField(mg);hookPreview(id);mg.focus();
}
function setFilter(f){try{sessionStorage.setItem('orFilter',f);}catch(e){}applyView();}
function applyView(){
  var f='all',q='';try{f=sessionStorage.getItem('orFilter')||'all';q=(sessionStorage.getItem('orSearch')||'').toLowerCase().trim();}catch(e){}
  var chips=document.querySelectorAll('.chip');for(var i=0;i<chips.length;i++){chips[i].classList.toggle('on',chips[i].dataset.f===f);}
  var cards=document.querySelectorAll('.card'),shown=0;
  for(var j=0;j<cards.length;j++){
    var c=cards[j];
    var okS=(f==='all')||(f==='due'?c.dataset.dueflag==='1':c.dataset.status===f);
    var okQ=!q||(c.dataset.search.indexOf(q)>=0);
    var v=okS&&okQ;c.style.display=v?'':'none';if(v){shown++;}
  }
  var e=document.getElementById('emptymsg');if(e){e.style.display=shown?'none':'';}
}
function toggleSort(){var m='default';try{m=sessionStorage.getItem('orSort')||'default';}catch(e){}m=(m==='due')?'default':'due';try{sessionStorage.setItem('orSort',m);}catch(e){}applySort();}
function applySort(){
  var m='default';try{m=sessionStorage.getItem('orSort')||'default';}catch(e){}
  var cont=document.getElementById('cards');if(!cont){return;}
  var arr=[].slice.call(cont.querySelectorAll('.card'));
  arr.sort(function(a,b){
    if(m==='due'){var d=(+a.dataset.due)-(+b.dataset.due);if(d){return d;}}
    return (+a.dataset.order)-(+b.dataset.order);
  });
  for(var i=0;i<arr.length;i++){cont.appendChild(arr[i]);}
  var btn=document.getElementById('sortbtn');if(btn){btn.textContent=(m==='due')?'Sort: overdue first':'Sort: default';}
}
function toggleAdd(){var b=document.getElementById('addbox');if(b){b.style.display=(b.style.display==='none')?'':'none';if(b.style.display===''){var n=b.querySelector('input[name=name]');if(n){n.focus();}}}}
function saveScroll(){try{sessionStorage.setItem('orScroll',String(window.scrollY));}catch(e){}}
window.addEventListener('scroll',debounce(saveScroll,200));
document.addEventListener('submit',function(){saveScroll();},true);
function restoreScroll(){
  if(location.hash){var el=document.querySelector(location.hash);if(el){el.scrollIntoView({block:'center'});return;}}
  var y=null;try{y=sessionStorage.getItem('orScroll');}catch(e){}
  if(y!==null){window.scrollTo(0,parseInt(y,10)||0);}
}
(function(){
  var as=document.querySelectorAll('.autosave');
  for(var i=0;i<as.length;i++){(function(el){el.addEventListener('input',function(){saveDebounced(el);});el.addEventListener('blur',function(){saveField(el);});})(as[i]);}
  var s=document.getElementById('search');
  if(s){try{s.value=sessionStorage.getItem('orSearch')||'';}catch(e){}s.addEventListener('input',function(){try{sessionStorage.setItem('orSearch',this.value);}catch(e){}applyView();});}
  applySort();applyView();
  var hs=document.querySelectorAll('.hookin');for(var k=0;k<hs.length;k++){hookPreview(hs[k].id.substring(5));}
  restoreScroll();
})();
</script>'''


@app.route('/outreach')
def outreach_board():
    if not _operator_ok():
        abort(404)
    rows = Outreach.query.order_by(Outreach.created_at.asc()).all()
    root = request.url_root.rstrip('/')
    keyq = _outreach_keyq()
    today = datetime.utcnow().date()
    active = sum(1 for r in rows if r.status in _OUTREACH_ACTIVE)
    won = sum(1 for r in rows if r.status == 'won')

    # status counts for the filter chips + how many need a follow-up right now
    counts = {}
    due_count = 0
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
        if (r.status in _OUTREACH_ACTIVE and r.next_followup_due
                and r.next_followup_due <= today):
            due_count += 1

    def _set_action(oid):
        return f"{root}/outreach/{oid}/set{keyq}"

    cards = []
    for idx, o in enumerate(rows):
        color = _OUTREACH_STATUS_COLORS.get(o.status, '#888')
        label = _OUTREACH_STATUS_LABELS.get(o.status, o.status)
        track_url = f"{root}/r/{o.link_token}" if o.link_token else '—'
        first = _outreach_first_name(o)
        # Per-card open analytics: genuine human opens of this prospect's /r/ link
        opens, last_open = 0, None
        if o.link_token:
            _oc = (LinkClick.query.filter_by(token=o.link_token, is_bot=False)
                   .order_by(LinkClick.clicked_at.desc()).all())
            opens = len(_oc)
            last_open = _oc[0].clicked_at if _oc else None
        if opens:
            open_badge = (f'<span style="color:#1db954;font-weight:600">👀 {opens} open'
                          f'{"s" if opens != 1 else ""}</span>'
                          f'<span style="color:#999"> · last {html.escape(_fmt_et(last_open))}</span>')
        else:
            open_badge = '<span style="color:#bbb">no opens yet</span>'
        days = f"{(datetime.utcnow() - o.sent_at).days}d ago" if o.sent_at else 'not sent'
        is_due = False
        if o.next_followup_due:
            overdue = o.next_followup_due <= today
            is_due = o.status in _OUTREACH_ACTIVE and overdue
            due = ('<span style="color:#b00020;font-weight:600">follow-up due</span>'
                   if overdue else
                   f'follow-up {html.escape(_fmt_et(datetime(o.next_followup_due.year, o.next_followup_due.month, o.next_followup_due.day)))[:12]}')
            dueval = o.next_followup_due.toordinal()
        else:
            due = '—'
            dueval = 99999999

        # quick one-click pipeline advances (kept alongside the manual dropdown)
        btns = []
        for act, lbl in _OUTREACH_ACTIONS.get(o.status, []):
            btns.append(
                f'<form method="post" action="{root}/outreach/{o.id}/{act}{keyq}">'
                f'<button type="submit" class="act">{html.escape(lbl)}</button></form>'
            )
        # manual status override — jump straight to any stage
        opts = ''.join(
            f'<option value="{s}"{" selected" if s == o.status else ""}>{html.escape(lbl)}</option>'
            for s, lbl in _OUTREACH_STATUS_LABELS.items())
        status_sel = (f'<select class="statussel" data-action="{_set_action(o.id)}" '
                      f'onchange="setStatus(this)" title="Set status manually">{opts}</select>')
        del_btn = (
            f'<form method="post" action="{root}/outreach/{o.id}/delete{keyq}" class="delform" '
            'onsubmit="return confirm(\'Delete this prospect entirely? This permanently removes the '
            'card and its tracked link, and cannot be undone.\');">'
            '<button type="submit" class="act del" title="Delete prospect">Delete</button></form>')

        show_text = o.status in _OUTREACH_ACTIVE
        fu = _followup_text(o) if show_text else ''
        proposed = (f'<div class="prop"><span class="lbl">proposed follow-up:</span> '
                    f'<button type="button" class="act" data-v="{html.escape(fu)}" onclick="cpV(this)">Copy</button>'
                    f'<br>{html.escape(fu)}</div>') if show_text else ''
        rel_show = _clean_relationship(o.relationship)
        rel_badge = (f'<span class="hookbadge">🤝 {html.escape(rel_show)}</span>' if rel_show else '')

        # shared-connection hook — autosaves + drives the live opener preview
        hook_block = (
            '<div class="fld"><div class="flab">shared connection — weave into the opener'
            f'<span class="sv" id="sv-relationship-{o.id}"></span></div>'
            f'<input id="hook-{o.id}" class="autosave hookin" data-oid="{o.id}" data-field="relationship" '
            f'data-action="{_set_action(o.id)}" data-first="{html.escape(first)}" '
            f'value="{html.escape(o.relationship or "")}" '
            f'placeholder="e.g. fellow Oregon alum · ex-Edelman · intro via Sam" oninput="hookPreview({o.id})">'
            f'<div class="hp" id="hp-{o.id}"></div></div>'
        )
        # suggested message — autosaves; copy controls (no redundant +link button)
        msg_block = (
            f'<div class="fld"><div class="flab">suggested message<span class="sv" id="sv-message-{o.id}"></span></div>'
            f'<textarea id="msg-{o.id}" class="autosave msg" rows="8" data-oid="{o.id}" data-field="message" '
            f'data-action="{_set_action(o.id)}">{html.escape(o.message or "")}</textarea>'
            '<div class="btns">'
            f'<button type="button" class="prim" data-t="msg-{o.id}" onclick="cpEl(this)">Copy message</button>'
            f'<button type="button" data-v="{html.escape(track_url)}" onclick="cpV(this)">Copy link</button>'
            '</div></div>'
        )
        # private notes — autosaves
        notes_block = (
            f'<div class="fld"><div class="flab">notes<span class="sv" id="sv-notes-{o.id}"></span></div>'
            f'<textarea id="notes-{o.id}" class="autosave notes" rows="2" data-oid="{o.id}" data-field="notes" '
            f'data-action="{_set_action(o.id)}" placeholder="private — never sent">{html.escape(o.notes or "")}</textarea></div>'
        )
        search_blob = html.escape(' '.join(
            x for x in (o.prospect_name, o.company, o.prospect_title, o.slug) if x).lower())
        cards.append(
            f'<div class="card" id="card-{o.id}" data-status="{o.status}" data-order="{idx}" '
            f'data-due="{dueval}" data-dueflag="{1 if is_due else 0}" data-search="{search_blob}">'
            '<div class="chead">'
            f'<div><span class="pname">{html.escape(o.prospect_name)}</span> '
            f'<span class="ptitle">{html.escape(o.prospect_title or "")}'
            f'{" · " + html.escape(o.company) if o.company else ""}</span>{rel_badge}</div>'
            f'<span class="pill" style="background:{color}">{html.escape(label)}</span></div>'
            f'<div class="meta"><a href="{root}/signal/{html.escape(o.slug)}">{html.escape(o.slug)}</a> · '
            f'{days} · {due} · {open_badge} · <code>{html.escape(track_url)}</code></div>'
            f'<div class="dashrow"><a href="{root}/signal/{html.escape(o.slug)}" target="_blank">📊 Open dashboard</a> '
            '<span class="muted">(preview — doesn\'t count as an open)</span></div>'
            f'{hook_block}{msg_block}{notes_block}{proposed}'
            f'<div class="actions"><span class="setlbl">Status</span>{status_sel}{"".join(btns)}{del_btn}</div>'
            '</div>'
        )

    # ---- filter chips: All, Due, then every status that has prospects ----
    chip_defs = [('all', 'All', len(rows))]
    if due_count:
        chip_defs.append(('due', 'Due', due_count))
    for s, lbl in _OUTREACH_STATUS_LABELS.items():
        if counts.get(s):
            chip_defs.append((s, lbl, counts[s]))
    chips = ''.join(
        f'<button type="button" class="chip" data-f="{f}" onclick="setFilter(\'{f}\')">'
        f'{html.escape(lbl)}<span class="cct">{ct}</span></button>'
        for f, lbl, ct in chip_defs)

    # ---- quick-add: datalist of report slugs so the operator can't fat-finger one ----
    slug_opts = ''.join(
        f'<option value="{html.escape(s)}">'
        for (s,) in db.session.query(SharedResult.slug)
                              .order_by(SharedResult.created_at.desc()).all())
    add_form = (
        '<div id="addbox" class="addbox" style="display:none">'
        f'<form method="post" action="{root}/outreach/add{keyq}" class="addform">'
        '<div class="arow"><input name="name" placeholder="Prospect name *" required>'
        '<input name="slug" list="slugs" placeholder="Report slug *" required></div>'
        '<div class="arow"><input name="title" placeholder="Title">'
        '<input name="company" placeholder="Company">'
        '<select name="channel"><option value="linkedin">LinkedIn</option>'
        '<option value="email">Email</option></select></div>'
        '<input name="relationship" placeholder="Shared connection (optional)">'
        '<input name="insight" placeholder="One-line lead insight (optional)">'
        '<div class="arow"><button type="submit" class="prim">Add prospect</button>'
        '<button type="button" onclick="toggleAdd()">Cancel</button></div></form>'
        f'<datalist id="slugs">{slug_opts}</datalist></div>'
    )

    # ---- post-action banners ----
    err = request.args.get('err')
    added = request.args.get('added')
    if err:
        banner = f'<div class="banner err">{html.escape(err)}</div>'
    elif added:
        banner = f'<div class="banner ok">Added {html.escape(added)} ✓</div>'
    else:
        banner = ''

    cadence_label = html.escape(rows[0].cadence) if rows else "5,14"
    toolbar = (
        '<div class="toolbar"><div class="chips">' + chips + '</div>'
        '<div class="tools"><input id="search" class="search" placeholder="Search name / company…">'
        '<button type="button" id="sortbtn" onclick="toggleSort()">Sort: default</button>'
        '<button type="button" class="prim" onclick="toggleAdd()">＋ Add prospect</button></div></div>'
    )

    body = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="robots" content="noindex"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Outreach board</title>' + _OUTREACH_BOARD_CSS + '</head><body>'
        + _operator_nav('outreach')
        + '<h1>Outreach board</h1>'
        + f'<p class="sub">{len(rows)} prospects · {active} active · {won} won · '
          f'opens auto-advance to “Opened” · reminders at days {cadence_label} after sent</p>'
        + banner + toolbar + add_form
        + '<div id="cards">' + ''.join(cards) + '</div>'
        + ('' if rows else '<p class="emptymsg">No prospects yet — use ＋ Add prospect above.</p>')
        + '<p id="emptymsg" class="emptymsg" style="display:none">No prospects match this filter.</p>'
        + _OUTREACH_BOARD_JS
        + '</body></html>'
    )
    return Response(body, mimetype='text/html')


@app.route('/dashboards')
def dashboards_index():
    """Operator-only repository of every audit dashboard built to date — including
    samples and demos not tied to an outreach prospect."""
    if not _operator_ok():
        abort(404)
    root = request.url_root.rstrip('/')
    # Memory-safe: bulk-fetch only id/slug/date (NOT the big payloads), then load
    # and parse each payload one at a time so peak memory is ~one report, not all of
    # them at once (payloads total tens of MB across all reports).
    recs = (db.session.query(SharedResult.id, SharedResult.slug, SharedResult.created_at)
            .order_by(SharedResult.created_at.desc()).all())
    pros = {o.slug: o for o in Outreach.query.all()}
    rows_html = []
    for rid, slug, created_at in recs:
        raw = db.session.query(SharedResult.payload).filter(SharedResult.id == rid).scalar()
        try:
            p = json.loads(raw) if raw else {}
        except Exception:
            p = {}
        raw = None
        brand = p.get('brand') or slug
        mind = ''
        if p.get('brand_mention_count') is not None and p.get('total_responses'):
            pct = round(100 * p['brand_mention_count'] / p['total_responses'])
            mind = f"{p['brand_mention_count']}/{p['total_responses']} · {pct}%"
        hm = p.get('headline_move') or {}
        move = f"{hm.get('verb', '')} {hm.get('outlet', '')}".strip()
        p = None
        o = pros.get(slug)
        if o:
            who = html.escape(o.prospect_name) + (f" · {html.escape(o.company)}" if o.company else "")
        else:
            who = '<span style="color:#bbb">— sample —</span>'
        rows_html.append(
            '<tr>'
            f'<td><a href="{root}/signal/{html.escape(slug)}" target="_blank">{html.escape(brand)}</a><br>'
            f'<code style="font-size:11px;color:#999">{html.escape(slug)}</code></td>'
            f'<td>{html.escape(mind)}</td>'
            f'<td style="font-size:12px">{html.escape(move) or "—"}</td>'
            f'<td style="font-size:12px">{who}</td>'
            f'<td style="font-size:11px;color:#888">{html.escape(_fmt_et(created_at))}</td>'
            f'<td style="font-size:11px"><a href="{root}/signal/{html.escape(slug)}.json" target="_blank">json</a> · '
            f'<a href="{root}/signal/{html.escape(slug)}.csv" target="_blank">csv</a></td>'
            '</tr>'
        )
    body = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="robots" content="noindex"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>All dashboards</title>'
        '<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        'margin:28px auto;max-width:980px;color:#1a1a1a;padding:0 16px}'
        'h1{font-size:20px;margin:0 0 2px}.sub{color:#777;font-size:13px;margin:0 0 18px}'
        'a{color:#2356c7;text-decoration:none}code{background:#f3f3f3;padding:1px 4px;border-radius:3px}'
        'table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top}'
        'th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#999}</style></head><body>'
        + _operator_nav('dashboards') +
        '<h1>All dashboards</h1>'
        f'<p class="sub">{len(recs)} reports built to date · newest first · samples + client work</p>'
        '<table><thead><tr><th>Brand / slug</th><th>AI mindshare</th><th>Headline move</th>'
        '<th>Prospect</th><th>Built</th><th>Export</th></tr></thead><tbody>'
        + ''.join(rows_html) +
        '</tbody></table></body></html>'
    )
    return Response(body, mimetype='text/html')


@app.route('/signal/<slug>.pdf')
def signal_report_pdf(slug):
    """Render the saved report as a PDF via WeasyPrint."""
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))
    try:
        from weasyprint import HTML  # imported lazily — heavy dependency
    except Exception as e:
        print("WeasyPrint import failed:", e)
        return jsonify({"error": "PDF export is not available in this environment."}), 503

    html_str = render_template('citation_audit_print.html', data=data)
    try:
        pdf_bytes = HTML(string=html_str, base_url=request.url_root).write_pdf()
    except Exception as e:
        print("WeasyPrint render failed:", e)
        return jsonify({"error": "Failed to render PDF."}), 500

    brand_slug = re.sub(r'[^a-z0-9]+', '-', (data.get('brand') or 'report').lower()).strip('-') or 'report'
    filename = f"signal-finder-{brand_slug}-{slug}.pdf"
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'inline; filename="{filename}"'}
    )


@app.route('/signal/<slug>.csv')
def signal_report_csv(slug):
    """Return the saved report as a multi-section CSV download.

    Sections (each preceded by a `==== SECTION NAME ====` header row so Excel
    users can navigate):
      1. METADATA — brand, category, totals
      2. PROMPTS — the full prompt list sent to the LLMs
      3. LLM RESPONSES — one row per (prompt × LLM) with the full response
         text + per-response citation URLs (pipe-joined)
      4. ALL CITATION URLS — one row per extracted URL with domain + which
         LLM/prompt surfaced it (verified post-resolution)
      5. SHARE OF VOICE BY OUTLET — strength / opportunity verdicts per outlet
      6. EDITORIAL MEDIA TARGETS
      7. INSTITUTIONAL TARGETS
      8. ANALYST TARGETS

    The 5-column "raw" schema for sections 2–5 is intentionally wider than
    the analyzed-targets schema so each section reads cleanly in Excel even
    with mixed column counts. Excel treats blank-line + bold-text-row pairs
    as informal section dividers — there's no native CSV mechanism for this,
    so this is the most-portable approach. Mirrors the PDF route's behavior
    (loads SharedResult by slug, flashes + redirects on miss).
    """
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))

    buf = io.StringIO()
    writer = csv.writer(buf)

    def _join(v):
        if not v:
            return ""
        if isinstance(v, (list, tuple)):
            return "; ".join(str(x) for x in v)
        return str(v)

    def _join_urls(v):
        # Pipe-join URLs so they don't collide with the semicolon used for
        # lists elsewhere — easier to split downstream.
        if not v:
            return ""
        if isinstance(v, (list, tuple)):
            return " | ".join(str(x) for x in v)
        return str(v)

    def _section(label):
        writer.writerow([])
        writer.writerow([f"==== {label} ===="])

    # --- 1. METADATA ---
    _section("METADATA")
    writer.writerow(["field", "value"])
    writer.writerow(["brand", data.get("brand") or ""])
    writer.writerow(["category", data.get("category") or ""])
    writer.writerow(["brand_mention_count", data.get("brand_mention_count") or 0])
    writer.writerow(["total_responses", data.get("total_responses") or 0])
    writer.writerow(["total_citations_extracted", data.get("total_citations_extracted") or 0])
    writer.writerow(["responses_completed", data.get("responses_completed") or 0])
    writer.writerow(["responses_planned", data.get("responses_planned") or 0])
    writer.writerow(["completion_rate", data.get("completion_rate") or ""])
    writer.writerow(["tier", data.get("tier") or ""])
    writer.writerow(["slug", data.get("slug") or slug])

    # --- 2. PROMPTS ---
    _section("PROMPTS")
    writer.writerow(["prompt_id", "prompt_text"])
    for i, p in enumerate(data.get("prompts_used") or [], 1):
        writer.writerow([i, p])

    # Build a prompt→id lookup so LLM RESPONSES + ALL CITATION URLS can
    # cross-reference the prompt by its index in the PROMPTS section.
    prompt_id_lookup = {p: i for i, p in enumerate(data.get("prompts_used") or [], 1)}

    # --- 3. LLM RESPONSES ---
    _section("LLM RESPONSES (one row per prompt × LLM call)")
    writer.writerow([
        "prompt_id",
        "llm",
        "prompt_text",
        "response_text",
        "citation_count",
        "citation_urls",
        "error",
    ])
    for r in (data.get("all_responses") or []):
        prompt_text = r.get("prompt") or ""
        cits = r.get("citations") or []
        urls = [c.get("url") for c in cits if isinstance(c, dict) and c.get("url")]
        writer.writerow([
            prompt_id_lookup.get(prompt_text, ""),
            r.get("llm") or "",
            prompt_text,
            r.get("response") or "",
            len(urls),
            _join_urls(urls),
            r.get("error") or "",
        ])

    # --- 4. ALL CITATION URLS (every URL extracted, verified) ---
    _section("ALL CITATION URLS (post-verification)")
    writer.writerow([
        "domain",
        "url",
        "llm",
        "prompt_id",
        "prompt_text",
    ])
    for r in (data.get("all_responses") or []):
        ptext = r.get("prompt") or ""
        pid = prompt_id_lookup.get(ptext, "")
        llm = r.get("llm") or ""
        for c in (r.get("citations") or []):
            if not isinstance(c, dict):
                continue
            writer.writerow([
                c.get("domain") or "",
                c.get("url") or "",
                llm,
                pid,
                ptext,
            ])

    # --- 5. SHARE OF VOICE ACROSS TOP EDITORIAL MEDIA ---
    sov_rows = data.get("outlet_sov") or []
    if sov_rows:
        _section("SHARE OF VOICE ACROSS TOP EDITORIAL MEDIA (strengths to defend, opportunities to pitch)")
        writer.writerow([
            "verdict",
            "domain",
            "responses_citing_outlet",
            "brand_mentions_at_outlet",
            "brand_sov_at_outlet_pct",
            "brand_overall_sov_pct",
            "brand_sov_differential_pct",
            "top_competitor_at_outlet",
            "top_competitor_sov_at_outlet_pct",
            "verdict_label",
        ])
        for row in sov_rows:
            tc = row.get("top_competitor_at_outlet") or {}
            writer.writerow([
                row.get("verdict") or "",
                row.get("domain") or "",
                row.get("responses_citing") or 0,
                row.get("brand_mentions_at_outlet") or 0,
                round((row.get("brand_sov_at_outlet") or 0) * 100, 1),
                round((row.get("brand_overall_sov") or 0) * 100, 1),
                round((row.get("brand_sov_differential") or 0) * 100, 1),
                tc.get("name") or "",
                round((tc.get("sov_at_outlet") or 0) * 100, 1) if tc else "",
                row.get("verdict_label") or "",
            ])

    # --- 6. EDITORIAL MEDIA TARGETS ---
    _section("EDITORIAL MEDIA TARGETS")
    writer.writerow([
        "rank",
        "outlet",
        "domain",
        "reporter",
        "citation_frequency",
        "cited_by_llms",
        "competitors_citing",
        "rationale",
        "gap_insight",
        "sample_urls",
    ])
    for t in (data.get("media_targets") or []):
        writer.writerow([
            t.get("rank") or "",
            t.get("outlet") or "",
            t.get("domain") or "",
            t.get("reporter") or "",
            t.get("citation_frequency") or "",
            _join(t.get("cited_by_llms")),
            _join(t.get("competitors_citing")),
            t.get("rationale") or "",
            t.get("gap_insight") or "",
            _join_urls(t.get("sample_urls")),
        ])

    # --- 7. INSTITUTIONAL TARGETS ---
    institutional_targets = data.get("institutional_targets") or []
    if institutional_targets:
        _section("INSTITUTIONAL / PARTNERSHIP TARGETS")
        writer.writerow([
            "rank",
            "institution",
            "domain",
            "type",
            "citation_frequency",
            "cited_by_llms",
            "partnership_play",
            "sample_urls",
        ])
        for t in institutional_targets:
            writer.writerow([
                t.get("rank") or "",
                t.get("institution") or "",
                t.get("domain") or "",
                t.get("type") or "",
                t.get("citation_frequency") or "",
                _join(t.get("cited_by_llms")),
                t.get("partnership_play") or "",
                _join_urls(t.get("sample_urls")),
            ])

    # --- 8. ANALYST TARGETS ---
    analyst_targets = data.get("analyst_targets") or []
    if analyst_targets:
        _section("ANALYST TARGETS")
        writer.writerow([
            "rank",
            "firm",
            "domain",
            "type",
            "citation_frequency",
            "cited_by_llms",
            "evaluations_referenced",
            "analyst_play",
            "sample_urls",
        ])
        for t in analyst_targets:
            writer.writerow([
                t.get("rank") or "",
                t.get("firm") or "",
                t.get("domain") or "",
                t.get("type") or "",
                t.get("citation_frequency") or "",
                _join(t.get("cited_by_llms")),
                _join(t.get("evaluations_referenced")),
                t.get("analyst_play") or "",
                _join_urls(t.get("sample_urls")),
            ])

    csv_bytes = buf.getvalue().encode('utf-8-sig')  # BOM so Excel reads UTF-8 cleanly
    brand_slug = re.sub(r'[^a-z0-9]+', '-', (data.get('brand') or 'report').lower()).strip('-') or 'report'
    filename = f"signal-finder-{brand_slug}-{slug}.csv"
    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.route('/signal/<slug>.json')
def signal_report_json(slug):
    """Return the full saved payload as a JSON download — includes the analyzed
    target lists AND the raw per-response data (`all_responses`) and the
    complete ranked domain list. Lets consultants re-analyze, audit, or
    feed the data into other tools.
    """
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))

    brand_slug = re.sub(r'[^a-z0-9]+', '-', (data.get('brand') or 'report').lower()).strip('-') or 'report'
    filename = f"signal-finder-{brand_slug}-{slug}.json"
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


@app.route('/citation-audit/request-demo', methods=['POST'])
def citation_audit_request_demo():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    title = (data.get('title') or '').strip()
    org = (data.get('org') or '').strip()
    email = (data.get('email') or '').strip()
    slug = (data.get('slug') or '').strip() or None
    problem_statement = (data.get('problem_statement') or '').strip()
    # 'csv' when triggered by Request CSV button; 'bespoke' otherwise.
    # Controls email subject + body so the operator can triage CSV-export
    # requests separately from full-engagement leads.
    request_type = (data.get('request_type') or 'bespoke').strip().lower()
    if request_type not in ('csv', 'bespoke'):
        request_type = 'bespoke'

    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if '@' not in email or '.' not in email:
        return jsonify({"error": "Please enter a valid email address."}), 400

    extra = json.dumps({"name": name, "title": title, "org": org,
                        "problem_statement": problem_statement,
                        "request_type": request_type})
    lead = LeadCapture(email=email, slug=slug, app_name='signal_finder_demo', extra=extra)
    db.session.add(lead)
    db.session.commit()

    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if sg_key:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            # Use the configured signal base URL (e.g. https://signal.innatec3.com)
            # so the link points at the actual /signal/<slug> route. The old
            # https://innatec3.com/results/{slug} URL 404s — that route doesn't
            # exist on innatec3.com.
            base = os.environ.get("SIGNAL_BASE_URL") or (request.url_root.rstrip('/'))
            report_link = f"{base}/signal/{slug}" if slug else "(no audit slug)"
            # CSV-request leads get their own subject + framing so they're
            # easy to triage separately. Both types still capture all the same
            # contact fields + audit context + report link.
            if request_type == 'csv':
                lead_label = "CSV export request"
                subject_prefix = "[PR Signal Finder · CSV REQUEST]"
                lead_note = (
                    "ACTION: Send the CSV export for this user's audit "
                    f"({slug or 'no slug'}). Includes all prompts, full LLM "
                    "responses (Claude / ChatGPT / Gemini), and every citation "
                    "URL extracted."
                )
            else:
                lead_label = "bespoke audit request"
                subject_prefix = "[PR Signal Finder]"
                lead_note = ""

            text_body = (
                f"New PR Signal Finder {lead_label}:\n\n"
                f"Name: {name}\n"
                f"Title: {title or '(not provided)'}\n"
                f"Organization: {org or '(not provided)'}\n"
                f"Email: {email}\n\n"
                f"Problem statement they audited:\n{problem_statement or '(not provided)'}\n\n"
                f"Their light audit report: {report_link}\n"
            )
            if lead_note:
                text_body += f"\n{lead_note}\n"

            report_link_html = (
                f'<a href="{report_link}">{report_link}</a>' if slug else '<em>(no slug captured)</em>'
            )
            email_link_html = f'<a href="mailto:{html.escape(email)}">{html.escape(email)}</a>'
            html_body = (
                f"<h3>New PR Signal Finder {lead_label}</h3>"
                f"<p><strong>Name:</strong> {html.escape(name)}<br>"
                f"<strong>Title:</strong> {html.escape(title) or '<em>(not provided)</em>'}<br>"
                f"<strong>Organization:</strong> {html.escape(org) or '<em>(not provided)</em>'}<br>"
                f"<strong>Email:</strong> {email_link_html}</p>"
                f"<p><strong>Problem they audited:</strong><br>{html.escape(problem_statement) or '<em>(not provided)</em>'}</p>"
                f"<p><strong>Their light audit report:</strong> {report_link_html}</p>"
            )
            if lead_note:
                if slug:
                    csv_link = f"{base}/signal/{slug}.csv"
                    csv_link_html = f'<a href="{csv_link}">{csv_link}</a>'
                else:
                    csv_link_html = '<em>(no slug captured)</em>'
                html_body += (
                    f'<p style="background:#fffae0;padding:10px 14px;border-left:3px solid #d4a500;'
                    f'border-radius:4px;margin-top:14px"><strong>Action:</strong> {html.escape(lead_note)}<br>'
                    f'<strong>CSV download URL (operator-only):</strong> {csv_link_html}</p>'
                )

            subject_suffix = f" from {name}" + (f" ({org})" if org else "")
            msg = Mail(
                from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
                to_emails=["nstrauss@innatec3.com"],
                subject=f"{subject_prefix} {lead_label}{subject_suffix}",
                plain_text_content=text_body,
                html_content=html_body,
            )
            _send_mail_object(msg)
        except Exception as e:
            print("SendGrid lead notification error:", e)

    return jsonify({"ok": True, "dev": not bool(sg_key)})


@app.route('/signal/feedback', methods=['POST'])
def signal_feedback():
    """MVP branch: thumbs-up / thumbs-down widget at the bottom of each audit.

    Stored in the AuditFeedback table; also fire-and-forget to Nathan via
    SendGrid so he sees the signal in real time.
    """
    data = request.get_json(silent=True) or {}
    slug = (data.get('slug') or '').strip()
    rating = (data.get('rating') or '').strip().lower()
    if not slug or rating not in ('up', 'down'):
        return jsonify({"error": "bad request"}), 400
    fb = AuditFeedback(slug=slug, rating=rating, ip=_client_ip())
    db.session.add(fb)
    db.session.commit()
    try:
        sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
        if sg_key:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            emoji = '\U0001F44D' if rating == 'up' else '\U0001F44E'
            link = (SIGNAL_BASE_URL or 'https://signal.innatec3.com') + url_for('view_signal_report', slug=slug)
            msg = Mail(
                from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
                to_emails=["nstrauss@innatec3.com"],
                subject=f"[Signal Finder feedback] {emoji} — {slug}",
                plain_text_content=f"User rated audit {slug} as {rating}.\n\nReport: {link}",
                html_content=f"<p>User rated audit <strong>{slug}</strong> as {emoji} <strong>{rating}</strong>.</p><p>Report: <a href=\"{link}\">{link}</a></p>",
            )
            _send_mail_object(msg)
    except Exception as e:
        print("Feedback email failed:", e)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# PR Signal Finder — paid-tier auth, billing, dashboard
# ---------------------------------------------------------------------------

def _hash_token(raw):
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def current_signal_user():
    uid = session.get('signal_user_id')
    if not uid:
        return None
    return SignalUser.query.get(uid)


def current_user_credits(user):
    if not user:
        return 0
    bal = CreditBalance.query.filter_by(user_id=user.id).first()
    return (bal.credits_remaining if bal else 0)


def signal_login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('signal_user_id')
        if not uid:
            return redirect(url_for('signal_login', next=request.path))
        # Defensive: the session cookie can outlive the underlying user row
        # (Render Starter's disk can reset across deploys, wiping SQLite). Without
        # this check, signal_login redirects to signal_dashboard, dashboard finds
        # no user and redirects back to login → infinite loop. Clear stale sessions.
        if not SignalUser.query.get(uid):
            session.pop('signal_user_id', None)
            session.pop('signal_post_login_next', None)
            return redirect(url_for('signal_login', next=request.path))
        return view(*args, **kwargs)
    return wrapper


def _signal_base_url():
    return (SIGNAL_BASE_URL or request.url_root.rstrip('/'))


def _send_magic_link_email(email, raw_token):
    base = _signal_base_url()
    link = f"{base}{url_for('signal_auth', token=raw_token)}"
    sg_key = (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY"))
    if not sg_key:
        print(f"[DEV] Magic-link for {email}: {link}")
        return link
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        msg = Mail(
            from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
            to_emails=[email],
            subject="Your PR Signal Finder sign-in link",
            plain_text_content=(
                "Click the link below to sign in to PR Signal Finder. "
                "Link expires in 20 minutes.\n\n"
                f"{link}\n\nIf you didn't request this, you can ignore this email."
            ),
            html_content=(
                f'<p>Click below to sign in to PR Signal Finder. Link expires in 20 minutes.</p>'
                f'<p><a href="{link}">{link}</a></p>'
                f'<p style="color:#888;font-size:13px">If you didn\'t request this, ignore this email.</p>'
            ),
        )
        _send_mail_object(msg)
    except Exception as e:
        print("SendGrid magic-link error:", e)
        print(f"[FALLBACK] Magic-link for {email}: {link}")
    return link


@app.route('/signal/login', methods=['GET', 'POST'])
def signal_login():
    if session.get('signal_user_id'):
        # Only redirect to dashboard if the session points at an actual user.
        # Otherwise wipe the stale cookie and fall through to render the login form
        # — avoids the login↔dashboard redirect loop when the user row has been
        # deleted/lost (e.g. Render disk reset between deploys).
        if SignalUser.query.get(session['signal_user_id']):
            return redirect(url_for('signal_dashboard'))
        session.pop('signal_user_id', None)
        session.pop('signal_post_login_next', None)
    nxt = request.values.get('next') or url_for('signal_dashboard')
    if request.method == 'GET':
        return render_template(
            'signal_login.html',
            ga_measurement_id=GA_MEASUREMENT_ID,
            sent=False,
            email='',
            next_url=nxt,
            dev_magic_link=None,
        )

    email = (request.form.get('email') or '').strip().lower()
    if not email or '@' not in email or '.' not in email:
        flash("Please enter a valid email address.")
        return render_template('signal_login.html', ga_measurement_id=GA_MEASUREMENT_ID, sent=False, email=email, next_url=nxt, dev_magic_link=None)

    # Rate-limit by both email and requester IP. Either being over the limit
    # blocks the attempt; this stops both single-target spamming (one email
    # from many IPs) and broad fishing (many emails from one IP).
    requester_ip = (request.headers.get('X-Forwarded-For', request.remote_addr or '')
                    .split(',')[0].strip() or 'unknown')
    if not _check_login_rate(f"email:{email}") or not _check_login_rate(f"ip:{requester_ip}"):
        flash("Too many sign-in attempts. Try again in 10 minutes.")
        return render_template(
            'signal_login.html',
            ga_measurement_id=GA_MEASUREMENT_ID,
            sent=False,
            email=email,
            next_url=nxt,
            dev_magic_link=None,
        )

    # Invalidate any prior unused tokens for this email before issuing a new one.
    # Without this, a stolen-but-uninstalled-yet old link would still work once
    # the user requests a fresh link. We also limit blast radius if a previously
    # sent email was forwarded.
    LoginToken.query.filter_by(email=email, used_at=None).update(
        {'used_at': datetime.utcnow()}, synchronize_session=False
    )

    raw = secrets.token_urlsafe(32)
    db.session.add(LoginToken(
        email=email,
        token_hash=_hash_token(raw),
        expires_at=datetime.utcnow() + timedelta(minutes=20),
    ))
    db.session.commit()
    session['signal_post_login_next'] = nxt
    link = _send_magic_link_email(email, raw)
    # In dev (no SendGrid), expose the link inline so the user can sign in without email delivery.
    dev_link = link if not (os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY")) else None
    return render_template(
        'signal_login.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        sent=True,
        email=email,
        next_url=nxt,
        dev_magic_link=dev_link,
    )


@app.route('/signal/auth')
def signal_auth():
    raw = request.args.get('token', '')
    if not raw:
        flash("Invalid sign-in link.")
        return redirect(url_for('signal_login'))
    token_hash = _hash_token(raw)
    rec = LoginToken.query.filter_by(token_hash=token_hash).first()
    if not rec or rec.expires_at < datetime.utcnow():
        flash("Sign-in link expired or already used. Request a new one.")
        return redirect(url_for('signal_login'))

    # Atomic consume: only flip used_at if it's still NULL. If two requests
    # race (e.g. browser pre-fetch + user click), only one wins (rowcount==1),
    # the other is rejected. Without this, a token could be consumed twice
    # in concert with read-modify-write timing.
    now = datetime.utcnow()
    consumed = db.session.execute(
        db.update(LoginToken)
          .where(LoginToken.token_hash == token_hash)
          .where(LoginToken.used_at.is_(None))
          .values(used_at=now)
    )
    db.session.commit()
    if consumed.rowcount != 1:
        flash("Sign-in link expired or already used. Request a new one.")
        return redirect(url_for('signal_login'))

    user = SignalUser.query.filter_by(email=rec.email).first()
    if not user:
        user = SignalUser(email=rec.email)
        db.session.add(user)
        db.session.flush()
        db.session.add(CreditBalance(user_id=user.id, credits_remaining=0))
    user.last_login_at = now
    db.session.commit()
    session['signal_user_id'] = user.id
    nxt = session.pop('signal_post_login_next', None) or url_for('signal_dashboard')
    return redirect(nxt)


@app.route('/signal/logout', methods=['POST'])
def signal_logout():
    session.pop('signal_user_id', None)
    session.pop('signal_post_login_next', None)
    return redirect(url_for('citation_audit'))


@app.route('/signal/dashboard')
def signal_dashboard():
    # MVP branch: dashboard is unreachable from the UI. Old bookmarks land on
    # the audit form instead. The original implementation lives below as
    # `_legacy_signal_dashboard` and on the `feat/pr-signal-finder` branch.
    return redirect(url_for('citation_audit'))


def _legacy_signal_dashboard():
    """Original paid-tier dashboard. Preserved for reference; not routed."""
    user = current_signal_user()
    if not user:
        return redirect(url_for('signal_login'))
    credits = current_user_credits(user)
    runs = (AuditRun.query
            .filter_by(user_id=user.id)
            .order_by(AuditRun.created_at.desc())
            .limit(20).all())
    purchases = (Purchase.query
                 .filter_by(user_id=user.id)
                 .order_by(Purchase.created_at.desc())
                 .limit(10).all())
    products = []
    for key, cfg in STRIPE_PRODUCTS.items():
        products.append({
            "key": key,
            "label": cfg["label"],
            "credits": cfg["credits"],
            "amount_display": cfg["amount_display"],
            "description": cfg["description"],
            "configured": bool(cfg["price_id"]),
        })
    return render_template(
        'signal_dashboard.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        user=user,
        credits=credits,
        runs=runs,
        purchases=purchases,
        products=products,
        stripe_configured=bool(stripe_lib and STRIPE_SECRET_KEY),
    )


def _grant_credits_from_stripe_session(sess_obj):
    """Idempotently grant credits + find-or-create the user from a Stripe Checkout Session.

    Returns (user, purchase) tuple, or (None, None) if the session is not a paid credit purchase.
    Safe to call multiple times for the same session — dedupes via Purchase.stripe_session_id.
    """
    sess_id = sess_obj.get('id')
    if not sess_id:
        return None, None
    payment_status = sess_obj.get('payment_status')
    if payment_status and payment_status != 'paid':
        return None, None

    existing = Purchase.query.filter_by(stripe_session_id=sess_id).first()
    if existing:
        user = SignalUser.query.get(existing.user_id) if existing.user_id else None
        return user, existing

    md = sess_obj.get('metadata') or {}
    try:
        credits = int(md.get('credits') or 0)
    except (TypeError, ValueError):
        credits = 0
    if credits <= 0:
        return None, None

    cust_details = sess_obj.get('customer_details') or {}
    email = (
        sess_obj.get('customer_email')
        or cust_details.get('email')
        or md.get('email')
        or ''
    ).strip().lower()
    if not email:
        print("Stripe session has no email; cannot grant credits:", sess_id)
        return None, None

    user = SignalUser.query.filter_by(email=email).first()
    new_user = False
    if not user:
        user = SignalUser(email=email)
        db.session.add(user)
        db.session.flush()
        db.session.add(CreditBalance(user_id=user.id, credits_remaining=0))
        new_user = True

    bal = CreditBalance.query.filter_by(user_id=user.id).first()
    if not bal:
        bal = CreditBalance(user_id=user.id, credits_remaining=0)
        db.session.add(bal)
    bal.credits_remaining = (bal.credits_remaining or 0) + credits

    purchase = Purchase(
        user_id=user.id,
        stripe_session_id=sess_id,
        amount_cents=sess_obj.get('amount_total') or 0,
        credits_granted=credits,
        product_label=md.get('product') or '',
    )
    db.session.add(purchase)
    db.session.commit()

    # Best-effort: email a magic-link so the buyer has a path back if they lose the tab.
    if new_user:
        try:
            raw = secrets.token_urlsafe(32)
            db.session.add(LoginToken(
                email=email,
                token_hash=_hash_token(raw),
                expires_at=datetime.utcnow() + timedelta(hours=24),
            ))
            db.session.commit()
            _send_magic_link_email(email, raw)
        except Exception as e:
            print("Welcome magic-link send failed:", e)

    return user, purchase


@app.route('/signal/checkout/<product>', methods=['POST'])
def signal_checkout(product):
    cfg = STRIPE_PRODUCTS.get(product)
    if not cfg:
        return jsonify({"error": "Unknown product."}), 400
    if not stripe_lib or not STRIPE_SECRET_KEY:
        return jsonify({"error": "Payments are not configured."}), 503
    if not cfg["price_id"]:
        return jsonify({"error": f"Price ID for {product} not configured."}), 503

    body = request.get_json(silent=True) or {}
    problem_statement = (body.get('problem_statement') or '').strip()[:1000]

    user = current_signal_user()
    base = _signal_base_url()
    md = {"product": product, "credits": str(cfg["credits"])}
    if user:
        md["user_id"] = str(user.id)
    if problem_statement:
        md["problem_statement"] = problem_statement

    session_kwargs = {
        "mode": "payment",
        "line_items": [{"price": cfg["price_id"], "quantity": 1}],
        "success_url": f"{base}{url_for('signal_checkout_success')}?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base}{url_for('citation_audit')}",
        "metadata": md,
        "allow_promotion_codes": True,
        "billing_address_collection": "auto",
    }
    if user:
        session_kwargs["customer_email"] = user.email
    # For guests, Stripe Checkout collects the email itself.

    try:
        sess = stripe_lib.checkout.Session.create(**session_kwargs)
        return jsonify({"url": sess.url})
    except Exception as e:
        print("Stripe checkout error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/signal/checkout/success')
def signal_checkout_success():
    sid = (request.args.get('session_id') or '').strip()
    user = current_signal_user()  # truthy ONLY if the buyer was already signed in
    purchase = None
    problem_statement = None
    granted_user = None
    granted_email = None

    if sid and stripe_lib and STRIPE_SECRET_KEY:
        try:
            sess_obj = stripe_lib.checkout.Session.retrieve(sid)
            try:
                # stripe-python returns a StripeObject; cast to dict for our helper
                sess_dict = sess_obj.to_dict() if hasattr(sess_obj, 'to_dict') else dict(sess_obj)
            except Exception:
                sess_dict = sess_obj
            problem_statement = ((sess_dict.get('metadata') or {}).get('problem_statement') or '').strip() or None
            granted_user, granted_purchase = _grant_credits_from_stripe_session(sess_dict)
            if granted_purchase:
                purchase = granted_purchase
            if granted_user:
                granted_email = granted_user.email
        except Exception as e:
            print("Stripe success-page processing error:", e)

    # SECURITY: do NOT auto-login the request based on possession of session_id.
    # The Stripe session_id appears in the URL and would leak via Referer,
    # browser history, sharing, etc. Previously this route did
    # `session['signal_user_id'] = granted_user.id`, which silently logged in
    # anyone who held the URL. Now: if the buyer was already signed in, keep
    # them; otherwise show a "credits granted, sign in via magic-link" message.
    # The welcome magic-link is already dispatched by
    # _grant_credits_from_stripe_session for new users (see ~line 4015).
    credits = current_user_credits(user) if user else (
        current_user_credits(granted_user) if granted_user else 0
    )
    return render_template(
        'signal_checkout_success.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        user=user,
        credits=credits,
        purchase=purchase,
        problem_statement=problem_statement,
        granted_email=granted_email,
    )


@app.route('/signal/stripe-webhook', methods=['POST'])
def signal_stripe_webhook():
    if not stripe_lib:
        return jsonify({"error": "Stripe SDK not installed"}), 503
    if not STRIPE_WEBHOOK_SECRET:
        return jsonify({"error": "Webhook secret not configured"}), 503
    payload = request.data
    sig = request.headers.get('Stripe-Signature', '')
    try:
        event = stripe_lib.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print("Stripe webhook verification failed:", e)
        return jsonify({"error": "Invalid signature"}), 400

    if event.get('type') == 'checkout.session.completed':
        sess = event['data']['object']
        try:
            sess_dict = sess.to_dict() if hasattr(sess, 'to_dict') else dict(sess)
        except Exception:
            sess_dict = sess
        try:
            _grant_credits_from_stripe_session(sess_dict)
        except Exception as e:
            print("Stripe webhook credit-grant failed:", e)
            return jsonify({"error": "processing failed"}), 500
    return jsonify({"ok": True})


def _ensure_homepage_link():
    """Idempotently mint the trackable homepage link (/r/try -> homepage), so
    the operator can share a click-logged CTA instead of the bare URL."""
    try:
        with app.app_context():
            if not TrackedLink.query.filter_by(token='try').first():
                _mint_tracked_link('__home__', 'Homepage CTA', 'homepage', token='try')
    except Exception as e:
        print("homepage-link ensure failed:", e)


# Apply post-creation column migrations, then ensure the seed prospects exist.
_ensure_outreach_columns()
_ensure_inbound_columns()
_backfill_inbound_operator_flags()
_ensure_outreach_seed()
_ensure_homepage_link()


if __name__ == "__main__":
    # Get port from environment variable or default to 5009
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
