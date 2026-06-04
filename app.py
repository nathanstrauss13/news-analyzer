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
                model="claude-3-haiku-20240307",
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
                model="claude-3-haiku-20240307",
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
    return redirect("/citation-audit", code=308)


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

        sg_key = os.environ.get("SENDGRID_API_KEY")
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
            sg = SendGridAPIClient(sg_key)
            resp = sg.send(msg)
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
        sg_key = os.environ.get("SENDGRID_API_KEY")
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
        sg = SendGridAPIClient(sg_key)
        resp = sg.send(msg)
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

# Start scheduler (guard against double-start under reloader)
if BackgroundScheduler:
    try:
        if not getattr(app, "_alerts_scheduler_started", False):
            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(run_realtime_alerts, 'interval', minutes=10)
            scheduler.add_job(run_daily_alerts, 'cron', hour=8, minute=0)
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
    'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'who.int', 'cdc.gov',
    'arxiv.org', 'researchgate.net', 'sciencedirect.com', 'springer.com',
    'jstor.org', 'nature.com', 'science.org',
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


def _registrable_label_candidates(segs):
    """Given a domain's alnum dot-segments, return the labels that could be the
    'registrable' name (the bit before the public-suffix-ish TLD). Handles both
    foo.com (label 'foo') and foo.co.uk (label 'foo', not 'co')."""
    cands = set()
    if len(segs) >= 2:
        cands.add(segs[-2])
    if len(segs) >= 3:
        cands.add(segs[-3])  # foo.co.uk → also consider 3rd-to-last
    if segs:
        cands.add(segs[0])
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
    brand_slug = re.sub(r'[^a-z0-9]', '', brand.lower())
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
            model="claude-sonnet-4-20250514",
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
        u = u.rstrip('.,;:!?\'"\)\]\}')
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


def _count_brand_mentions(brand, all_responses):
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

    if len(parts) > 1 and len(parts[0]) > 5:
        first_pat = re.compile(r'\b' + re.escape(parts[0]) + r'\b', re.IGNORECASE)
        first_count = sum(
            1 for r in all_responses
            if first_pat.search(r.get('response', '') or '')
        )
        # Guardrail: accept the fallback only when the first word doesn't
        # dominate. Compare to max(2×full, 3) so very small full-counts still
        # leave room for legitimate fallback (e.g. full=0 but first=2 is OK;
        # full=0 but first=29 is the Beauty bug).
        if first_count <= max(full_count * 2, 3):
            combined = sum(
                1 for r in all_responses
                if full_pat.search(r.get('response', '') or '') or
                   first_pat.search(r.get('response', '') or '')
            )
            return combined
    return full_count


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
            model="claude-sonnet-4-20250514",
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


def _compute_outlet_share_of_voice(brand, competitor_counts, all_responses, editorial_domains, max_outlets=10):
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
    def _patterns_for(name):
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

        if not (len(parts) > 1 and len(parts[0]) > 5):
            return [full_pat]
        first_pat = re.compile(r'\b' + re.escape(parts[0]) + r'\b', re.IGNORECASE)
        # Reject fallback if first word dominates (category-word false positive).
        full_count = sum(
            1 for r in all_responses
            if full_pat.search(r.get('response', '') or '')
        )
        first_count = sum(
            1 for r in all_responses
            if first_pat.search(r.get('response', '') or '')
        )
        if first_count > max(full_count * 2, 3):
            return [full_pat]
        return [full_pat, first_pat]

    brand_patterns = _patterns_for(brand or '')
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
        DOMINANT_BRAND_THRESHOLD = 0.70  # overall SoV at which we switch modes
        DOMINANT_STRENGTH_FLOOR = 0.95   # in dominant mode, strength = brand at >= 95%
        DOMINANT_OPPORTUNITY_DELTA = 0.10  # in dominant mode, opportunity = brand drops 10pp below baseline
        # The 10pp opportunity floor for dominant brands is intentionally looser
        # than the 15pp delta for normal-mode brands. At a 97% baseline, even a
        # 10pp slip (e.g. 83% at an outlet) is signal — the brand is essentially
        # ubiquitous everywhere except a few specific places, and those places
        # are where targeted earned media moves the needle the most.
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
            if brand_patterns and brand_diff >= STRENGTH_PP and (not leader_comp or brand_sov_at >= leader_comp["sov_at_outlet"]):
                verdict = "strength"
                verdict_label = (
                    f"{brand} over-indexes here. Defend the relationship; "
                    f"pitch follow-up coverage."
                )
            elif leader_comp:
                comp_lead = leader_comp["sov_at_outlet"] - brand_sov_at
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


def _compute_headline_move(brand, outlet_sov):
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
    """
    sov = outlet_sov or []
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

    # 1. Best opportunity. Rank by CITATION VOLUME first (a gap at a 6×-cited
    # outlet is a far stronger signal than a 100%-vs-0% at a 2×-cited one),
    # then by the raw mention gap. Phrase with concrete counts, not just %, so
    # the claim is honest at small sample sizes.
    if opportunities:
        def _opp_key(r):
            tc = r.get('top_competitor_at_outlet') or {}
            gap = (tc.get('mentions_at_outlet') or 0) - _brand_at(r)
            # Prefer a genuine competitor lead (gap > 0), then citation volume,
            # then gap size — so the headline isn't a zero-gap "opportunity".
            return (1 if gap > 0 else 0, _n(r), gap)
        r = max(opportunities, key=_opp_key)
        tc = r.get('top_competitor_at_outlet') or {}
        dom, n, ba = r.get('domain'), _n(r), _brand_at(r)
        cm = tc.get('mentions_at_outlet') or 0
        if tc.get('name') and cm > ba:
            text = (f"Pitch {dom}. {tc['name']} shows up in {cm} of the {n} AI responses "
                    f"citing it, {b} in {ba} — closing this gap is your highest-leverage "
                    f"earned-media move.")
        else:
            text = (f"Pitch {dom}. {b} appears in only {ba} of the {n} AI responses citing it, "
                    f"below its overall visibility — the clearest place to grow.")
        return {"verb": "Pitch", "outlet": dom, "text": text}

    # 2/3. No opportunities — defend. Prefer the most-CITED strength where a
    # competitor also appears (most contested + most important), else the
    # most-cited uncontested strength.
    if strengths:
        contested = [r for r in strengths if (r.get('top_competitor_at_outlet') or {}).get('name')]
        if contested:
            r = max(contested, key=lambda x: (_n(x), (x.get('top_competitor_at_outlet') or {}).get('mentions_at_outlet') or 0))
            tc = r.get('top_competitor_at_outlet') or {}
            dom, n, ba = r.get('domain'), _n(r), _brand_at(r)
            cm = tc.get('mentions_at_outlet') or 0
            text = (f"Defend {dom}. {b} leads there ({ba} of {n} responses), but {tc['name']} "
                    f"is also present ({cm}) — protect this relationship first so the lead holds.")
            return {"verb": "Defend", "outlet": dom, "text": text}
        r = max(strengths, key=lambda x: (_n(x), _brand_at(x)))
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


def _compute_per_llm_visibility(brand, all_responses):
    """Per-assistant brand visibility: for each LLM, how many of ITS responses
    mention the brand. A headline "20% mindshare" can hide the real story —
    e.g. a brand cited in 9/10 Gemini responses but 0/10 on ChatGPT/Claude/Grok
    is concentrated in one (search-grounded) assistant, not broadly embedded.

    Returns a list of {llm, mentions, total, rate, grounded} sorted by rate
    desc. Reuses _count_brand_mentions per-LLM so the counting matches the
    headline number exactly.
    """
    if not brand or not all_responses:
        return []
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
        m = _count_brand_mentions(brand, subset)
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
            # "All 5 surfacing" doesn't always mean balanced — Lululemon hit all 5
            # but ranged 1-5 (ChatGPT 1 vs Grok 5). Calling that "consistent" is
            # misleading. Only claim consistency when the spread is genuinely
            # tight; otherwise name the strongest + weakest honestly.
            mentions = [x.get('mentions') or 0 for x in per_llm]
            mn, mx = min(mentions), max(mentions)
            is_balanced = mn >= 2 and mx <= mn * 3
            if is_balanced:
                return (f"{b} surfaces across all {total} assistants in live-search mode — strong, "
                        f"consistent discoverability when AI searches the web for your category.")
            leader = max(per_llm, key=lambda x: x.get('mentions') or 0)
            weakest = min(per_llm, key=lambda x: x.get('mentions') or 0)
            return (f"{b} surfaces across all {total} assistants but unevenly — strongest on "
                    f"{leader.get('llm')} ({leader.get('mentions')}/{leader.get('total')}), "
                    f"thinnest on {weakest.get('llm')} ({weakest.get('mentions')}/{weakest.get('total')}). "
                    f"Broad reach but uneven recommendation depth.")
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
        return f"{b} surfaces across all {total} assistants — broad, embedded AI visibility."
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
            model="claude-sonnet-4-20250514", max_tokens=1400, timeout=45.0,
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


def _page_brand_mentions(url, brand, timeout=5):
    """Fetch a URL and count how many times `brand` appears in its title+body
    (word-boundary, case-insensitive, with the multi-word first-word fallback).
    Returns the count, or -1 if the page couldn't be fetched/parsed.

    This is what separates CONFIRMED COVERAGE (the brand is actually on the
    cited page) from a CATEGORY OUTLET (the AI cited the outlet for the topic,
    but the page is about a competitor or generic trends). The existing
    topic-verify pass returns 'verified' on brand OR category-keyword match, so
    a Rare-Beauty packaging article passes as "on-topic" without covering the
    brand at all — this check closes that gap.
    """
    try:
        if not _is_safe_url(url):
            return -1
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                         headers=_TOPIC_CHECK_HEADERS, stream=False)
        if r.status_code >= 400:
            return -1
        raw = r.text or ''
        if not raw.strip():
            return -1
        title_m = re.search(r'<title[^>]*>([^<]+)</title>', raw, re.IGNORECASE)
        text = (title_m.group(1) if title_m else '') + ' ' + re.sub(r'<[^>]+>', ' ', raw[:200000])
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
        if row.get('verdict') == 'strength' and cov and cov != 'confirmed':
            row['_pre_guard_verdict'] = 'strength'  # diagnostic
            row['verdict'] = 'opportunity'
            tc = row.get('top_competitor_at_outlet') or {}
            comp = tc.get('name') if isinstance(tc, dict) else None
            if comp:
                row['verdict_label'] = (
                    f"AI co-mentions {b} alongside {comp} when it cites this "
                    f"outlet, but the cited page doesn't actually cover {b}. "
                    f"Pitch to convert co-mention into coverage."
                )
            else:
                row['verdict_label'] = (
                    f"AI co-mentions {b} when it cites this outlet, but the "
                    f"cited page doesn't actually cover {b}. Pitch coverage."
                )
            downgraded.append(dom)
    if downgraded:
        print(f"_coverage_guard_verdicts: downgraded {len(downgraded)} 'strength' -> "
              f"'opportunity' (no on-page coverage): {', '.join(downgraded[:10])}")


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
        model="claude-sonnet-4-20250514",
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
            model="claude-sonnet-4-20250514",
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
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
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


def _send_audit_failure_email(error, problem_statement, tier):
    """Fire-and-forget diagnostic email when an audit fails (AuditAnalysisError
    or any other exception during run_citation_audit).

    Includes the raw Claude response (if available), the parse strategies
    attempted, and the problem statement. Skipped silently if SENDGRID_API_KEY
    or AUDIT_DEBUG_EMAIL is unset.
    """
    recipient = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = os.environ.get("SENDGRID_API_KEY")
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
        SendGridAPIClient(sg_key).send(msg)
    except Exception as e:
        print("Audit FAILURE-email send failed:", e)


def _send_audit_debug_email(result, duration_seconds, per_provider_done, per_provider_errors, tier, problem_statement=""):
    """Fire-and-forget summary email to the owner after each audit.

    Skipped silently if SENDGRID_API_KEY or AUDIT_DEBUG_EMAIL is unset.
    Never raises — all errors are caught and logged.
    """
    recipient = os.environ.get("AUDIT_DEBUG_EMAIL", "nstrauss@innatec3.com").strip()
    sg_key = os.environ.get("SENDGRID_API_KEY")
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
        SendGridAPIClient(sg_key).send(msg)
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
        model="claude-sonnet-4-20250514",
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
                model="claude-sonnet-4-20250514",
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
    brand_mention_count = _count_brand_mentions(brand, all_responses)
    # Per-assistant visibility — computed here (before the analysis prompt) so
    # the summary can call out lopsided concentration. (Re-stored on the
    # analysis dict below for the UI.)
    _per_llm_vis = _compute_per_llm_visibility(brand, all_responses)
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
    competitor_counts = competitor_counts[:10]
    top_competitor_block = "\n".join(
        f"  - {c['name']}: cited in {c['mention_count']} of {len(all_responses)} responses"
        for c in competitor_counts
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
    analysis_max_tokens = 8000 if tier == "paid" else 5000
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
  "executive_summary": "EXACTLY 3 sentences — this is the 'What we found' headline a comms director will paste into a CMO briefing. Each sentence must carry a specific, non-obvious finding tied to the actual numbers. Lead with the single most important insight, not a throat-clearing preamble. STRUCTURE:\n  Sentence 1 — THE POSITION: {brand}'s AI mindshare ({brand_mention_count} of {len(all_responses)} responses) framed against the top competitor's count. If {brand} >= top competitor, lead with strength ('{brand} leads/holds the AI conversation in [category]…'); if behind, lead with the gap. NEVER say 'lacks authority' if the brand out-mentions competitors.\n  Sentence 2 — THE SURPRISE: the single most non-obvious thing in the data. STRONGLY PREFER the PER-ASSISTANT VISIBILITY finding when it's lopsided — if the brand surfaces on only one or two of the five assistants and is absent from the rest, lead the surprise with that (e.g. 'almost all of {brand}'s visibility is Gemini; it's absent from ChatGPT, Claude, and Grok'), because it means the brand is search-surfaced but not embedded in the models people use most. Otherwise: a specific outlet where the brand is absent but a competitor owns it; analyst firms (Gartner/Forrester/etc.) dominating citations over editorial press; a competitor you'd expect to lead that doesn't. Name the specific entity + number.\n  Sentence 3 — THE MOVE: the one highest-leverage action this data points to. Tie it to a named outlet or a named competitive dynamic from the strengths/opportunities lists. Use action verbs: defend, displace, cultivate, pitch.\n  RULES: No filler ('this audit reveals…', 'in today's landscape…'). No generic 'competitors dominate' unless the per-outlet data supports it (empty competitors_citing = open whitespace, not a bloodbath). Discuss analysts ONLY if analyst_targets has entries OR analyst firms appear heavily in the citation data; silence is fine for consumer categories. Write it so a CMO who reads ONLY these 3 sentences still walks away with the strategic takeaway."
}}"""

    # Retry once with reduced output budget on APITimeoutError. The most common
    # cause of timeout is Claude generating until it hits max_tokens; trimming
    # the budget forces it to ship concise JSON inside the deadline.
    def _call_analysis(max_tok):
        return anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tok,
            timeout=analysis_call_timeout,
            messages=[{"role": "user", "content": _analysis_prompt_content}],
        )

    try:
        analysis_response = _call_analysis(analysis_max_tokens)
    except Exception as _e:
        _err_name = type(_e).__name__
        if 'Timeout' in _err_name or 'Connection' in _err_name:
            _reduced = max(4000, analysis_max_tokens // 2)
            print(f"[analysis] {_err_name} at max_tokens={analysis_max_tokens}; retrying with max_tokens={_reduced}")
            analysis_response = _call_analysis(_reduced)
        else:
            raise

    analysis_text = analysis_response.content[0].text
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
        analysis["outlet_sov"] = _compute_outlet_share_of_voice(
            brand, competitor_counts, all_responses, _sov_eds,
            max_outlets=max(10, len(_sov_eds))
        )
    except Exception as _sov_e:
        print("outlet share-of-voice computation failed (continuing without):", _sov_e)
        analysis["outlet_sov"] = []
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
    # Coverage-guard the SoV verdicts: an outlet can only be 'strength' if the
    # brand is on the cited page. Otherwise it's 'opportunity'. Keeps Media
    # Targets agreeing with Media Landscape so 'strength' honestly means 'they
    # cover you and you lead.'
    try:
        _coverage_guard_verdicts(analysis.get("media_targets"), analysis.get("outlet_sov"), brand)
    except Exception as _cg_e:
        print("coverage-guard failed (continuing without):", _cg_e)
    # Rank targets by how prominently the AI surfaces them (responses citing +
    # citation frequency) — relevance + share-of-voice, not coverage tiers.
    try:
        _sort_targets_by_prominence(analysis.get("media_targets"), analysis.get("outlet_sov"))
    except Exception as _st_e:
        print("target prominence sort failed (continuing):", _st_e)
    # Single highest-priority action — gives every dashboard one unmistakable
    # next step even when it's all-strength or all-emerging.
    try:
        analysis["headline_move"] = _compute_headline_move(brand, analysis.get("outlet_sov"))
    except Exception as _hm_e:
        print("headline-move computation failed (continuing without):", _hm_e)
        analysis["headline_move"] = None
    # Per-assistant visibility — which of the 5 AIs actually surface the brand.
    try:
        analysis["per_llm_visibility"] = _compute_per_llm_visibility(brand, all_responses)
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
    return analysis


# MVP branch: free-tier rate limit + client-IP helper.
FREE_DAILY_CAP = 10

# Comma-separated list of client IPs exempt from the per-day cap. Set on Render
# (Environment tab) when you want unlimited audits from a specific IP — your own
# residential / office IP, a co-worker, a demo machine. Look up the IP from any
# previous request in Render Logs (clientIP="..."). Whitespace + empty entries
# are ignored, so e.g. " 1.2.3.4 , 5.6.7.8" parses cleanly.
_FREE_AUDIT_BYPASS_IPS = set(
    ip.strip() for ip in (os.environ.get("FREE_AUDIT_BYPASS_IPS", "") or "").split(",")
    if ip.strip()
)


def _client_ip():
    """Return the best-effort client IP. Respects Render's X-Forwarded-For."""
    fwd = request.headers.get('X-Forwarded-For', '')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def _ip_is_exempt_from_cap(ip):
    """True if this IP should bypass FREE_DAILY_CAP enforcement (allowlisted
    via FREE_AUDIT_BYPASS_IPS env var)."""
    return bool(ip) and ip in _FREE_AUDIT_BYPASS_IPS


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


@app.route('/citation-audit', methods=['GET', 'POST'])
def citation_audit():
    # MVP branch: audits are always anonymous + free tier. We still pass
    # signal_user / signal_credits through to the template (the JS globals
    # reference them) but they're always falsy.
    user = current_signal_user()
    credits = current_user_credits(user)

    if request.method == 'GET':
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

    # Two-layer cap on free audits:
    #   1. Email gate (primary): required email, with a hard lifetime cap per
    #      email (EMAIL_AUDIT_CAP, default 1). This is the main control against
    #      runaway LLM spend on shared links — each audit costs real money.
    #      Also doubles as our lead capture (we know who's running audits and
    #      what brand they're investigating).
    #   2. Per-IP per-day cap (secondary): abuse protection on top of the email
    #      gate, so one browser can't burn through dozens of fake emails.
    # IPs in FREE_AUDIT_BYPASS_IPS skip BOTH gates — operator self-testing +
    # close-partner walkthroughs don't pollute the lead table.
    today = date.today()
    ip = _client_ip()
    lead_email = None
    if _ip_is_exempt_from_cap(ip):
        print(f"[audit] FREE_AUDIT_BYPASS_IPS hit: {ip} (unlimited)")
    else:
        # Email gate — captures the lead before any LLM call.
        lead_email = _normalize_email(request.form.get('email', ''))
        if not lead_email:
            return jsonify({
                "error": "Please enter a valid email so we can send you the report and any updates.",
                "code": "email_required",
            }), 400
        lead = AuditLead.query.filter_by(email=lead_email).first()
        if lead and lead.audit_count >= _EMAIL_AUDIT_CAP:
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
        # Per-IP per-day abuse cap.
        use = FreeAuditUse.query.filter_by(ip=ip, day=today).first()
        if use and use.count >= FREE_DAILY_CAP:
            return jsonify({
                "error": f"You've used your {FREE_DAILY_CAP} free audits for today. Talk to us about a bespoke audit for unlimited access.",
                "code": "rate_limited",
            }), 429
        if use:
            use.count += 1
        else:
            use = FreeAuditUse(ip=ip, day=today, count=1)
            db.session.add(use)
        db.session.commit()
        print(f"[audit] lead capture: {lead_email} (audit #{lead.audit_count})")

    user_id = user.id if user else None
    cfg = TIER_CONFIG[tier]

    q = queue.Queue()

    def on_progress(step, detail, current, total):
        q.put(json.dumps({"type": "progress", "step": step, "detail": detail, "current": current, "total": total}))

    def worker():
        start_dt = datetime.utcnow()
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

    t = threading.Thread(target=worker)
    t.start()

    def generate():
        while True:
            msg = q.get()
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
        comp_block = "\n".join(
            f"  - {c.get('name')}: cited in {c.get('mention_count')} of {total} responses"
            for c in competitors[:8]
        ) or "  (none surfaced)"

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

THE #1 MOVE (already computed from this data — Sentence 3 MUST be about THIS outlet so the summary agrees with the report's headline action):
{hm_block}

Write EXACTLY 3 sentences. Lead with the single most important insight, no preamble.
  Sentence 1 — THE POSITION: {brand}'s mindshare framed against the top competitor's count. If {brand} >= top competitor, lead with strength; if behind, lead with the gap. NEVER say 'lacks authority' if {brand} out-mentions competitors.
  Sentence 2 — THE SURPRISE: the single most non-obvious thing in the data. STRONGLY PREFER the per-assistant concentration when it's lopsided — e.g. "{brand}'s visibility is almost entirely one assistant (Gemini 9/10) and absent from ChatGPT, Claude, and Grok". Otherwise surface the sharpest share-of-voice gap — name a specific OUTLET and the competitor out-citing {brand} there (but NOT the #1-move outlet — save that for Sentence 3).
  Sentence 3 — THE MOVE: build this around THE #1 MOVE outlet above — name it and the competitor to displace there. Use action verbs: pitch, displace, defend, cultivate.
FRAMING RULE (critical): these are AI-citation SHARE-OF-VOICE signals — how often each AI names {brand} vs competitors when it cites an outlet — NOT press clips. Never say an outlet "covers", "features", or "wrote about" {brand}; say {brand} "over-indexes at", "is out-cited at", or "is absent from" the outlet.
BANNED: filler like 'this audit reveals', 'in today's landscape'. Discuss analysts only if they actually dominate the source domains above. Respond with ONLY the 3 sentences — no preamble, no JSON, no labels."""

        resp = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
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

    out = dict(data)
    ranked_domains = aggregate_citations(all_responses)
    ranked_domains = _augment_citations_with_named_outlets(ranked_domains, all_responses)

    editorial_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'editorial']
    editorial_domains = [d for d in editorial_domains if not _is_brand_own_domain(d['domain'], brand)]

    # Recount the brand's own mention total so the headline number reflects
    # the current guardrails (e.g. a brand named "On" was previously inflated
    # to 47/50 by the common-word collision; now correctly 17/50).
    if all_responses and brand:
        new_brand_count = _count_brand_mentions(brand, all_responses)
        if new_brand_count != (data.get('brand_mention_count') or 0):
            print(f"[rerender] brand_mention_count recount: "
                  f"{data.get('brand_mention_count')} -> {new_brand_count}")
        out['brand_mention_count'] = new_brand_count

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
        out['per_llm_visibility'] = _compute_per_llm_visibility(brand, all_responses)
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
        if len(new_media) >= 10:
            break
    for i, t in enumerate(new_media):
        t['rank'] = i + 1
    out['media_targets'] = new_media

    # Compute SoV for EXACTLY the media-target outlets the report shows, so every
    # card gets its competitor breakdown and the verdicts aren't capped out by
    # obscure non-target outlets.
    try:
        _sov_eds = _editorial_dicts_for_targets(ranked_domains, new_media) or editorial_domains
        new_sov = _compute_outlet_share_of_voice(
            brand, competitor_counts, all_responses, _sov_eds,
            max_outlets=max(10, len(_sov_eds)))
    except Exception as e:
        print(f"[rerender] SoV failed: {e}")
        new_sov = data.get('outlet_sov') or []
    out['outlet_sov'] = new_sov

    # Patch synthesized rationales now that each outlet's verdict is known.
    sov_by_dom = {(r.get('domain') or '').lower(): r for r in new_sov}
    for t in new_media:
        if t.get('_synthesized'):
            t['rationale'] = _synth_rationale(
                (sov_by_dom.get((t.get('domain') or '').lower()) or {}).get('verdict'))

    try:
        out['headline_move'] = _compute_headline_move(brand, new_sov)
    except Exception:
        out['headline_move'] = None

    # Brand-coverage verification (BACKSTAGE ONLY — saved for the CSV, not shown
    # as a coverage tier). Only meaningful when grounded (real URLs).
    if any(r.get("grounded") for r in all_responses):
        try:
            _verify_brand_coverage(brand, out.get("media_targets") or [], ranked_domains)
        except Exception:
            pass
    # Coverage-guard: 'strength' requires on-page coverage; otherwise -> opportunity.
    try:
        _coverage_guard_verdicts(out.get("media_targets"), new_sov, brand)
    except Exception:
        pass
    # Rank by AI prominence (responses citing + citation frequency), not coverage.
    try:
        _sort_targets_by_prominence(out.get("media_targets"), new_sov)
    except Exception:
        pass

    # Optionally refresh the 'What we found' executive summary with the
    # current prompt wording — the one piece the pure-Python pass can't
    # regenerate. One Claude call.
    if regenerate_summary:
        out['executive_summary'] = _regenerate_executive_summary(out)

    # Mark this view so the UI / operator knows it's a rerender, not the
    # original saved analysis.
    out['_rerendered'] = True
    return out


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

    want_fresh = request.args.get('fresh') == '1'
    want_refresh = request.args.get('refresh') == '1'
    if (want_fresh or want_refresh) and _ip_is_exempt_from_cap(_client_ip()):
        try:
            data = _rerender_from_cached_responses(data, regenerate_summary=want_refresh)
            print(f"[rerender] applied current logic to slug={slug} "
                  f"(operator, summary_regen={want_refresh})")
        except Exception as e:
            print(f"[rerender] failed for slug={slug}: {e}")

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
    )


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

    sg_key = os.environ.get("SENDGRID_API_KEY")
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
            sg = SendGridAPIClient(sg_key)
            sg.send(msg)
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
        sg_key = os.environ.get("SENDGRID_API_KEY")
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
            SendGridAPIClient(sg_key).send(msg)
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
    sg_key = os.environ.get("SENDGRID_API_KEY")
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
        SendGridAPIClient(sg_key).send(msg)
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
    dev_link = link if not os.environ.get("SENDGRID_API_KEY") else None
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


if __name__ == "__main__":
    # Get port from environment variable or default to 5009
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
