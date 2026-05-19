import os
import json
import re
import random
import requests
import html
import uuid
import io
import queue
import threading
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use the OS native trust store (macOS keychain / Windows cert store / Linux /etc/ssl).
# Fixes "self-signed certificate in certificate chain" on Python.org Python builds
# whose bundled CA path is missing. Harmless no-op on platforms where it's not needed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from datetime import datetime, timedelta
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

# File upload configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'pdf', 'pptx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize file processor
file_processor = SimpleMediaFileProcessor(os.environ.get("ANTHROPIC_API_KEY"))

# Initialize SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///waitlist.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
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
    perplexity_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai") if PERPLEXITY_API_KEY else None
except Exception:
    perplexity_client = None
print(f"PERPLEXITY_API_KEY is {'set' if PERPLEXITY_API_KEY else 'NOT SET'}")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
try:
    openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_API_KEY else None
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

NON_EDITORIAL_DOMAINS = {
    'wikipedia.org', 'en.wikipedia.org', 'wikimedia.org',
    'reddit.com', 'quora.com', 'stackexchange.com', 'stackoverflow.com',
    'youtube.com', 'youtu.be',
    'linkedin.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
    'amazon.com', 'ebay.com',
    'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'who.int', 'cdc.gov',
    'arxiv.org', 'researchgate.net', 'sciencedirect.com', 'springer.com',
    'jstor.org', 'nature.com', 'science.org',
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
    # E-commerce / OTA aggregators
    'expedia.com', 'booking.com', 'kayak.com', 'tripadvisor.com', 'yelp.com',
    'glassdoor.com', 'indeed.com',
}


def classify_citation_domain(domain):
    """Return 'analyst', 'institutional', 'non_editorial', or 'editorial'."""
    d = (domain or "").lower().lstrip('.')
    d_stripped = d[4:] if d.startswith('www.') else d

    if d in NON_EDITORIAL_DOMAINS or d_stripped in NON_EDITORIAL_DOMAINS:
        return 'non_editorial'
    if d in NON_EDITORIAL_VENDORS or d_stripped in NON_EDITORIAL_VENDORS:
        return 'non_editorial'
    if d_stripped in ANALYST_DOMAINS or d in ANALYST_DOMAINS:
        return 'analyst'
    if d_stripped in EDITORIAL_ORG_ALLOWLIST:
        return 'editorial'
    if any(d.endswith(tld) for tld in INSTITUTIONAL_TLDS):
        return 'institutional'
    if d.endswith('.org'):
        return 'institutional'
    return 'editorial'


def verify_editorial_domains(editorial_domains, brand, category):
    """Filter the editorial list via Claude to drop B2B vendor/marketplace/non-media domains.

    Returns (verified_media, rejected) — both lists in original ranking order.
    On any failure, returns (editorial_domains, []) — safe to no-op.
    """
    if not editorial_domains:
        return [], []
    domains_to_check = editorial_domains[:30]
    domain_list = "\n".join(f"- {d['domain']}" for d in domains_to_check)
    try:
        resp = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": (
                    "You are filtering a list of web domains to identify which are legitimate editorial "
                    "MEDIA outlets that a PR professional could pitch — vs. which are SaaS vendors, "
                    "software review marketplaces, e-commerce, community forums, or corporate sites "
                    "that aren't pitchable media.\n\n"
                    f"Context — brand: {brand}\nCategory: {category}\n\n"
                    "Definition of MEDIA: newspapers, magazines, trade press, B2B publications, "
                    "consumer publications, broadcast outlets, and digital-native news sites WITH "
                    "editorial staff who write articles.\n\n"
                    "Definition of NON-MEDIA (reject these): software vendor sites (e.g. salesforce.com, "
                    "atlassian.com), review marketplaces (e.g. g2.com, capterra.com, trustradius.com, "
                    "trustpilot.com), corporate blogs, e-commerce/aggregators (e.g. amazon.com, "
                    "expedia.com), Wikipedia, Reddit/Stack Overflow communities, LinkedIn, Quora.\n\n"
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
    """Count citation URLs by domain across responses, tracking LLMs and prompts."""
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
        d['urls'] = list(set(d['urls']))[:10]

    return sorted(domain_data.values(), key=lambda x: x['count'], reverse=True)


TIER_CONFIG = {
    "free": {
        "prompt_count": 10,
        "llms": ["Claude", "ChatGPT", "Gemini"],
        "media_target_count": 5,
        "institutional_target_count": 5,
        "analyst_target_count": 5,
        "max_workers": 10,
    },
    "paid": {
        "prompt_count": 100,
        "llms": ["Claude", "ChatGPT", "Gemini", "Perplexity", "Grok"],
        "media_target_count": 25,
        "institutional_target_count": 10,
        "analyst_target_count": 10,
        "max_workers": 12,
    },
}


def _call_llm(provider, enriched_prompt):
    """Send one citation-forcing prompt to one provider; return response text or raise."""
    if provider == "Claude":
        resp = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=CITATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": enriched_prompt}],
        )
        return resp.content[0].text
    if provider == "ChatGPT":
        if not openai_client:
            raise RuntimeError("OPENAI_API_KEY not configured")
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": enriched_prompt},
            ],
        )
        return resp.choices[0].message.content
    if provider == "Gemini":
        if not gemini_client:
            raise RuntimeError("GEMINI_API_KEY not configured")
        gemini_prompt = enriched_prompt + "\n\nWhen providing information, please include specific sources and references. Format any sources as: Reference: [Source Name] - URL. Include authoritative sources like news sites, company websites, industry reports."
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
        )
        return resp.text
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
        return resp.choices[0].message.content
    if provider == "Grok":
        if not openrouter_client:
            raise RuntimeError("OPENROUTER_API_KEY not configured")
        resp = openrouter_client.chat.completions.create(
            model=os.environ.get("XAI_OPENROUTER_MODEL", "x-ai/grok-4"),
            max_tokens=2000,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": enriched_prompt},
            ],
        )
        return resp.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


def run_citation_audit(problem_statement, on_progress=None, tier="free"):
    """Full agent pipeline: one prompt in, ranked media/partnership/analyst lists out."""
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    prompt_count = cfg["prompt_count"]
    llms = cfg["llms"]
    media_limit = cfg["media_target_count"]
    institutional_limit = cfg["institutional_target_count"]
    analyst_limit = cfg["analyst_target_count"]
    max_workers = cfg["max_workers"]

    def emit(step, detail, current=0, total=0):
        if on_progress:
            on_progress(step, detail, current, total)

    emit("prompts", "Generating search prompts...", 0, 1)
    prompt_gen_response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000 if prompt_count > 20 else 2000,
        messages=[{
            "role": "user",
            "content": f"""You are an AI citation strategist. A communications professional wants their brand to be the answer when people ask AI assistants about a specific problem.

Their goal: "{problem_statement}"

Your job:
1. Identify the BRAND from their statement (the company/product they want to promote).
2. Identify the CATEGORY or PROBLEM SPACE.
3. Generate exactly {prompt_count} prompts that a real person would type into ChatGPT, Claude, or Gemini when researching this problem space. These should be natural, varied prompts — some broad ("best X for Y"), some specific ("X vs Y comparison"), some question-based ("how do I solve Y?"), some recommendation-seeking ("what do experts recommend for Y?"). At higher prompt counts, cover adjacent angles: pricing, alternatives, troubleshooting, expert opinions, regulatory considerations, real-world reviews, integration questions, regional perspectives.

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
    brand = prompt_data["brand"]
    category = prompt_data["category"]
    prompts = prompt_data["prompts"][:prompt_count]
    emit("prompts", f"Generated {len(prompts)} prompts for \"{brand}\"", 1, 1)

    tasks = [(provider, pi, prompt_text)
             for pi, prompt_text in enumerate(prompts)
             for provider in llms]
    total_calls = len(tasks)

    def run_one(provider, pi, prompt_text):
        enriched = prompt_text + CITATION_SUFFIX
        try:
            resp_text = _call_llm(provider, enriched)
            return {"llm": provider, "prompt": prompt_text, "response": resp_text, "citations": extract_urls(resp_text)}
        except Exception as e:
            return {"llm": provider, "prompt": prompt_text, "response": f"[Error: {e}]", "citations": []}

    all_responses = []
    completed = 0
    progress_lock = threading.Lock()

    emit("llm", f"Querying {len(llms)} LLMs × {len(prompts)} prompts ({total_calls} calls)...", 0, total_calls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, *t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            with progress_lock:
                all_responses.append(result)
                completed += 1
                emit("llm", f"Completed {completed}/{total_calls} ({result['llm']})", completed, total_calls)

    emit("extract", "Extracting and aggregating citations...", 0, 1)
    ranked_domains = aggregate_citations(all_responses)
    total_citations = sum(d['count'] for d in ranked_domains)
    emit("extract", f"Found {total_citations} citations across {len(ranked_domains)} domains", 1, 1)

    emit("analysis", "Verifying editorial sources...", 0, 1)

    editorial_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'editorial']
    institutional_domains = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'institutional']
    analyst_domains_found = [d for d in ranked_domains if classify_citation_domain(d['domain']) == 'analyst']

    # AI verification: filter out B2B vendors / marketplaces / non-media domains the heuristic missed.
    editorial_domains, rejected_editorial = verify_editorial_domains(editorial_domains, brand, category)
    if rejected_editorial:
        print(f"verify_editorial_domains: filtered out {len(rejected_editorial)} non-media domains: " +
              ", ".join(d['domain'] for d in rejected_editorial[:10]))

    emit("analysis", "Building your signal report...", 0, 1)

    def fmt_block(domains, limit):
        return "\n".join(
            f"  {i+1}. {d['domain']} — cited {d['count']}x by {', '.join(d['llms'])} — sample URLs: {', '.join(d['urls'][:3])}"
            for i, d in enumerate(domains[:limit])
        ) or "  (none)"

    editorial_block = fmt_block(editorial_domains, max(15, media_limit * 2))
    institutional_block = fmt_block(institutional_domains, max(10, institutional_limit * 2))
    analyst_block = fmt_block(analyst_domains_found, max(10, analyst_limit * 2))

    responses_block = ""
    for i, r in enumerate(all_responses):
        citation_urls = [c['url'] for c in r.get('citations', [])]
        responses_block += f"\n--- Response {i+1} [{r['llm']}] ---\nPrompt: {r['prompt']}\nCitations found: {citation_urls}\n{r['response'][:500]}\n"

    analysis_response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""You are an AI citation intelligence analyst. You have actual citation data extracted from 30 AI-generated responses (10 prompts × 3 LLMs: Claude, ChatGPT, Gemini) about the category "{category}".

The client's brand is "{brand}". Their goal: "{problem_statement}"

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

1. TOP {media_limit} MEDIA TARGETS — drawn ONLY from the EDITORIAL DOMAINS above. These are pitch-able publications.
2. AUTHORITY & PARTNERSHIP TARGETS — drawn ONLY from the INSTITUTIONAL / ASSOCIATION DOMAINS above. Up to {institutional_limit}, only if meaningfully present. Universities, agencies, certification bodies, trade associations, advocacy non-profits, or government bodies.
3. ANALYST TARGETS — drawn ONLY from the ANALYST FIRMS above. Up to {analyst_limit}, only if meaningfully present. Influence is via analyst relations, not pitching or partnership.

If a category has nothing, return an empty array for it.

For each target:
- Map domain to its proper name (e.g. allure.com → Allure, nih.gov → National Institutes of Health, harvard.edu → Harvard University, omri.org → OMRI, garden.org → National Gardening Association, gartner.com → Gartner, forrester.com → Forrester)
- Note competitors discussed alongside it
- Explain the influence strategy (editorial: how a placement helps; authority/partnership: what specific partnership/certification/sponsorship/research collab; analyst: which evaluation to target, briefing cadence, or research sponsorship that would shift citations)

CRITICAL: Only include entities that actually appear in the citation data above. Do NOT invent or guess.

Respond with ONLY valid JSON:
{{
  "brand": "{brand}",
  "category": "{category}",
  "brand_mention_count": <number of times {brand} appeared across all 30 responses>,
  "total_responses": {len(all_responses)},
  "total_citations_extracted": {total_citations},
  "competitors": [
    {{"name": "competitor name", "mention_count": <int>, "cited_by": ["Claude", "ChatGPT", "Gemini"]}}
  ],
  "media_targets": [
    {{
      "rank": 1,
      "outlet": "Publication name derived from actual citation domain",
      "domain": "the actual domain from citation data",
      "reporter": "Named journalist if identifiable from response content, otherwise null",
      "citation_frequency": <actual count from citation data>,
      "sample_urls": ["1-2 actual URLs that were cited"],
      "cited_by_llms": ["which LLMs cited this outlet"],
      "competitors_citing": ["competitors discussed in responses that cited this outlet"],
      "rationale": "One sentence on why earned media here would move the needle for {brand}",
      "gap_insight": "What specifically competitors are getting credit for via this outlet that {brand} should own"
    }}
  ],
  "institutional_targets": [
    {{
      "rank": 1,
      "institution": "Proper name of the institution",
      "domain": "the actual domain from citation data",
      "type": "Government | Academic | Research Institute | Trade Association | Certification Body | Advocacy Non-profit",
      "citation_frequency": <actual count from citation data>,
      "sample_urls": ["1-2 actual URLs that were cited"],
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
      "sample_urls": ["1-2 actual URLs that were cited"],
      "cited_by_llms": ["which LLMs cited this firm"],
      "evaluations_referenced": ["Specific reports/evaluations cited if identifiable from response text — e.g. 'Magic Quadrant for APM', 'Forrester Wave: Observability' — otherwise empty array"],
      "analyst_play": "One sentence on the specific analyst relations move: which evaluation to target, briefing cadence to establish, sponsored research, or client subscription that would shift citations"
    }}
  ],
  "executive_summary": "3-4 sentences. Lead with the editorial findings — what they reveal about how AI describes {brand} and competitors, and where earned media will move the needle. Discuss analyst relations or authority partnerships ONLY if the corresponding target lists below contain actual entries. For consumer brands or categories where analyst firms are not relevant (hospitality, fashion, food & beverage, consumer products, etc.), do NOT mention analysts or speculate about their absence — silence is fine."
}}"""
        }]
    )

    analysis_text = analysis_response.content[0].text
    analysis_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
    if not analysis_match:
        raise ValueError("Failed to parse analysis response")
    analysis = json.loads(analysis_match.group())
    analysis["prompts_used"] = prompts
    analysis["raw_citation_domains"] = ranked_domains[:20]
    return analysis


@app.route('/citation-audit', methods=['GET', 'POST'])
def citation_audit():
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

    requested_tier = (request.form.get('tier') or 'free').strip().lower()
    if requested_tier not in ('free', 'paid'):
        requested_tier = 'free'

    tier = 'free'
    credit_charged = False
    if requested_tier == 'paid':
        if not user:
            return jsonify({"error": "Please sign in to run a paid audit.", "code": "auth_required"}), 401
        if credits < 1:
            return jsonify({"error": "No audit credits remaining. Buy more to continue.", "code": "no_credits"}), 402
        bal = CreditBalance.query.filter_by(user_id=user.id).first()
        bal.credits_remaining = (bal.credits_remaining or 0) - 1
        db.session.commit()
        tier = 'paid'
        credit_charged = True

    user_id = user.id if user else None
    cfg = TIER_CONFIG[tier]

    q = queue.Queue()

    def on_progress(step, detail, current, total):
        q.put(json.dumps({"type": "progress", "step": step, "detail": detail, "current": current, "total": total}))

    def worker():
        try:
            result = run_citation_audit(problem_statement, on_progress=on_progress, tier=tier)
            slug = uuid.uuid4().hex[:10]
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
                db.session.commit()
            result["slug"] = slug
            result["tier"] = tier
            q.put(json.dumps({"type": "result", "data": result}))
        except Exception as e:
            if credit_charged and user_id is not None:
                try:
                    with app.app_context():
                        bal_inner = CreditBalance.query.filter_by(user_id=user_id).first()
                        if bal_inner:
                            bal_inner.credits_remaining = (bal_inner.credits_remaining or 0) + 1
                            db.session.commit()
                except Exception as refund_err:
                    print("Credit refund failed:", refund_err)
            q.put(json.dumps({"type": "error", "error": str(e)}))

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


@app.route('/signal/<slug>')
def view_signal_report(slug):
    """Render a shared PR Signal Finder report."""
    data = _load_signal_report(slug)
    if not data:
        flash("Report not found or expired.")
        return redirect(url_for('citation_audit'))
    share_url = request.url_root.rstrip('/') + url_for('view_signal_report', slug=slug)
    pdf_url = request.url_root.rstrip('/') + url_for('signal_report_pdf', slug=slug)
    user = current_signal_user()
    return render_template(
        'citation_audit.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        shared_data=data,
        share_url=share_url,
        pdf_url=pdf_url,
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


@app.route('/citation-audit/request-demo', methods=['POST'])
def citation_audit_request_demo():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    title = (data.get('title') or '').strip()
    org = (data.get('org') or '').strip()
    email = (data.get('email') or '').strip()
    slug = (data.get('slug') or '').strip() or None
    problem_statement = (data.get('problem_statement') or '').strip()

    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if '@' not in email or '.' not in email:
        return jsonify({"error": "Please enter a valid email address."}), 400

    extra = json.dumps({"name": name, "title": title, "org": org, "problem_statement": problem_statement})
    lead = LeadCapture(email=email, slug=slug, app_name='signal_finder_demo', extra=extra)
    db.session.add(lead)
    db.session.commit()

    sg_key = os.environ.get("SENDGRID_API_KEY")
    if sg_key:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            report_link = f"https://innatec3.com/results/{slug}" if slug else "(no audit slug)"
            text_body = (
                f"New PR Signal Finder bespoke audit request:\n\n"
                f"Name: {name}\n"
                f"Title: {title or '(not provided)'}\n"
                f"Organization: {org or '(not provided)'}\n"
                f"Email: {email}\n\n"
                f"Problem statement they audited:\n{problem_statement or '(not provided)'}\n\n"
                f"Their light audit report: {report_link}\n"
            )
            report_link_html = (
                f'<a href="{report_link}">{report_link}</a>' if slug else '<em>(no slug captured)</em>'
            )
            email_link_html = f'<a href="mailto:{html.escape(email)}">{html.escape(email)}</a>'
            html_body = (
                f"<h3>New PR Signal Finder bespoke audit request</h3>"
                f"<p><strong>Name:</strong> {html.escape(name)}<br>"
                f"<strong>Title:</strong> {html.escape(title) or '<em>(not provided)</em>'}<br>"
                f"<strong>Organization:</strong> {html.escape(org) or '<em>(not provided)</em>'}<br>"
                f"<strong>Email:</strong> {email_link_html}</p>"
                f"<p><strong>Problem they audited:</strong><br>{html.escape(problem_statement) or '<em>(not provided)</em>'}</p>"
                f"<p><strong>Their light audit report:</strong> {report_link_html}</p>"
            )
            msg = Mail(
                from_email=("nstrauss@innatec3.com", "PR Signal Finder"),
                to_emails=["nstrauss@innatec3.com"],
                subject=f"[PR Signal Finder] Bespoke audit request from {name}" + (f" ({org})" if org else ""),
                plain_text_content=text_body,
                html_content=html_body,
            )
            sg = SendGridAPIClient(sg_key)
            sg.send(msg)
        except Exception as e:
            print("SendGrid lead notification error:", e)

    return jsonify({"ok": True, "dev": not bool(sg_key)})


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
        if not session.get('signal_user_id'):
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
        return redirect(url_for('signal_dashboard'))
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
    rec = LoginToken.query.filter_by(token_hash=_hash_token(raw)).first()
    if not rec or rec.used_at or rec.expires_at < datetime.utcnow():
        flash("Sign-in link expired or already used. Request a new one.")
        return redirect(url_for('signal_login'))
    rec.used_at = datetime.utcnow()
    user = SignalUser.query.filter_by(email=rec.email).first()
    if not user:
        user = SignalUser(email=rec.email)
        db.session.add(user)
        db.session.flush()
        db.session.add(CreditBalance(user_id=user.id, credits_remaining=0))
    user.last_login_at = datetime.utcnow()
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
@signal_login_required
def signal_dashboard():
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
    user = current_signal_user()
    purchase = None
    problem_statement = None

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
            if granted_user:
                if not user or user.id != granted_user.id:
                    session['signal_user_id'] = granted_user.id
                    user = granted_user
            if granted_purchase:
                purchase = granted_purchase
        except Exception as e:
            print("Stripe success-page processing error:", e)

    credits = current_user_credits(user) if user else 0
    return render_template(
        'signal_checkout_success.html',
        ga_measurement_id=GA_MEASUREMENT_ID,
        user=user,
        credits=credits,
        purchase=purchase,
        problem_statement=problem_statement,
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
