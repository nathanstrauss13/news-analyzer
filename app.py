import os
import json
import re
import random
import requests
import html
import uuid
import io
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from markupsafe import Markup
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from dateutil.parser import parse
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from utils.simple_file_processor import SimpleMediaFileProcessor

load_dotenv()

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

@app.route("/", methods=["GET", "POST"])
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
                    note = Markup("<p><strong>Note:</strong> No results were returned for the selected historical range. "
                                  "Showing recent coverage from the last 30 days instead due to provider limitations.</p>")

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
                from_email=("no-reply@innatec3.com", "innate c3"),
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
            from_email=("no-reply@innatec3.com", "innate c3"),
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

if __name__ == "__main__":
    # Get port from environment variable or default to 5009
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
