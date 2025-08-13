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

# Create the database tables
with app.app_context():
    db.create_all()

# API keys and configuration
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GA_MEASUREMENT_ID = os.environ.get("GA_MEASUREMENT_ID")

# Debug logging for API keys
print(f"NEWS_API_KEY is {'set' if NEWS_API_KEY else 'NOT SET'}")
print(f"ANTHROPIC_API_KEY is {'set' if ANTHROPIC_API_KEY else 'NOT SET'}")
print(f"GA_MEASUREMENT_ID is {'set' if GA_MEASUREMENT_ID else 'NOT SET'}")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

def analyze_articles(articles, query):
    """Extract key metrics and patterns from articles."""
    # Batch sentiment analysis for all articles
    texts = [f"{article['title']} {article['description'] or ''}" for article in articles]
    
    # Create a numbered list for Claude to reference
    numbered_texts = "\n\n".join(f"Text {i+1}:\n{text}" for i, text in enumerate(texts))
    
    # If no Anthropic key, default sentiments to neutral to allow demo flows
    if not ANTHROPIC_API_KEY:
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
    
    # Topic extraction
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
        'after', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'can', 'could', 'may', 'might', 'must', 'it', 'its'
    }
    
    # Add search terms to stop words
    search_terms = set(query.lower().split())
    stop_words.update(search_terms)
    
    # Topic extraction with better filtering
    text = ' '.join(article['title'] + ' ' + (article['description'] or '')
                   for article in articles).lower()
    words = re.findall(r'\b\w+\b', text)
    topics = Counter(word for word in words 
                    if word not in stop_words 
                    and len(word) > 2 
                    and not word.isnumeric()
                    and not any(char.isdigit() for char in word))
    
    top_topics = [{'topic': topic, 'count': count} 
                  for topic, count in topics.most_common(30)
                  if count > 2]

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

                key = (title, link)
                if key in seen_keys:
                    continue

                all_articles.append({
                    'title': title,
                    'description': description,
                    'publishedAt': (dt.isoformat() if dt else datetime.utcnow().isoformat()),
                    'source': {'name': source_name},
                    'url': link,
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
            return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID)

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

        try:
            articles1 = fetch_news_api_articles(query1, from_date1, to_date1, language=language1, sources=sources1, page_size=60)
            articles2 = fetch_news_api_articles(query2, from_date2, to_date2, language=language2, sources=sources2, page_size=60) if query2 else []
        except Exception as e:
            print(f"NewsAPI error: {e}")
            articles1, articles2 = [], []

        # Fallback to RSS if NewsAPI returns nothing
        if not articles1:
            articles1 = fetch_rss_articles(query1, from_date1, to_date1, max_items=60)
        if query2 and not articles2:
            articles2 = fetch_rss_articles(query2, from_date2, to_date2, max_items=60)

        if not articles1 and (not query2 or not articles2):
            flash("No results found for the selected range and terms. Try broadening the date range or simplifying the query.")
            return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID)

        # Analyze and render
        analysis1 = analyze_articles(articles1, query1)
        analysis2 = analyze_articles(articles2, query2) if query2 else None

        # Persist sharable result with short slug
        form_data = {
            'language1': language1, 'language2': language2,
            'source1': sources1, 'source2': sources2,
            'from_date1': from_date1, 'to_date1': to_date1,
            'from_date2': from_date2, 'to_date2': to_date2
        }
        payload = {
            "query1": query1, "query2": query2,
            "enhanced_query1": {"enhanced_query": query1, "entity_type": "brand", "reasoning": "Live fetch (NewsAPI) with RSS fallback"},
            "enhanced_query2": ({"enhanced_query": query2, "entity_type": "brand", "reasoning": "Live fetch (NewsAPI) with RSS fallback"} if query2 else None),
            "textual_analysis": None,
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
    return render_template("index.html", ga_measurement_id=GA_MEASUREMENT_ID)

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
        sent1 = a1.get("avg_sentiment", 0) or 0
        total2 = a2.get("total_articles", 0) or 0
        sent2 = a2.get("avg_sentiment", 0) or 0

        share_url = request.url_root.rstrip('/') + f"/results/{slug}"
        summary_lines = [
            f'Media Analysis for "{query1}"' + (f' vs "{query2}"' if query2 else ""),
            "",
            f"Link: {share_url}",
            "",
            "Coverage Metrics:",
            f"- {query1}: {total1} articles, Avg Sentiment {sent1:.2f}",
        ]
        if query2:
            summary_lines.append(f"- {query2}: {total2} articles, Avg Sentiment {sent2:.2f}")

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
        avg = a1.get("avg_sentiment", 0) or 0
        dr = (a1.get("date_range") or {})
        date_start = dr.get("start") or ""
        date_end = dr.get("end") or ""
    except Exception:
        query1 = "Media Analysis"
        query2 = None
        total = 0
        avg = 0
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
    draw.text((60, 200), f"Articles: {total}  •  Avg Sentiment: {avg:.2f}", fill=fg, font=font_med)
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
        avg_sent = a1.get("avg_sentiment", 0) or 0
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
            "avg_sentiment": round(avg_sent, 2),
            "date_start": date_start,
            "date_end": date_end,
            "topics": topics_list,
            "created_at": rec.created_at.isoformat() if rec.created_at else None
        })

    return render_template("examples.html", cards=cards, ga_measurement_id=GA_MEASUREMENT_ID)


if __name__ == "__main__":
    # Get port from environment variable or default to 5009
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
