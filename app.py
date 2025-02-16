import os
import json
import re
from collections import Counter
from datetime import datetime
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from markupsafe import Markup
from dotenv import load_dotenv
from anthropic import Anthropic
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import os
from anthropic import Anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise Exception("Anthropic API key not found!")

client = Anthropic(api_key=api_key)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")

# API keys
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

def analyze_articles(articles, query):
    """Extract key metrics and patterns from articles."""
    # Publication timeline
    dates = {}
    for article in articles:
        date = parse(article['publishedAt']).strftime('%Y-%m-%d')
        dates[date] = dates.get(date, 0) + 1
    timeline = [{'date': k, 'count': v} for k, v in sorted(dates.items())]
    
    # News source distribution
    sources = Counter(article['source']['name'] for article in articles)
    top_sources = [{'name': name, 'count': count} 
                   for name, count in sources.most_common(10)]
    
    # Extended stop words list
    stop_words = {
        # Common English words
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
        'after', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'can', 'could', 'may', 'might', 'must', 'it', 'its',
        'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you',
        
        # Common content words
        'said', 'says', 'one', 'two', 'three', 'first', 'last', 'year',
        'years', 'month', 'months', 'day', 'days', 'today', 'tomorrow',
        'here', 'there', 'where', 'when', 'why', 'how', 'what', 'which',
        'who', 'whom', 'whose', 'get', 'got', 'getting', 'make', 'made',
        'making', 'take', 'took', 'taking', 'see', 'saw', 'seeing',
        'come', 'came', 'coming', 'go', 'went', 'going', 'know', 'knew',
        'knowing', 'think', 'thought', 'thinking', 'say', 'saying',
        'said', 'just', 'now', 'like', 'also', 'then', 'than',
        'more', 'most', 'some', 'all', 'any', 'many', 'much',
        'your', 'my', 'our', 'their', 'his', 'her', 'its',
        'during', 'while', 'before', 'after', 'under', 'over',
        
        # Numbers and time-related
        '2024', '2025', 'january', 'february', 'march', 'april', 'may',
        'june', 'july', 'august', 'september', 'october', 'november',
        'december', 'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday'
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
                  if count > 2]  # Only include topics mentioned more than twice
    
    return {
        'timeline': timeline,
        'sources': top_sources,
        'topics': top_topics,
        'total_articles': len(articles),
        'date_range': {
            'start': timeline[0]['date'] if timeline else None,
            'end': timeline[-1]['date'] if timeline else None
        }
    }

def get_date_range(time_frame):
    """Convert time frame to date range."""
    today = datetime.today()
    to_date = today.strftime("%Y-%m-%d")
    
    if time_frame == "past_week":
        from_date = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
    elif time_frame == "past_month":
        from_date = (today - relativedelta(months=1)).strftime("%Y-%m-%d")
    else:  # Default to past week
        from_date = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
    
    return from_date, to_date

def fetch_news(keywords, from_date=None, to_date=None, language="en", domains=None):
    """Fetch news articles from News API."""
    base_url = "https://newsapi.org/v2/everything"
    query_params = {
        "q": keywords,
        "language": language,
        "apiKey": NEWS_API_KEY,
        "sortBy": "relevancy",
        "pageSize": 100  # Get more articles for better analysis
    }
    
    if from_date:
        query_params["from"] = from_date
    if to_date:
        query_params["to"] = to_date
    if domains:
        query_params["domains"] = domains

    response = requests.get(base_url, params=query_params)
    if response.status_code != 200:
        raise Exception(f"News API error: {response.text}")
    return response.json().get("articles", [])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Add your existing code here
            print("POST request received")
            # Example: process form data
            data = request.form.get("data")
            print(f"Form data received: {data}")
            # Add more processing as needed
            # Example: if you have a function to handle the data, call it here
            # result = process_data(data)
            # print(f"Processing result: {result}")
        except Exception as e:
            print(f"Error processing POST request: {e}")
            flash("An error occurred while processing your request.", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    # Get the port from the environment variable or use 5001 as default
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)