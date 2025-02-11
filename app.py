import os
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
import openai
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateutil.parser
from collections import defaultdict
import statistics

load_dotenv()

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your secure key

# Set API keys from environment variables
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

#############################
# NLP Extraction Functions
#############################

def extract_query_parameters(user_prompt):
    prompt = f"""
You are an assistant that extracts search parameters for a news query.
Output a JSON object with these keys:
- "keywords": a string for the search term.
- "relative_time": if the query mentions a relative time period (e.g., "past_week" or "past_month"), output that value; otherwise, empty.
- "from_date": if an absolute start date is provided, output it in YYYY-MM-DD format; otherwise, empty.
- "to_date": if an absolute end date is provided, output it in YYYY-MM-DD format; otherwise, empty.
- "language": the news language (default "en").
- "domains": a string representing the news source domain (e.g., "gizmodo.com"), optional.

Example 1: For "I need the latest US tech news" output:
{{"keywords": "US tech", "relative_time": "", "from_date": "", "to_date": "", "language": "en", "domains": ""}}

Example 2: For "Give me the latest Apple news from the past month in gizmodo.com" output:
{{"keywords": "Apple", "relative_time": "past_month", "from_date": "", "to_date": "", "language": "en", "domains": "gizmodo.com"}}

User Request: "{user_prompt}"
JSON Output:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract structured search parameters for a News API query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print("Error extracting parameters:", e)
        raise

def extract_comparative_query_parameters(user_prompt):
    prompt = f"""
You are an assistant that extracts search parameters for two news queries from a single prompt.
The request includes two queries separated by "vs".
Output a JSON object with two keys: "dataset1" and "dataset2". Each value should be a JSON object with these keys:
- "keywords"
- "relative_time"
- "from_date"
- "to_date"
- "language"
- "domains" (optional)

Example: For "analyze recent coverage of Apple vs Google in gizmodo.com over the past month" output:
{{
  "dataset1": {{"keywords": "Apple", "relative_time": "past_month", "from_date": "", "to_date": "", "language": "en", "domains": "gizmodo.com"}},
  "dataset2": {{"keywords": "Google", "relative_time": "past_month", "from_date": "", "to_date": "", "language": "en", "domains": "gizmodo.com"}}
}}

User Request: "{user_prompt}"
JSON Output:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract structured search parameters for two News API queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print("Error extracting comparative parameters:", e)
        raise

#############################
# Utility Functions
#############################

def apply_relative_dates(params):
    today = datetime.today()
    params["to_date"] = today.strftime("%Y-%m-%d")
    relative_time = params.get("relative_time", "").lower()
    if relative_time == "past_week":
        params["from_date"] = (today - relativedelta(weeks=1)).strftime("%Y-%m-%d")
    elif relative_time == "past_month":
        params["from_date"] = (today - relativedelta(months=1)).strftime("%Y-%m-%d")
    else:
        params["from_date"] = params.get("from_date", "")
    return params

def fetch_news(params):
    base_url = "https://newsapi.org/v2/everything"
    query_params = {
        "q": params.get("keywords", ""),
        "language": params.get("language", "en"),
        "apiKey": NEWS_API_KEY,
        "sortBy": "relevancy"
    }
    if params.get("from_date"):
        query_params["from"] = params["from_date"]
    if params.get("to_date"):
        query_params["to"] = params["to_date"]
    if params.get("domains"):
        query_params["domains"] = params.get("domains")
    response = requests.get(base_url, params=query_params)
    if response.status_code != 200:
        print("News API response:", response.text)
        raise Exception(f"News API error: {response.text}")
    data = response.json()
    return data.get("articles", [])

def analyze_articles(articles, user_prompt):
    articles_summary = "\n\n".join(
        [f"Title: {article.get('title', 'No Title')}\nDescription: {article.get('description', 'No Description')}"
         for article in articles[:5]]
    )
    analysis_prompt = f"""
The user asked: "{user_prompt}"

Here are some news articles that match the query:
{articles_summary}

Provide a summary and analysis highlighting key points and overall sentiment.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert news analyst."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error analyzing articles:", e)
        raise

def compute_chart_data(articles):
    counts = {}
    for article in articles:
        outlet = article.get("source", {}).get("name", "Unknown")
        counts[outlet] = counts.get(outlet, 0) + 1
    labels = list(counts.keys())
    values = list(counts.values())
    return {"labels": labels, "values": values}

def compute_comparative_chart_data(articles1, articles2):
    def count_by_outlet(articles):
        counts = {}
        for article in articles:
            outlet = article.get("source", {}).get("name", "Unknown")
            counts[outlet] = counts.get(outlet, 0) + 1
        return counts
    counts1 = count_by_outlet(articles1)
    counts2 = count_by_outlet(articles2)
    all_outlets = set(counts1.keys()).union(set(counts2.keys()))
    labels = sorted(list(all_outlets))
    values1 = [counts1.get(label, 0) for label in labels]
    values2 = [counts2.get(label, 0) for label in labels]
    return {"labels": labels, "values1": values1, "values2": values2}

def compute_volume_chart_data(articles1, articles2):
    def count_by_date(articles):
        counts = defaultdict(int)
        for article in articles:
            published = article.get("publishedAt")
            if published:
                try:
                    date_obj = dateutil.parser.parse(published)
                    date_str = date_obj.strftime("%Y-%m-%d")
                    counts[date_str] += 1
                except Exception:
                    pass
        return counts
    counts1 = count_by_date(articles1)
    counts2 = count_by_date(articles2)
    all_dates = set(counts1.keys()).union(set(counts2.keys()))
    labels = sorted(list(all_dates))
    values1 = [counts1.get(date, 0) for date in labels]
    values2 = [counts2.get(date, 0) for date in labels]
    return {"labels": labels, "values1": values1, "values2": values2}

def compute_volume_chart_annotations(articles, volume_data):
    values = volume_data["values1"]
    if not values:
        return {}
    avg = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0
    annotations = {}
    for i, date in enumerate(volume_data["labels"]):
        if values[i] > avg + stdev:
            headlines = [article.get("title", "") for article in articles 
                         if article.get("publishedAt") and 
                         dateutil.parser.parse(article.get("publishedAt")).strftime("%Y-%m-%d") == date]
            prompt = f"On {date}, there was a spike in coverage with these headlines: {headlines}. What event or trend likely drove this spike?"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a media analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                explanation = response.choices[0].message.content.strip()
            except Exception as e:
                explanation = "Explanation not available."
            annotations[date] = explanation
    return annotations

#####################################
# Endpoints
#####################################

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("query")
        if not user_query:
            flash("Please enter a query.")
            return redirect(url_for("index"))
        return redirect(url_for("result", q=user_query))
    return render_template("simple_index.html")

@app.route("/result")
def result():
    user_query = request.args.get("q")
    if not user_query:
        flash("No query provided.")
        return redirect(url_for("index"))
    
    if ("vs" in user_query.lower()) or ("compare" in user_query.lower()):
        query_params = extract_comparative_query_parameters(user_query)
        query_params["dataset1"] = apply_relative_dates(query_params["dataset1"])
        query_params["dataset2"] = apply_relative_dates(query_params["dataset2"])
        articles1 = fetch_news(query_params["dataset1"])
        articles2 = fetch_news(query_params["dataset2"])
        response_data = {
            "query_type": "comparative",
            "query_params": query_params,
            "articles_dataset1": articles1,
            "articles_dataset2": articles2
        }
    else:
        query_params = extract_query_parameters(user_query)
        query_params = apply_relative_dates(query_params)
        articles = fetch_news(query_params)
        response_data = {
            "query_type": "single",
            "query_params": query_params,
            "articles": articles
        }
    formatted_json = json.dumps(response_data, indent=2)
    return render_template("simple_result.html", q=user_query, json_data=formatted_json)

if __name__ == "__main__":
    app.run(debug=True)
