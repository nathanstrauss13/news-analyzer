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

# Set API keys from environment
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

##########################
# NLP Extraction Functions
##########################

def extract_query_parameters(user_prompt):
    prompt = f"""
You are an assistant that extracts search parameters for a news query.
Given the user's request, output a JSON object with the following keys:
- "keywords": a string for the search term.
- "relative_time": if the user mentions a relative time period (e.g., "past_week" or "past_month"), output that value; otherwise, leave it empty.
- "from_date": if an absolute start date is provided, output it in YYYY-MM-DD format; otherwise, leave it empty.
- "to_date": if an absolute end date is provided, output it in YYYY-MM-DD format; otherwise, leave it empty.
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
                {"role": "user", "content": prompt},
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
The user's request will include two separate queries separated by "vs".
For each dataset, output a JSON object with the following keys:
- "keywords": a string for the search term.
- "relative_time": if a relative time is mentioned (e.g., "past_week" or "past_month"), output that value; otherwise, leave it empty.
- "from_date": if an absolute start date is provided, output it in YYYY-MM-DD format; otherwise, leave it empty.
- "to_date": if an absolute end date is provided, output it in YYYY-MM-DD format; otherwise, leave it empty.
- "language": the news language (default "en").
- "domains": a string representing the news source domain (e.g., "gizmodo.com"), optional.

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
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print("Error extracting comparative parameters:", e)
        raise

##########################
# Utility Functions
##########################

def apply_relative_dates(params):
    if params.get("from_date") and params.get("to_date"):
        return params
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
        "sortBy": "relevancy"  # sort by relevancy
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

##########################
# Endpoints
##########################

# This simple UI will show a form to enter the query and then display the raw JSON result.
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
    
    # Determine if the query is comparative based on a simple heuristic.
    if (" vs " in user_query.lower()) or (" compare " in user_query.lower()):
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
    # Format the JSON response for display
    formatted_json = json.dumps(response_data, indent=2)
    return render_template("simple_result.html", q=user_query, json_data=formatted_json)

if __name__ == "__main__":
    app.run(debug=True)
