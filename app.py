from flask import Flask, request, Response, render_template_string
from openai import OpenAI
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import os
import requests
from datetime import datetime, timedelta
from urllib.parse import quote

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# 1. Fetch 30-day stock data
# ------------------------------
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="30d", interval="1d")
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

# ------------------------------
# 2. Fetch recent headlines (NewsAPI.org)
# ------------------------------
def get_recent_news(ticker):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return "⚠️ No API key found (missing NEWSAPI_KEY)."

    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    query = quote(ticker)

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"from={from_date}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize=5&"
        f"apiKey={api_key}"
    )

    try:
        r = requests.get(url)
        if r.status_code != 200:
            return f"⚠️ NewsAPI returned {r.status_code}: {r.json().get('message', 'unknown error')}"
        data = r.json()
    except Exception as e:
        return f"⚠️ Network error while fetching headlines: {e}"

    if data.get("status") != "ok":
        msg = data.get("message", "unknown issue")
        if "rateLimited" in msg:
            return "⚠️ API quota reached for today (NewsAPI free plan)."
        elif "apiKeyInvalid" in msg or "apiKeyMissing" in msg:
            return "⚠️ Invalid or missing NewsAPI key."
        else:
            return f"⚠️ NewsAPI issue: {msg}"

    articles = data.get("articles", [])
    if not articles:
        return "⚠️ No recent headlines available for this ticker."

    headlines = [a["title"] for a in articles if "title" in a]
    return "\n".join(f"- {h}" for h in headlines)

# ------------------------------
# 3. Flask route
# ------------------------------
@app.route("/", methods=["GET"])
def index():
    ticker = request.args.get("ticker", "AAPL").upper()
    data = get_stock_data(ticker)

    if data is None:
        return Response(f"<h3>No stock data found for {ticker}</h3>", mimetype="text/html")

    # Calculate price move
    first_close = float(data["Close"].iloc[0])
    last_close = float(data["Close"].iloc[-1])
    pct_change = (last_close - first_close) / first_close * 100

    # Plot chart (last 30 days)
    plt.figure(figsize=(6, 4))
    plt.plot(data.index, data["Close"], color="blue")
    plt.title(f"{ticker} - Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(
        data.index[::7],  # every 7th day
        [d.strftime("%m-%d") for d in data.index[::7]],
        rotation=30,
        ha="right"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save plot as base64
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Fetch news
    headlines = get_recent_news(ticker)

    # Create GPT prompt
    prompt = f"""
You are a professional equity research analyst writing a short market note.

Ticker: {ticker}
Recent 30-day move: ${first_close:.2f} → ${last_close:.2f} ({pct_change:.2f}%)

Recent news headlines:
{headlines}

Write a concise, professional commentary explaining what likely drove this price performance.
Follow these rules:
- Base reasoning on the specific headlines above (e.g., product launches, earnings, analyst calls).
- Avoid vague platitudes like "investor sentiment" unless clearly supported.
- Always finish your response in complete sentences — never cut off mid-thought.
- Maintain a polished institutional tone.
- End with one analytical takeaway about near-term direction.
"""

    # Generate AI summary
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        comment = response.choices[0].message.content.strip()
    except Exception as e:
        comment = f"(AI summary unavailable: {e})"

    # HTML template
    html = f"""
    <html>
    <head>
        <title>{ticker} Stock Snapshot</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 40px;
                background-color: #fafafa;
            }}
            img {{
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-top: 20px;
            }}
            p {{
                width: 80%;
                margin: 20px auto;
                line-height: 1.6;
                font-size: 16px;
                text-align: justify;
            }}
            .warning {{
                color: gray;
                font-style: italic;
                font-size: 14px;
            }}
            input, button {{
                font-size: 16px;
                padding: 6px;
                margin: 4px;
                border-radius: 6px;
                border: 1px solid #ccc;
            }}
            button {{
                background-color: #007bff;
                color: white;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <h2>{ticker} Stock Snapshot</h2>
        <form method="get">
            <input type="text" name="ticker" placeholder="Enter ticker (e.g. AAPL)" value="{ticker}" />
            <button type="submit">Go</button>
        </form>
        <img src="data:image/png;base64,{plot_url}" alt="Stock Chart" width="500"/>
        <p><b>Market Recap:</b> {comment}</p>
        {"<p class='warning'>" + headlines + "</p>" if headlines.startswith("⚠️") else ""}
    </body>
    </html>
    """

    return Response(html, mimetype="text/html")

# ------------------------------
# 4. Run app
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)