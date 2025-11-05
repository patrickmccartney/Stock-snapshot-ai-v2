import io, base64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
from flask import Flask, render_template_string, request, Response
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()
plt.switch_backend("Agg")

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>{{ ticker }} Snapshot</title>
  <style>
    body { font-family: system-ui, sans-serif; text-align:center; margin:2em; }
    img { max-width:100%%; border-radius:12px; box-shadow:0 0 10px #ccc; }
    input { padding:0.5em; font-size:1em; border-radius:6px; border:1px solid #aaa; }
    button { padding:0.5em 1em; font-size:1em; border:none; border-radius:6px; background:#007bff; color:white; cursor:pointer; }
    .comment { margin-top:1em; font-size:1.1em; }
  </style>
</head>
<body>
  <h1>{{ ticker }} Stock Snapshot</h1>
  <form action="/" method="get">
    <input type="text" name="symbol" placeholder="Enter ticker (e.g. AAPL)" required>
    <button type="submit">Go</button>
  </form>
  {% if chart %}
    <img src="data:image/png;base64,{{ chart }}" alt="Chart">
    <div class="comment">{{ comment }}</div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    ticker = request.args.get("symbol", "").upper()
    if not ticker:
        return render_template_string(TEMPLATE, ticker="Enter a symbol", chart=None, comment=None)

    # Download last 5 days of intraday prices
    data = yf.download(ticker, period="5d", interval="1h")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or "Close" not in data:
        return render_template_string(TEMPLATE, ticker=f"{ticker} (no data)", chart=None, comment=None)

    # Plot the chart
    plt.figure(figsize=(6,3))
    plt.plot(data.index, data["Close"], color="blue")
    plt.title(f"{ticker} - Last 5 Days")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Compute movement stats
    first_close = float(data["Close"].iloc[0])
    last_close = float(data["Close"].iloc[-1])
    change = round(last_close - first_close, 2)
    pct_change = (last_close - first_close) / first_close * 100

    # Smarter financial-style prompt
    prompt = f"""
You are a financial analyst writing a concise 2–3 sentence daily note.

Ticker: {ticker}
Price moved from ${first_close:.2f} to ${last_close:.2f} over the past 5 days ({pct_change:+.2f}% change).

Task:
- Identify notable news or events from the past 5 days that likely influenced {ticker}'s share price.
- If earnings occurred, summarize investor reaction.
- Include possible macro or peer factors.
- End with a short analytical takeaway.

Use a factual, market-commentary tone.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        comment = response.choices[0].message.content.strip()
    except Exception as e:
        comment = f"(AI summary unavailable: {e})"

    html = render_template_string(TEMPLATE, ticker=ticker, chart=chart_base64, comment=comment)
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)