import streamlit as st
import streamlit as st
import requests
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


# ✅ DEFINE api_key FIRST
api_key = st.secrets["OPENROUTER_API_KEY"]

# ✅ THEN create client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# ---------------- DATA ---------------- #


def get_market_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    data = requests.get(url).json()

    return {
        "price": data["market_data"]["current_price"]["usd"],
        "market_cap": data["market_data"]["market_cap"]["usd"],
        "change_24h": data["market_data"]["price_change_percentage_24h"],
    }


def get_price_history(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    data = requests.get(url).json()

    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


# ---------------- MODELS ---------------- #

MODELS = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]

# ---------------- ANALYSIS ---------------- #


def get_analysis(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a crypto analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# ---------------- JUDGE ---------------- #


def judge_best(responses):
    combined = "\n\n".join([f"Response {i+1}:\n{r}" for i, r in enumerate(responses)])

    judge_prompt = f"""
Evaluate each response based on:
- Accuracy (0-10)
- Clarity (0-10)
- Insight (0-10)

Then select best.

Return full scoring + best response.

{combined}
"""

    result = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
    )

    return result.choices[0].message.content


# ---------------- UI ---------------- #

st.set_page_config(page_title="AI Crypto Agent", layout="wide")

st.title("🚀 AI Crypto Analysis Agent")

coin = st.selectbox("Select Coin", ["bitcoin", "ethereum", "solana"])
timeframe = st.selectbox("Select Timeframe", [7, 30, 90], index=1)  # default 30
user_input = st.text_input("Ask something", "Should I buy this now?")

if st.button("Analyze"):
    with st.spinner("🔍 Analyzing crypto... please wait"):
        data = get_market_data(coin)

        prompt = f"""
Analyze {coin.upper()}:

Price: {data['price']}
Market Cap: {data['market_cap']}
24h Change: {data['change_24h']}

User Request: {user_input}

Give:
- Trend
- Risk
- Short-term
- Long-term
- Recommendation
"""

        st.subheader("📊 Market Data")

        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"${data['price']}")
        col2.metric("Market Cap", f"${data['market_cap']}")
        col3.metric("24h Change", f"{data['change_24h']}%")

        st.subheader(f"📈 Price Chart ({timeframe} Days)")

        df = get_price_history(coin, timeframe)
        st.line_chart(df.set_index("timestamp"))

        responses = []

        st.subheader("🤖 Model Outputs")

        for model in MODELS:
            st.write(f"### {model}")
            res = get_analysis(model, prompt)
            st.write(res)
            responses.append(res)

        st.subheader("⚖️ Final Decision")

        final = judge_best(responses)
        st.write(final)
