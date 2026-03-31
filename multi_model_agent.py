import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

# ---------------- DATA TOOL ---------------- #


def get_market_data(coin="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}"
    data = requests.get(url).json()

    return {
        "price": data["market_data"]["current_price"]["usd"],
        "market_cap": data["market_data"]["market_cap"]["usd"],
        "change_24h": data["market_data"]["price_change_percentage_24h"],
    }


# ---------------- INPUT HANDLING ---------------- #


def extract_coin(user_input):
    coins = {
        "bitcoin": "bitcoin",
        "btc": "bitcoin",
        "ethereum": "ethereum",
        "eth": "ethereum",
        "solana": "solana",
        "sol": "solana",
    }

    user_input = user_input.lower()

    for key in coins:
        if key in user_input:
            return coins[key]

    return "bitcoin"  # default fallback


# ---------------- MODELS ---------------- #

MODELS = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]

# ---------------- ANALYSIS ---------------- #


def get_analysis(model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional crypto analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ---------------- JUDGE ---------------- #


def judge_best(responses):
    combined = "\n\n".join([f"Response {i+1}:\n{r}" for i, r in enumerate(responses)])

    judge_prompt = f"""
You are an expert crypto analyst.

Evaluate each response based on:

1. Accuracy (0-10)
2. Clarity (0-10)
3. Insight (0-10)

Instructions:
- Give scores for EACH response
- Then compute TOTAL score
- Select the BEST response
- Return in this format:

Response 1 Score:
Accuracy: X/10
Clarity: X/10
Insight: X/10
Total: X/30

Response 2 Score:
...

FINAL BEST RESPONSE:
<PASTE FULL BEST RESPONSE HERE>

IMPORTANT:
- You MUST include the full best response
- Do NOT just say "Response 1"
- Be objective

{combined}
"""

    result = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
    )

    return result.choices[0].message.content.strip()


# ---------------- SAVE ---------------- #


def save_report(text):
    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- MAIN AGENT ---------------- #


def run_agent(user_input):
    coin = extract_coin(user_input)
    data = get_market_data(coin)

    prompt = f"""
You are a professional crypto analyst.

Analyze {coin.upper()} using the data below.

Data:
- Price: {data['price']}
- Market Cap: {data['market_cap']}
- 24h Change: {data['change_24h']}

User Request: {user_input}

Give structured output:

1. Trend (Bullish/Bearish/Neutral)
2. Risk Level (Low/Medium/High)
3. Short-term Outlook
4. Long-term Outlook
5. Final Recommendation (Buy/Hold/Wait)

Be clear and concise.
"""

    responses = []

    print("\n🔍 Running multi-model analysis...\n")

    for model in MODELS:
        print(f"⚡ Using {model}")
        try:
            res = get_analysis(model, prompt)
            responses.append(res)
        except Exception as e:
            print(f"❌ {model} failed: {e}")

    if not responses:
        print("❌ All models failed.")
        return

    print("\n⚖️ Selecting best response...\n")
    best = judge_best(responses)

    save_report(best)

    print("✅ Final Selected Analysis:\n")
    print(best)


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    print("\n💡 Examples:")
    print("- Analyze Bitcoin")
    print("- Should I buy Ethereum now?")
    print("- Analyze Solana long term\n")

    user_input = input("👉 Enter your request: ")
    run_agent(user_input)
