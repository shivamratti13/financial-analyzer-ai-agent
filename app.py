import os
import requests
from dotenv import load_dotenv
import streamlit as st
import yfinance as yf
from stocks_to_ticker import us_stocks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt

from langchain.agents import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.callbacks import StreamlitCallbackHandler

import alpaca_trade_api as tradeapi

load_dotenv()

st.title('Autonomous Stock & News Analyzer')
st.caption("An AI agent that analyzes stocks using news, sentiment, and financial data.")

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

@st.cache_data
def get_stock_data(ticker_symbol: str):
    stock = yf.Ticker(ticker_symbol)
    hist_data = stock.history(period="1mo")
    return hist_data

@st.cache_data
def get_news_articles(query: str):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    titles = [f"- {a['title']}" for a in data.get('articles', [])]
    return "\n".join(titles)

@st.cache_resource # Use cache_resource for models
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment(text: str) -> str:
    tokenizer, model = load_finbert_model()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    prediction_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive = prediction_probs[0][0].item()
    negative = prediction_probs[0][1].item()
    neutral = prediction_probs[0][2].item()
    return f"Sentiment Analysis: Positive: {positive:.2f}, Negative: {negative:.2f}, Neutral: {neutral:.2f}"

# We need to wrap the agent tools for Streamlit's context
@tool
def get_stock_data_tool(ticker_symbol: str) -> str:
    """
    Use this tool to get the latest stock price data for a company.
    The input should be the company's ticker symbol (e.g., AAPL, GOOGL).
    """
    return get_stock_data(ticker_symbol)
@tool
def get_news_articles_tool(query: str) -> str:
    """
    Use this tool to get recent news articles for a company or topic.
    The input should be the company name or a search query (e.g., Apple Inc, Tesla electric vehicles).
    """
    return get_news_articles(query)
@tool
def analyze_sentiment_tool(text: str) -> str:
    """
    Use this tool to analyze the sentiment of a piece of text (like news headlines).
    The input should be the text you want to analyze.
    """
    return analyze_sentiment(text)

@st.cache_resource
def get_alpaca_api():
    return tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )

def get_position(api, ticker):
    try:
        position = api.get_position(ticker)
        return float(position.qty)
    except tradeapi.rest.APIError:
        return 0

def execute_paper_trade(api, ticker, qty, side):
    try:
        order = api.submit_order(symbol=ticker, qty=qty, side=side, type='market', time_in_force='day')
        st.success(f"✅ Success! Order to {side} {qty} share(s) of {ticker} submitted.")
        st.write(order)
    except Exception as e:
        st.error(f"❌ Error executing trade: {e}")

# --- 3. SESSION STATE INITIALIZATION ---
# Initialize session state variables to store data across reruns
if 'report' not in st.session_state:
    st.session_state.report = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""

company_name = st.selectbox(
    'Select a Company for Analysis',
    options=us_stocks,
)

if st.button('Go'):
    df = get_stock_data(us_stocks[company_name])
    st.write(f"📈 Showing last 10 days of data for {company_name} ({us_stocks[company_name]})")
    st.dataframe(df.tail(10)) 

    news = get_news_articles(company_name)
    st.write(f"📈 Showing recent 5 most relevent News for {company_name} ({us_stocks[company_name]})")
    st.write(news)

    with st.spinner(f"🤖 The AI agent is analyzing {company_name} ({us_stocks[company_name]})... This may take a moment."):
        st_callback_container = st.expander("Show Agent's Thought Process")
        with st_callback_container:
            st_callback = StreamlitCallbackHandler(st_callback_container)
        
        # Setup Agent
        tools = [get_stock_data_tool, get_news_articles_tool, analyze_sentiment_tool]
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv('GEMINI_API_KEY'), temperature=0)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        report_template = f"""
            Generate a stock analysis report for {company_name} ({us_stocks[company_name]}).
            Your final answer MUST be structured in the following format. Use the tools to find the information for each section:

            # Stock Analysis Report: {company_name} ({us_stocks[company_name]})
            **Date:** {dt.date.today().strftime('%Y-%m-%d')}

            ## 1. Overall Recommendation
            (Your single-word recommendation: **STRONG BUY**, **MONITOR**, or **AVOID**. Base this on the news and sentiment.)

            ## 2. Key Price Metrics
            (Use the get_stock_data tool to report the last closing price and briefly describe the recent 5-day trend.)

            ## 3. News & Sentiment Summary
            (Use get_news_articles and analyze_sentiment to create a 2-3 sentence summary of the key news themes and the overall sentiment.)

            ## 4. Top Headlines
            (List the top 3-4 most relevant news headlines you found using the get_news_articles tool.)

            ---
            *Disclaimer: This is an AI-generated report for informational purposes only and does not constitute financial advice.*
            """

        # Run the agent
        response = agent_executor.invoke(
                {"input": report_template},
                {"callbacks": [st_callback]} # Pass the callback handler here
            )
        st.session_state.report = response['output']
        st.success("Analysis complete!")

# --- 5. REPORT DISPLAY & TRADING INTERFACE ---
if st.session_state.report:
    st.divider()
    st.header("2. AI-Generated Report")
    st.markdown(st.session_state.report)

    st.header("3. Take Action")
    st.caption("Confirm any trades based on the AI's recommendation.")

    api = get_alpaca_api()
    recommendation = ""
    if "**STRONG BUY**" in st.session_state.report:
        recommendation = "STRONG BUY"
    elif "**AVOID**" in st.session_state.report:
        recommendation = "AVOID"

    # Create two columns for Buy and Sell actions
    col1, col2 = st.columns(2)

    # --- BUY INTERFACE ---
    with col1:
        st.subheader("Buy Shares")
        if recommendation == "STRONG BUY":
            st.info("AI Recommendation: **STRONG BUY**")
            buy_qty = st.number_input("Quantity to Buy:", min_value=1, step=1, key="buy_qty")
            if st.button("Execute Buy"):
                execute_paper_trade(api, us_stocks[company_name], buy_qty, 'buy')
        else:
            st.info("No 'STRONG BUY' recommendation from the AI.")

    # --- SELL INTERFACE ---
    with col2:
        st.subheader("Sell Shares")
        owned_qty = get_position(api, us_stocks[company_name])
        st.write(f"You currently own: **{owned_qty}** shares.")

        if recommendation == "AVOID":
            st.warning("AI Recommendation: **AVOID**")
            if owned_qty > 0:
                sell_qty = st.number_input("Quantity to Sell:", min_value=1, max_value=int(owned_qty), step=1, key="sell_qty")
                if st.button("Execute Sell"):
                    execute_paper_trade(api, us_stocks[company_name], sell_qty, 'sell')
            else:
                st.info("You do not own any shares to sell.")
        else:
            st.info("No 'AVOID' recommendation from the AI.")