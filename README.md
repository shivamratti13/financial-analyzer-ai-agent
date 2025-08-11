# 🤖 Your Personal AI Stock Analyst

A Streamlit-based dashboard that utilizes a LangChain agent for real-time stock analysis, news aggregation, sentiment analysis, and simulated trading.
A brief demonstration of the application in action.

---

## 📖 What's This All About?

Ever wish you had a personal assistant to help you make sense of the stock market? That's exactly what this project is.  
It's a simple webpage that does the heavy lifting of stock market research for you.

At its heart is a clever AI agent built using **LangChain**.  
You can give this agent a stock ticker, and it will go to work, using its special tools to:

- 📈 Check the latest stock prices using **yFinance**
- 📰 Read the news from thousands of sources with the **News API**
- 🧠 Get the vibe of the news (good, bad, or neutral) using **FinBERT**
- 📝 Think it all over to give you a **straightforward recommendation**
- 💹 Even practice trading by connecting to an **Alpaca** paper-trading account

The whole thing is wrapped in a **Streamlit** web app so you don’t need to be a tech wizard to use it.

---

## ✨ Cool Features

- **AI-Powered Reports:** Just type in a stock symbol (like `AAPL`) and get a full AI-generated report.
- **Fresh Data:** Pulls stock prices and news in real-time.
- **Know the Mood:** Understand instantly if the chatter around a stock is positive or negative.
- **See the AI's Brain:** Watch the AI's step-by-step thought process.
- **Practice Trading:** Safely try out buying or selling with virtual money.
- **Safe & Secure:** API keys are stored privately and never uploaded to GitHub.

---

## 🛠️ The Tech Behind the Magic

- **Frontend:** Streamlit
- **AI Brain:** LangChain
- **Language Model:** Google Gemini Pro
- **Data Sources:** yFinance, News API
- **Sentiment Analysis:** FinBERT (Hugging Face)
- **Paper Trading:** Alpaca Trade API
- **Language:** Python

---

## 🚀 How to Get It Running

### What You'll Need

- Python **3.8+**
- GitHub account
- Free API keys:
  - **Google AI** (for AI analysis)
  - **News API** (for latest news)
  - **Alpaca** (for paper trading)

---

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name
