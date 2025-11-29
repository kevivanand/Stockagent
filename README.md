# StockAgent: NIFTY Options Trading Simulation

**StockAgent** is an LLM-based multi-agent simulation system designed to model investor behaviors in the **NIFTY 50** options market. Unlike traditional backtesters, it uses AI agents (powered by Gemini or GPT) to make trading decisions based on real market data, simulated news/forum discussions, and their own risk profiles.

## Key Features

-   **Real Market Data**: Automatically fetches the last 30 days of **NIFTY 50 (`^NSEI`)** data using `yfinance`.
-   **Options Simulation**: Simulates a dynamic Option Chain (Calls/Puts) for NIFTY using the Black-Scholes model, allowing agents to trade derivatives without needing expensive real-time options data feeds.
-   **AI Agents**: 50+ autonomous agents with distinct personalities (Aggressive, Conservative, Balanced) that:
    -   Analyze market trends.
    -   Participate in a simulated forum.
    -   Decide on **Loans** (Leverage/Margin).
    -   Trade **NIFTY Stocks** and **Options**.
-   **Secure Configuration**: Uses `.env` for API key management.

## Quick Start

### 1. Prerequisites

-   Python 3.9+
-   A Google Gemini API Key (recommended) or OpenAI API Key.

### 2. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repository_url>
cd Stockagent

# Install dependencies
pip install -r requirements.txt
pip install yfinance pandas numpy scipy colorama tiktoken python-dotenv
```

### 3. Configuration

Create a `.env` file in the root directory to store your API key securely:

```bash
# Create .env file
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

*(Alternatively, you can use `OPENAI_API_KEY` if you prefer GPT models)*

### 4. Running the Simulation

Run the main simulation script. By default, it uses `gemini-2.0-flash`.

```bash
python3 main.py
```

To use a different model:

```bash
python3 main.py --model gemini-1.5-flash
# or
python3 main.py --model gpt-4
```

### 5. Output

The simulation results are saved in the `res/` directory:

-   `res/trades.xlsx`: Log of all executed trades (Buy/Sell).
-   `res/stocks.xlsx`: Daily price history of NIFTY and Options.
-   `res/agent_day_record.xlsx`: Daily summary of agent decisions (Loans, Estimates).
-   `res/agent_session_record.xlsx`: Detailed actions per trading session.

## Architecture

1.  **DataLoader**: Fetches historical NIFTY data and generates simulated option chains.
2.  **Agent**: LLM-driven entity that observes the market, chats in the forum, and makes trading decisions.
3.  **Secretary**: Validates Agent actions to ensure they conform to market rules (e.g., sufficient cash, valid symbols).
4.  **Main Loop**: Iterates through the fetched dates, updating prices and processing agent actions.

## License

[MIT License](LICENSE)
