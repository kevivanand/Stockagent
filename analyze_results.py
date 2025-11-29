import pandas as pd
import os

def analyze():
    print("\n" + "="*50)
    print("STOCKAGENT: MARKET ANALYSIS REPORT")
    print("="*50 + "\n")

    # 1. Analyze Sentiment (Loans)
    loan_file = "res/agent_day_record.xlsx"
    if os.path.exists(loan_file):
        try:
            df_loan = pd.read_excel(loan_file)
            total_agents = len(df_loan)
            loans_taken = df_loan[df_loan['Loan Taken?'] == 'yes'].shape[0]
            sentiment_score = (loans_taken / total_agents) * 100 if total_agents > 0 else 0
            
            print(f"1. MARKET SENTIMENT (Risk Appetite)")
            print(f"   - Agents Taking Leverage: {loans_taken}/{total_agents} ({sentiment_score:.1f}%)")
            
            if sentiment_score > 60:
                print("   - Verdict: BULLISH (High conviction, agents are leveraging up)")
            elif sentiment_score < 40:
                print("   - Verdict: BEARISH/CAUTIOUS (Agents are avoiding risk)")
            else:
                print("   - Verdict: NEUTRAL (Mixed signals)")
        except Exception as e:
            print(f"   - Error reading loan data: {e}")
    else:
        print("   - No loan data found yet.")

    print("\n" + "-"*30 + "\n")

    # 2. Analyze Trades (if any)
    trade_file = "res/trades.xlsx"
    if os.path.exists(trade_file):
        try:
            df_trade = pd.read_excel(trade_file)
            if not df_trade.empty:
                print(f"2. TRADING ACTIVITY")
                print(f"   - Total Trades Executed: {len(df_trade)}")
                
                # Most traded stock/option
                top_traded = df_trade['Stock Type'].value_counts().idxmax()
                print(f"   - Most Traded Instrument: {top_traded}")
                
                # Buy vs Sell pressure
                # This requires looking at who initiated, but simplified:
                print(f"   - Recent Activity: See res/trades.xlsx for details.")
            else:
                print("2. TRADING ACTIVITY")
                print("   - No trades executed yet (Agents might be waiting for better prices).")
        except Exception as e:
            print(f"   - Error reading trade data: {e}")
    else:
        print("2. TRADING ACTIVITY")
        print("   - No trade data found.")

    print("\n" + "="*50)
    print("RECOMMENDATION:")
    if 'sentiment_score' in locals():
        if sentiment_score > 60:
            print(">> CONSIDER LONG POSITIONS (Calls / Bull Spreads)")
        elif sentiment_score < 40:
            print(">> CONSIDER SHORT POSITIONS (Puts / Cash)")
        else:
            print(">> WAIT AND WATCH / IRON CONDOR")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze()
