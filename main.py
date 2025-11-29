import argparse
import random
from datetime import datetime, timedelta
from typing import Optional

import util
from agent import Agent
from secretary import Secretary
from stock import Instrument, Stock, Option
from data_loader import DataLoader
from log.custom_logger import log
from record import create_stock_record, create_trade_record, AgentRecordDaily, create_agentses_record

def get_agent(all_agents, order) -> Optional[Agent]:
    for agent in all_agents:
        if agent.order == order:
            return agent
    return None

def handle_action(action, deals, all_agents, instruments_dict, session):
    # action = JSON{"agent": 1, "action_type": "buy"|"sell", "stock": "SYMBOL", "amount": 10, "price": 10}
    acting_agent = get_agent(all_agents, action["agent"])
    if acting_agent is None:
        log.logger.error("handle_action error: agent %s not found", action["agent"])
        return
    
    stock_name = action["stock"]
    if stock_name not in deals:
        deals[stock_name] = {"buy": [], "sell": []}
    
    stock_deals = deals[stock_name]
    instrument = instruments_dict.get(stock_name)
    
    try:
        if action["action_type"] == "buy":
            for sell_action in stock_deals["sell"][:]:
                if action["price"] >= sell_action["price"]: # Match if buy price >= sell price
                    # 交易成交
                    close_amount = min(action["amount"], sell_action["amount"])
                    acting_agent.buy_stock(stock_name, close_amount, action["price"])
                    seller_order = sell_action["agent"]
                    if seller_order != -1:  # Not market maker
                        seller_agent = get_agent(all_agents, seller_order)
                        if seller_agent is None:
                            log.logger.error("handle_action error: seller agent %s not found", seller_order)
                            stock_deals["sell"].remove(sell_action)
                            continue
                        seller_agent.sell_stock(stock_name, close_amount, action["price"])
                    
                    if instrument:
                        instrument.add_session_deal({"price": action["price"], "amount": close_amount})
                    
                    create_trade_record(action["date"], session, stock_name, action["agent"], sell_action["agent"],
                                        close_amount, action["price"])

                    if action["amount"] > close_amount:  # Buy not finished
                        log.logger.info(f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                                        f"STOCK:{stock_name}, PRICE:{action['price']}, AMOUNT:{close_amount}")
                        stock_deals["sell"].remove(sell_action)
                        action["amount"] -= close_amount
                    else:  # Sell not finished
                        log.logger.info(f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                                        f"STOCK:{stock_name}, PRICE:{action['price']}, AMOUNT:{close_amount}")
                        sell_action["amount"] -= close_amount
                        return
            # Add remaining buy order
            stock_deals["buy"].append(action)

        else: # Sell
            for buy_action in stock_deals["buy"][:]:
                if action["price"] <= buy_action["price"]: # Match if sell price <= buy price
                    # 交易成交
                    close_amount = min(action["amount"], buy_action["amount"])
                    acting_agent.sell_stock(stock_name, close_amount, action["price"])
                    buyer_agent = get_agent(all_agents, buy_action["agent"])
                    if buyer_agent is None:
                        log.logger.error("handle_action error: buyer agent %s not found", buy_action["agent"])
                        stock_deals["buy"].remove(buy_action)
                        continue
                    buyer_agent.buy_stock(stock_name, close_amount, action["price"])
                    
                    if instrument:
                        instrument.add_session_deal({"price": action["price"], "amount": close_amount})
                        
                    create_trade_record(action["date"], session, stock_name, buy_action["agent"], action["agent"],
                                        close_amount, action["price"])

                    if action["amount"] > close_amount:  # Sell not finished
                        log.logger.info(f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                                        f"STOCK:{stock_name}, PRICE:{action['price']}, AMOUNT:{close_amount}")
                        stock_deals["buy"].remove(buy_action)
                        action["amount"] -= close_amount
                    else:  # Buy not finished
                        log.logger.info(f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                                        f"STOCK:{stock_name}, PRICE:{action['price']}, AMOUNT:{close_amount}")
                        buy_action["amount"] -= close_amount
                        return
            stock_deals["sell"].append(action)
    except Exception as e:
        log.logger.error(f"handle_action error: {e}")
        return


def simulation(args):
    # init
    secretary = Secretary(args.model)
    loader = DataLoader()
    
    # Fetch NIFTY data for the last 30 days to assess current market context
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Format dates for yfinance (YYYY-MM-DD)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    try:
        nifty_data = loader.fetch_nifty_data(start_date_str, end_date_str)
    except Exception as e:
        log.logger.error(f"Failed to fetch data: {e}")
        return

    dates = nifty_data.index.tolist()
    
    # Initialize instruments
    # We will update their prices daily
    nifty_stock = Stock("NIFTY", float(nifty_data.iloc[0]["Close"]))
    
    # Initial options (ATM)
    options_data = loader.generate_option_chain(nifty_stock.get_price(), dates[0])
    options = []
    for opt in options_data:
        options.append(Option(opt["symbol"], opt["price"], opt["strike"], opt["expiry"], opt["type"], opt["underlying"]))
    
    instruments = [nifty_stock] + options
    instruments_dict = {inst.symbol: inst for inst in instruments}

    all_agents = []
    log.logger.debug("Agents initial...")
    for i in range(0, util.AGENTS_NUM):  # agents start from 0, -1 refers to admin
        agent = Agent(i, instruments, secretary, args.model)
        all_agents.append(agent)
        log.logger.debug("cash: {}, holdings: {}, debt: {}".format(agent.cash, agent.holdings, agent.loans))

    # start simulation
    last_day_forum_message = []
    deals = {} # {symbol: {buy: [], sell: []}}

    log.logger.debug("--------Simulation Start!--------")
    
    for date_idx, current_date in enumerate(dates):
        date_str = current_date.strftime("%Y-%m-%d")
        log.logger.debug(f"--------DAY {date_str}---------")
        
        # Update prices
        current_price = float(nifty_data.iloc[date_idx]["Close"])
        nifty_stock.price = current_price
        
        # Update option prices (simulate)
        new_options_data = loader.generate_option_chain(current_price, current_date)
        # For simplicity in this demo, we just update existing options if they match, or add new ones?
        # To keep it simple, let's just update the prices of the options we track.
        # Realistically options expire and new ones are added. 
        # For this short demo, we stick to the initial set or update them.
        # Let's just update the prices of existing options based on new underlying price.
        # We re-calculate price for existing options.
        for opt in options:
            # Re-calc price using Black-Scholes with new underlying price and less time to expiry
            # This logic is inside loader.generate_option_chain but we need to call it for specific strikes.
            # Simplified: Just re-generate chain and match symbols.
            for new_opt in new_options_data:
                if new_opt["symbol"] == opt.symbol:
                    opt.price = new_opt["price"]
                    break

        # Clear deals
        deals = {}

        # check if an agent needs to repay loans
        # Map date_idx to simulation days for loan logic (simplified)
        sim_day = date_idx + 1
        
        for agent in all_agents[:]:
            agent.chat_history.clear()  # 只保存当天的聊天记录
            agent.loan_repayment(sim_day)

        # repayment days
        if sim_day in util.REPAYMENT_DAYS:
            for agent in all_agents[:]:
                agent.interest_payment()

        # deal with cash<0 agents
        for agent in all_agents[:]:
            if agent.is_bankrupt:
                quit_sig = agent.bankrupt_process()
                if quit_sig:
                    agent.quit = True
                    all_agents.remove(agent)

        # agent decide whether to loan
        daily_agent_records = []
        for agent in all_agents:
            loan = agent.plan_loan(sim_day, last_day_forum_message)
            daily_record = AgentRecordDaily(agent.order, sim_day, loan)
            daily_record.write_to_excel() # Write immediately
            daily_agent_records.append(daily_record)

        for session in range(1, util.TOTAL_SESSION + 1):
            log.logger.debug(f"SESSION {session}")
            # 随机定义交易顺序
            sequence = list(range(len(all_agents)))
            random.shuffle(sequence)
            for i in sequence:
                agent = all_agents[i]

                action = agent.plan_stock(sim_day, session, deals)
                proper, cash, stock_values = agent.get_proper_cash_value()
                # create_agentses_record needs update or we skip it for now?
                # It expects valua_a, value_b. We might break it.
                # Let's skip detailed recording for now or pass dummy values.
                # create_agentses_record(agent.order, sim_day, session, proper, cash, 0, 0, action)
                
                action["agent"] = agent.order
                action["date"] = sim_day
                if not action["action_type"] == "no":
                    handle_action(action, deals, all_agents, instruments_dict, session)

            # 交易时段结束，更新股票价格 (In simulation we update at start of day from real data, 
            # but here we could update based on trading? No, we are price takers from NIFTY data)
            # So we don't update price based on agent actions for NIFTY.
            # create_stock_record(sim_day, session, nifty_stock.get_price(), 0)

        # agent预测明天行动
        for idx, agent in enumerate(all_agents):
            estimation = agent.next_day_estimate()
            log.logger.info("Agent {} tomorrow estimation: {}".format(agent.order, estimation))
            if idx >= len(daily_agent_records):
                break
            daily_agent_records[idx].add_estimate(estimation)
            # daily_agent_records[idx].write_to_excel() # Skip excel for now
        daily_agent_records.clear()

        # 交易日结束，论坛信息更新
        last_day_forum_message.clear()
        log.logger.debug(f"DAY {date_str} ends, display forum messages...")
        for agent in all_agents:
            message = agent.post_message()
            log.logger.info("Agent {} says: {}".format(agent.order, message))
            last_day_forum_message.append({"name": agent.order, "message": message})

    log.logger.debug("--------Simulation finished!--------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="model name")
    args = parser.parse_args()
    simulation(args)
