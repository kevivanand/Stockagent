from procoder.prompt import *

# BACKGROUND_PROMPT = NamedBlock(
#     name="Background",
#     content="""
#     你是一名股票交易员，接下来你将在市场中模拟与其他交易员的交互。市场中一共有两支股票，分别为A和B，其中B为新上市的股票。
#     接下来，请根据指令完成你的交易行动。
#     """
# )

BACKGROUND_PROMPT = NamedBlock(
    name="Background",
    content="""
        You are a stock trader, and next you will simulate interactions with other traders in the market.
        There are multiple instruments in the market.
        Next, please complete your trading actions according to the order.
    """
)

# LASTDAY_FORUM_AND_STOCK_PROMPT = NamedBlock(
#     name="Last Day Forum and Stock",
#     content="""
#     昨天交易截止后，A公司股票和B公司股票的股价分别是{stock_a_price}元/股和{stock_b_price}元/股。
#     其他交易员在论坛上发布的帖子如下：
#     {lastday_forum_message}
#     """
# )

LASTDAY_FORUM_AND_STOCK_PROMPT = NamedBlock(
    name="Last Day Forum and Stock",
    content="""
        After the close of trading yesterday, the prices were: {stock_prices}.
        Posts by other traders on the forum are as follows: {lastday_forum_message}
    """
)

# LOAN_TYPE_PROMPT = NamedVariable(
#     refname="loan_type_prompt",
#     name="Loan Type",
#     content="""
#     0. 1年期，基准利率{loan_rate1}
#     1. 2年期，基准利率{loan_rate2}
#     2. 3年期，基准利率{loan_rate3}
#     """
# )

LOAN_TYPE_PROMPT = NamedVariable(
    refname="loan_type_prompt",
    name="Loan Type",
    content="""
    0. 22days, the benchmark interest rate {loan_rate1}
    1. 44days, the benchmark interest rate {loan_rate2}
    2. 66days, the benchmark interest rate {loan_rate3}
    """
)

# DECIDE_IF_LOAN_PROMPT = NamedBlock(
#     name="Instruction",
#     content="""
#     现在是第{date}天，你当前的性格是{character}，持有{stock_a}股A公司股票，持有{stock_b}股B公司股票，
#     现在你有{cash}元现金，贷款情况为{debt}。
#     你需要决定是否继续贷款和贷款金额。
#     可供选择的种类为{loan_type_prompt}，你应当用编号选择一个贷款种类。贷款金额不得超过{max_loan}。
#     用json形式返回结果，例如：
#     {{"loan": "yes", "loan_type": 3, "amount": 1000}}
#     如果不需贷款，则返回：
#     {{"loan" : "no"}}
#     """
# )

DECIDE_IF_LOAN_PROMPT = NamedBlock(
    name="Instruction",
    content="""
    It is the {date} day, and your current character is {character}. 
    You hold {holdings}.
    Now you have {cash} dollars in cash and {debt} in your loan situation.
    You need to decide whether to continue the loan and the amount of the loan.
    The alternative type is {loan_type_prompt}, and you should use the number to select a loan type. 
    The loan amount shall not exceed {max_loan}.

    Return the result as json, for example:
    {{"loan": "yes", "loan_type": 3, "amount": 1000}}

    If no loan is required, return:
    {{"loan" : "no"}}
    """
)

# LOAN_RETRY_PROMPT = NamedBlock(
#     name="Instruction",
#     content="""
#     The following questions appeared in the loan format you last answered: {fail_response}.
#     你应当用json形式返回结果，例如：
#     {{"loan": "yes", "loan_type": 2, "amount": 1000}}
#     如果不需贷款，则返回：
#     {{"loan" : "no"}}
#     Please answer again."""
# )

LOAN_RETRY_PROMPT = NamedBlock(
    name="Instruction",
    content="""
    The following questions appeared in the loan format you last answered: {fail_response}.
    You should return the results as json, for example:
    {{"loan": "yes", "loan_type": 2, "amount": 1000}}
    If no loan is required, return:
    {{"loan" : "no"}}
    Please answer again."""
)

# DECIDE_BUY_STOCK_PROMPT = NamedBlock(
#     name="Instruction",
#     content="""
#     现在是第{date}天的{time}交易时段，前一时段结束后，A公司的股票股价为{stock_a_price}，B公司的股票股价为{stock_b_price}。
#     在目前时段，股票A的买卖盘为{stock_a_deals}，股票B的买卖盘为{stock_b_deals}
#     你当前持有{stock_a}股A公司股票，持有{stock_b}股B公司股票，{cash}元现金。
#     你需要决定是否购买/卖出A公司或B公司的股票，以及购买/卖出的数量与价格。你可以参考当前股价和大盘自己决定价格，无需确定为当前股价。数量必须为整数。
#     鼓励尽可能多地买入和卖出。
#     用json形式返回结果，例如：
#     {{"action_type":"buy"|"sell", "stock":"A"|"B", amount: 100, price : 30}}
#     如果既不购买也不卖出，则返回：
#     {{"action_type" : "no"}}"""
# )

DECIDE_BUY_STOCK_PROMPT = NamedBlock(
    name="Instruction",
    content="""
    It is the {time} trading session on the {date} day, and after the previous session, 
    the prices are: {stock_prices}.
    In the current session, the deals are: {deals}
    You currently hold {holdings}, and {cash} yuan in cash.
    You need to decide whether to buy/sell instruments, and how much to buy/sell and at what price.
    You can refer to the current share price and the market to determine the price yourself, not the current share price. 
    The quantity must be an integer.
    We encourage you to buy and sell more. You can only answer one json action.
    Return the result as json, for example:
    {{"action_type":"buy"|"sell", "stock":"SYMBOL", amount: 100, price : 30.1}}
    If neither buy nor sell, return:
    {{"action_type" : "no"}}
    """
)

# BUY_STOCK_RETRY_PROMPT = NamedBlock(
#     name="Instruction",
#     content="""
#     The following questions appeared in the action format you last answered: {fail_response}.
#     你应当用json形式返回结果，例如：
#     {{"action_type":"buy"|"sell", "stock":"A"|"B", amount: 100, price: 30}}
#     如果既不购买也不卖出，则返回：
#     {{"action_type" : "no"}}
#     Please answer again."""
# )

BUY_STOCK_RETRY_PROMPT = NamedBlock(
    name="Instruction",
    content="""
    The following questions appeared in the action format you last answered: {fail_response}.
    You should return the result as json, for example:
    {{"action_type":"buy"|"sell", "stock":"SYMBOL", amount: 100, price: 30.1}}
    If neither buy nor sell, return:
    {{"action_type" : "no"}}
    Please answer again. You can only answer one json action.
    """
)

# FIRST_DAY_FINANCIAL_REPORT = NamedVariable(
#     refname="first_day_financial_report",
#     name="The initial financial situation of Stock A and B",
#     content="""
#     ●公司A：这只股票超级棒！
#     ●公司B：这只股票风险大收益大！"""
# )

FIRST_DAY_FINANCIAL_REPORT = NamedVariable(
    refname="first_day_financial_prompt",
    name="Recent Market Context",
    content="""
    The market has been volatile recently.
    NIFTY 50 has shown resilience but faces global headwinds.
    Options volatility (IV) is moderate.
    """
)

FIRST_DAY_BACKGROUND_KNOWLEDGE = NamedBlock(
    name="Market Background",
    content="""
    You are trading the NIFTY 50 index and its options.
    NIFTY 50 represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange.
    It is a key indicator of the Indian economy.
    
    You can trade the underlying index (NIFTY) or its Options (Calls/Puts).
    Calls profit when the market rises. Puts profit when the market falls.
    Be aware of time decay (Theta) in options.
    
    Recent context: {first_day_financial_prompt}
    """
)

# SEASONAL_FINANCIAL_REPORT = NamedVariable(
#     refname="seasonal_financial_report",
#     name="The Seasonal financial report of Stock A and B",
#     content="""
#         Stock A: {stock_a_report}
#         Stock B: {stock_b_report}
#     """
# )

SEASONAL_FINANCIAL_REPORT = NamedVariable(
    refname="seasonal_financial_report",
    name="Periodic Market Report",
    content="""
        Market Report: {market_report}
    """
)

# POST_MESSAGE_PROMPT = NamedBlock(
#     refname="post_message",
#     name="Instruction",
#     content="""
#     当前交易日结束了，请在论坛上简短地发表你的交易心得，并将其发布在论坛上。你发布的内容将对所有交易员公开可见。回答中只包含需要发布的内容。"""
# )

POST_MESSAGE_PROMPT = NamedBlock(
    refname="post_message",
    name="Instruction",
    content="""
    The current trading day is over, please briefly post your trading tips on the forum and post them on the forum.
    What you post will be publicly visible to all traders. The responses contain only what needs to be posted.
    """
)

# NEXT_DAY_ESTIMATE_PROMPT = NamedBlock(
#     refname="next_day_estimate",
#     name="Instruction",
#     content="""
#     请根据当前交易日的大盘信息和论坛信息，预估明天你是否会买入、卖出股票A和股票B，以及是否会选择贷款。预计会进行的行动标记为yes，不会进行标记为no。
#     用json格式返回结果，例如：
#     {{"buy_A":"yes", "buy_B":"no", "sell_A":"yes", "sell_B": "no", "loan": "yes"}}
#     """
# )

NEXT_DAY_ESTIMATE_PROMPT = NamedBlock(
    refname="next_day_estimate",
    name="Instruction",
    content="""
    Based on the market information and forum information of the current trading day, 
    please estimate whether you will buy and sell instruments tomorrow, and whether you will choose loan.
    Actions that are expected to take place are marked yes, and actions that will not take place are marked no. 
    Return the result in json format, for example:
    {{"buy_NIFTY":"yes", "sell_NIFTY":"no", "buy_OPTION":"yes", "sell_OPTION": "no", "loan": "yes"}}
    """
)

# NEXT_DAY_ESTIMATE_RETRY = NamedBlock(
#     refname="next_day_estimate_retry",
#     name="Instruction",
#     content="""
#     The following questions appeared in the JSON format you last answered: {fail_response}.
#     用json格式返回结果，例如：
#     {{"buy_A":"yes", "buy_B":"no", "sell_A":"yes", "sell_B": "no", "loan": "yes"}}
#     """
# )

NEXT_DAY_ESTIMATE_RETRY = NamedBlock(
    refname="next_day_estimate_retry",
    name="Instruction",
    content="""
    The following questions appeared in the JSON format you last answered: {fail_response}.
    Return the result in json format, for example:
    {{"buy_NIFTY":"yes", "sell_NIFTY":"no", "buy_OPTION":"yes", "sell_OPTION": "no", "loan": "yes"}}
    """
)