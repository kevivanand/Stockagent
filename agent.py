"""
This module defines the Agent class, which represents an individual trader in the simulation.
Each agent makes decisions about loans and stock trades by interacting with an LLM.
"""
import math
import time
import openai
import random
from typing import Any, Dict

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled at runtime only when Gemini is enabled
    genai = None

import util
from log.custom_logger import log

from prompt.agent_prompt import (
    BACKGROUND_PROMPT,
    LOAN_TYPE_PROMPT,
    DECIDE_IF_LOAN_PROMPT,
    LASTDAY_FORUM_AND_STOCK_PROMPT,
    LOAN_RETRY_PROMPT,
    DECIDE_BUY_STOCK_PROMPT,
    FIRST_DAY_BACKGROUND_KNOWLEDGE,
    FIRST_DAY_FINANCIAL_REPORT,
    SEASONAL_FINANCIAL_REPORT,
    BUY_STOCK_RETRY_PROMPT,
    POST_MESSAGE_PROMPT,
    NEXT_DAY_ESTIMATE_PROMPT,
    NEXT_DAY_ESTIMATE_RETRY
)
from procoder.functional import format_prompt
from secretary import Secretary
from stock import Stock
from procoder.prompt import Collection, sharp2_indexing


def random_init(stock_a_initial: float, stock_b_initial: float) -> tuple[int, int, float, dict]:
    """
    Initializes a random portfolio for an agent, including stocks, cash, and an initial loan.

    Args:
        stock_a_initial: The initial price of stock A.
        stock_b_initial: The initial price of stock B.

    Returns: A tuple containing the initial amount of stock A, stock B, cash, and a dictionary representing the initial debt.
    """
    stock_a, stock_b, cash = 0, 0, 0.0
    total_property = 0
    while not (util.MIN_INITIAL_PROPERTY <= total_property <= util.MAX_INITIAL_PROPERTY):
        stock_a = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_a_initial))
        stock_b = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_b_initial))
        cash = random.uniform(0, util.MAX_INITIAL_PROPERTY)
        total_property = stock_a * stock_a_initial + stock_b * stock_b_initial + cash

    # Cap initial debt to a more reasonable level, e.g., 30% of initial property
    debt_amount = random.uniform(0, total_property * 0.3)
    debt = {
        "loan": "yes",
        "amount": debt_amount,
        "loan_type": random.randint(0, len(util.LOAN_TYPE) - 1),
        "repayment_date": random.choice(util.REPAYMENT_DAYS)
    }
    return stock_a, stock_b, cash, debt


class Agent:
    """Represents a trading agent in the stock market simulation."""

    def __init__(self, agent_id: int, stock_a_price: float, stock_b_price: float, secretary: Secretary, model: str):
        """
        Initializes an Agent.

        Args:
            agent_id: The unique identifier for the agent.
            stock_a_price: The initial price of stock A.
            stock_b_price: The initial price of stock B.
            secretary: The secretary object for validating LLM responses.
            model: The name of the LLM to use for decision-making.
        """
        self.order = agent_id
        self.secretary = secretary
        self.model = model
        self.character = random.choice(["Conservative", "Aggressive", "Balanced", "Growth-Oriented"])

        self.stock_a_amount, self.stock_b_amount, self.cash, init_debt = random_init(stock_a_price, stock_b_price)
        # The initial total property is used as a basis for the maximum loan an agent can take.
        self.init_proper = self.get_total_proper(stock_a_price, stock_b_price)

        self.chat_history = []
        self.loans = [init_debt]
        self.is_bankrupt = False
        self.quit = False

    def run_api(self, prompt: str, temperature: float = 1) -> str:
        """
        Runs the appropriate LLM API based on the configured model.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: The temperature for the LLM generation.

        Returns: The text response from the LLM.
        """
        if 'gpt' in self.model:
            return self.run_api_gpt(prompt, temperature)
        elif 'gemini' in self.model:
            return self.run_api_gemini(prompt, temperature)
        return ""

    def run_api_gemini(self, prompt: str, temperature: float = 1) -> str:
        """Sends a prompt to the Google Gemini API and returns the response."""
        if genai is None:
            log.logger.error("ERROR: google-generativeai is not installed. Skipping Gemini call.")
            return ""

        configure_fn = getattr(genai, "configure", None)
        generative_cls = getattr(genai, "GenerativeModel", None)
        if configure_fn is None or generative_cls is None:
            log.logger.error("ERROR: google-generativeai missing expected attributes. Skipping Gemini call.")
            return ""

        configure_fn(api_key=util.GOOGLE_API_KEY, transport='rest')

        generation_config: Any = {"candidate_count": 1, "temperature": temperature}
        types_module = getattr(genai, "types", None)
        if types_module is not None:
            generation_config_cls = getattr(types_module, "GenerationConfig", None)
            if generation_config_cls is not None:
                generation_config = generation_config_cls(candidate_count=1, temperature=temperature)

        model = generative_cls(self.model)
        self.chat_history.append({"role": "user", "parts": [prompt]})
        max_retry = 2
        retry = 0
        while retry < max_retry:
            try:
                response = model.generate_content(contents=self.chat_history, generation_config=generation_config)
                response_text = getattr(response, "text", "") or ""
                new_message_dict = {"role": 'model', "parts": [response_text]}
                self.chat_history.append(new_message_dict)
                return response_text
            except Exception as e:
                log.logger.warning("Gemini api retry...{}".format(e))
                retry += 1
                time.sleep(1)
        log.logger.error("ERROR: GEMINI API FAILED. SKIP THIS INTERACTION.")
        return ""


    def run_api_gpt(self, prompt: str, temperature: float = 1) -> str:
        """Sends a prompt to the OpenAI GPT API and returns the response."""
        openai.api_key = util.OPENAI_API_KEY
        client = openai.OpenAI(api_key=openai.api_key)
        self.chat_history.append({"role": "user", "content": prompt})
        max_retry = 2
        retry = 0

        # just cut off the overflow tokens
        # tokens = encoding.encode(self.chat_history)

        while retry < max_retry:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.chat_history,
                    temperature=temperature,
                )
                message = response.choices[0].message
                message_content = message.content or ""
                new_message_dict = {"role": message.role, "content": message_content}
                self.chat_history.append(new_message_dict)
                resp = message_content
                return resp
            except openai.OpenAIError as e:
                log.logger.warning("OpenAI api retry...{}".format(e))
                retry += 1
                time.sleep(1)
        log.logger.error("ERROR: OPENAI API FAILED. SKIP THIS INTERACTION.")
        return ""

    def get_total_proper(self, stock_a_price: float, stock_b_price: float) -> float:
        """Calculates the total property value of the agent (stocks + cash)."""
        return self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash

    def get_proper_cash_value(self, stock_a_price: float, stock_b_price: float) -> tuple[float, float, float, float]:
        """Returns the agent's total property, cash, and value of each stock holding."""
        proper = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash
        a_value = self.stock_a_amount * stock_a_price
        b_value = self.stock_b_amount * stock_b_price
        return proper, self.cash, a_value, b_value

    def _get_loan_prompt_and_inputs(self, date: int, stock_a_price: float, stock_b_price: float,
                                    lastday_forum_message: list) -> tuple[Any, dict[str, Any]]:
        """Constructs the prompt and inputs for the loan decision."""
        total_debt = sum(loan['amount'] for loan in self.loans)
        max_loan = self.init_proper - total_debt
        base_inputs = {
            'date': date,
            'character': self.character,
            'stock_a': self.stock_a_amount,
            'stock_b': self.stock_b_amount,
            'cash': self.cash,
            'debt': self.loans,
            'max_loan': max_loan,
            'loan_rate1': util.LOAN_RATE[0],
            'loan_rate2': util.LOAN_RATE[1],
            'loan_rate3': util.LOAN_RATE[2],
        }

        if date == 1:
            prompt = Collection(BACKGROUND_PROMPT,
                                LOAN_TYPE_PROMPT,
                                DECIDE_IF_LOAN_PROMPT).set_indexing_method(sharp2_indexing).set_sep("\n")
            return prompt, base_inputs
        else:
            prompt = Collection(BACKGROUND_PROMPT,
                                LASTDAY_FORUM_AND_STOCK_PROMPT,
                                LOAN_TYPE_PROMPT,
                                DECIDE_IF_LOAN_PROMPT).set_indexing_method(sharp2_indexing).set_sep("\n")
            other_days_inputs = {
                "stock_a_price": stock_a_price,
                "stock_b_price": stock_b_price,
                "lastday_forum_message": lastday_forum_message,
            }
            base_inputs.update(other_days_inputs)
            return prompt, base_inputs

    def _process_loan_decision(self, loan: dict, date: int):
        """Updates agent state based on the loan decision."""
        if loan["loan"] == "yes":
            loan["repayment_date"] = date + util.LOAN_TYPE_DATE[loan["loan_type"]]
            self.loans.append(loan)
            self.cash += loan["amount"]
            log.logger.info("INFO: Agent {} decide to loan: {}".format(self.order, loan))
        else:
            log.logger.info("INFO: Agent {} decide not to loan".format(self.order))

    def plan_loan(self, date: int, stock_a_price: float, stock_b_price: float, lastday_forum_message: list) -> dict:
        """
        Decides whether to take a loan by querying the LLM.

        Args:
            date: The current simulation day.
            stock_a_price: The current price of stock A.
            stock_b_price: The current price of stock B.
            lastday_forum_message: A list of messages from the previous day's forum.

        Returns: A dictionary representing the loan decision.
        """
        if self.quit:
            return {"loan": "no"}

        prompt, inputs = self._get_loan_prompt_and_inputs(date, stock_a_price, stock_b_price, lastday_forum_message)
        max_loan = inputs['max_loan']

        if max_loan <= 0:
            return {"loan": "no"}

        try_times = 0
        MAX_TRY_TIMES = 3
        prompt_text = format_prompt(prompt, inputs)
        if prompt_text is None:
            log.logger.warning("WARNING: Failed to format loan prompt. Skipping loan decision.")
            return {"loan": "no"}

        resp = self.run_api(prompt_text)
        if resp == "":
            return {"loan": "no"}

        loan_format_check, fail_response, loan = self.secretary.check_loan(resp, max_loan)
        loan_data: Dict[str, Any] = loan if loan is not None else {"loan": "no"}

        while not loan_format_check:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Loan format try times > MAX_TRY_TIMES. Skip as no loan today.")
                loan_data = {"loan": "no"}
                break

            retry_prompt = format_prompt(LOAN_RETRY_PROMPT, {"fail_response": fail_response})
            if retry_prompt is None:
                log.logger.warning("WARNING: Failed to format loan retry prompt. Skipping loan decision.")
                return {"loan": "no"}
            resp = self.run_api(retry_prompt)
            if resp == "":
                return {"loan": "no"}
            loan_format_check, fail_response, loan = self.secretary.check_loan(resp, max_loan)
            loan_data = loan if loan is not None else {"loan": "no"}

        self._process_loan_decision(loan_data, date)
        return loan_data

    def plan_stock(self, date: int, time: int, stock_a: Stock, stock_b: Stock, stock_a_deals: dict, stock_b_deals: dict) -> dict:
        """
        Decides whether to buy or sell stocks by querying the LLM.

        Args:
            date: The current simulation day.
            time: The current trading session.
            stock_a: The Stock object for stock A.
            stock_b: The Stock object for stock B.
            stock_a_deals: The current order book for stock A.
            stock_b_deals: The current order book for stock B.

        Returns: A dictionary representing the trade action.
        """
        if self.quit:
            return {"action_type": "no"}

        # Base prompt and inputs, used in all conditions
        prompt_collection = [DECIDE_BUY_STOCK_PROMPT]
        inputs = {
            "date": date,
            "time": time,
            "stock_a": self.stock_a_amount,
            "stock_b": self.stock_b_amount,
            "stock_a_price": stock_a.get_price(),
            "stock_b_price": stock_b.get_price(),
            "stock_a_deals": stock_a_deals,
            "stock_b_deals": stock_b_deals,
            "cash": self.cash
        }

        # Add financial reports and background on the first session of any day
        if time == 1:
            prompt_collection.insert(0, FIRST_DAY_BACKGROUND_KNOWLEDGE)
            prompt_collection.insert(0, FIRST_DAY_FINANCIAL_REPORT)

        # Add seasonal reports on specific days
        if date in util.SEASON_REPORT_DAYS and time == 1:
            index = util.SEASON_REPORT_DAYS.index(date)
            prompt_collection.insert(2, SEASONAL_FINANCIAL_REPORT)
            inputs["stock_a_report"] = stock_a.gen_financial_report(index)
            inputs["stock_b_report"] = stock_b.gen_financial_report(index)

        prompt = Collection(*prompt_collection).set_indexing_method(sharp2_indexing).set_sep("\n")

        try_times = 0
        MAX_TRY_TIMES = 3
        prompt_text = format_prompt(prompt, inputs)
        if prompt_text is None:
            log.logger.warning("WARNING: Failed to format stock action prompt. Skipping action.")
            return {"action_type": "no"}

        resp = self.run_api(prompt_text)
        if resp == "":
            return {"action_type": "no"}

        action_format_check, fail_response, action = self.secretary.check_action(
            resp, self.cash, self.stock_a_amount, self.stock_b_amount, stock_a.get_price(), stock_b.get_price())
        action_data: Dict[str, Any] = action if action is not None else {"action_type": "no"}
        while not action_format_check:
            # log.logger.debug("Action format check failed because of these issues: {}".format(fail_response))
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Action format try times > MAX_TRY_TIMES. Skip as no action today.")
                action_data = {"action_type": "no"}
                break

            retry_prompt = format_prompt(BUY_STOCK_RETRY_PROMPT, {"fail_response": fail_response})
            if retry_prompt is None:
                log.logger.warning("WARNING: Failed to format stock retry prompt. Skipping action.")
                return {"action_type": "no"}
            resp = self.run_api(retry_prompt)
            if resp == "":
                return {"action_type": "no"}
            action_format_check, fail_response, action = self.secretary.check_action(
                resp, self.cash, self.stock_a_amount, self.stock_b_amount, stock_a.get_price(), stock_b.get_price())
            action_data = action if action is not None else {"action_type": "no"}

        action_type = action_data.get("action_type")
        if action_type == "buy":
            log.logger.info("INFO: Agent {} decide to action: {}".format(self.order, action_data))
            return action_data
        elif action_type == "sell":
            log.logger.info("INFO: Agent {} decide to action: {}".format(self.order, action_data))
            return action_data
        elif action_type == "no":
            log.logger.info("INFO: Agent {} decide not to action".format(self.order))
            return action_data

        log.logger.error("ERROR: WRONG ACTION: {}".format(action_data))
        return {"action_type": "no"}

    def buy_stock(self, stock_name: str, amount: int, price: float) -> bool:
        """Updates agent's state after buying a stock."""
        if self.quit:
            return False
        if self.cash < price * amount or stock_name not in ['A', 'B']:
            log.logger.warning("ILLEGAL STOCK BUY BEHAVIOR: remain cash {}".format(self.cash))
            return False
        self.cash -= price * amount
        if stock_name == 'A':
            self.stock_a_amount += amount
        elif stock_name == 'B':
            self.stock_b_amount += amount

        return True

    def sell_stock(self, stock_name: str, amount: int, price: float) -> bool:
        """Updates agent's state after selling a stock."""
        if self.quit:
            return False
        if stock_name == 'B' and self.stock_b_amount < amount:
            log.logger.warning("ILLEGAL STOCK SELL BEHAVIOR: remain stock_b {}, amount {}".format(self.stock_b_amount,
                                                                                                  amount))
            return False
        elif stock_name == 'A' and self.stock_a_amount < amount:
            log.logger.warning("ILLEGAL STOCK SELL BEHAVIOR: remain stock_a {}, amount {}".format(self.stock_a_amount,
                                                                                                  amount))
            return False
        if stock_name == 'A':
            self.stock_a_amount -= amount
        elif stock_name == 'B':
            self.stock_b_amount -= amount
        self.cash += price * amount
        return True

    def loan_repayment(self, date: int):
        """Processes loan repayments due on the current date."""
        if self.quit:
            return
        for loan in self.loans[:]:
            if loan["repayment_date"] == date:
                self.cash -= loan["amount"] * (1 + util.LOAN_RATE[loan["loan_type"]])
                self.loans.remove(loan)
        if self.cash < 0:
            self.is_bankrupt = True


    def interest_payment(self):
        """Processes interest payments for all active loans."""
        if self.quit:
            return
        # Assuming interest is paid monthly (approx. 22 trading days)
        # The rate is annual, so we divide by the number of payment periods in a year.
        for loan in self.loans:
            # 264 total days, 22 days per period = 12 periods
            self.cash -= loan["amount"] * util.LOAN_RATE[loan["loan_type"]] / 12
            if self.cash < 0:
                self.is_bankrupt = True

    def bankrupt_process(self, stock_a_price: float, stock_b_price: float) -> bool:
        """Handles the bankruptcy process by liquidating assets to cover negative cash."""
        if self.quit:
            return False
        total_value_of_stock = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price
        if total_value_of_stock + self.cash < 0:
            log.logger.warning(f"Agent {self.order} bankrupt and is removed from the simulation.")
            return True
        # Liquidate stock A first to cover debt
        if stock_a_price * self.stock_a_amount >= -self.cash:
            sell_a = math.ceil(-self.cash / stock_a_price)
            self.stock_a_amount -= sell_a
            self.cash += sell_a * stock_a_price
        else:
            self.cash += stock_a_price * self.stock_a_amount
            self.stock_a_amount = 0
            # If liquidating all of stock A is not enough, liquidate stock B
            sell_b = math.ceil(-self.cash / stock_b_price)
            self.stock_b_amount -= sell_b
            self.cash += sell_b * stock_b_price

        if self.stock_a_amount < 0 or self.stock_b_amount < 0 or self.cash < 0:
            raise RuntimeError("ERROR: WRONG BANKRUPT PROCESS")
        self.is_bankrupt = False
        return False

    def post_message(self) -> str:
        """Generates and returns a forum message based on the agent's experience."""
        if self.quit:
            return ""
        prompt = format_prompt(POST_MESSAGE_PROMPT, inputs={})
        if prompt is None:
            log.logger.warning("WARNING: Failed to format post message prompt. Returning empty message.")
            return ""
        resp = self.run_api(prompt)
        return resp

    def next_day_estimate(self) -> dict:
        """Generates and returns an estimation of the next day's actions."""
        if self.quit:
            return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
        prompt = format_prompt(NEXT_DAY_ESTIMATE_PROMPT, inputs={})
        if prompt is None:
            log.logger.warning("WARNING: Failed to format next day estimate prompt. Using default estimate.")
            return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
        resp = self.run_api(prompt)
        if resp == "":
            return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
        format_check, fail_response, estimate = self.secretary.check_estimate(resp)
        try_times = 0
        MAX_TRY_TIMES = 3
        while not format_check:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Estimation format try times > MAX_TRY_TIMES. Skip as all 'no' today.")
                estimate = {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
                break
            retry_prompt = format_prompt(NEXT_DAY_ESTIMATE_RETRY, {"fail_response": fail_response})
            if retry_prompt is None:
                log.logger.warning("WARNING: Failed to format next day estimate retry prompt. Using default estimate.")
                return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
            resp = self.run_api(retry_prompt)
            if resp == "":
                return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
            format_check, fail_response, estimate = self.secretary.check_estimate(resp)
        return estimate
