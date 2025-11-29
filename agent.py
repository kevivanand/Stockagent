"""
This module defines the Agent class, which represents an individual trader in the simulation.
Each agent makes decisions about loans and stock trades by interacting with an LLM.
"""
import math
import time
import openai
import random
from typing import Any, Dict, List

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
from stock import Instrument, Stock, Option
from procoder.prompt import Collection, sharp2_indexing


def random_init() -> tuple[Dict[str, int], float, dict]:
    """
    Initializes a random portfolio for an agent.
    """
    holdings = {}
    cash = random.uniform(util.MIN_INITIAL_PROPERTY, util.MAX_INITIAL_PROPERTY)
    total_property = cash 

    # Cap initial debt
    debt_amount = random.uniform(0, total_property * 0.3)
    debt = {
        "loan": "yes",
        "amount": debt_amount,
        "loan_type": random.randint(0, len(util.LOAN_TYPE) - 1),
        "repayment_date": random.choice(util.REPAYMENT_DAYS)
    }
    return holdings, cash, debt


class Agent:
    """Represents a trading agent in the stock market simulation."""

    def __init__(self, agent_id: int, instruments: List[Instrument], secretary: Secretary, model: str):
        """
        Initializes an Agent.
        """
        self.order = agent_id
        self.secretary = secretary
        self.model = model
        self.character = random.choice(["Conservative", "Aggressive", "Balanced", "Growth-Oriented"])
        self.instruments = instruments

        self.holdings, self.cash, init_debt = random_init()
        # The initial total property is used as a basis for the maximum loan an agent can take.
        self.init_proper = self.get_total_proper()

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

    def get_total_proper(self) -> float:
        """Calculates the total property value of the agent (stocks + cash)."""
        stock_value = sum(self.holdings.get(inst.symbol, 0) * inst.get_price() for inst in self.instruments)
        return stock_value + self.cash

    def get_proper_cash_value(self) -> tuple[float, float, Dict[str, float]]:
        """Returns the agent's total property, cash, and value of each stock holding."""
        stock_values = {inst.symbol: self.holdings.get(inst.symbol, 0) * inst.get_price() for inst in self.instruments}
        proper = sum(stock_values.values()) + self.cash
        return proper, self.cash, stock_values

    def _get_loan_prompt_and_inputs(self, date: int, lastday_forum_message: list) -> tuple[Any, dict[str, Any]]:
        """Constructs the prompt and inputs for the loan decision."""
        total_debt = sum(loan['amount'] for loan in self.loans)
        max_loan = self.init_proper - total_debt
        base_inputs = {
            'date': date,
            'character': self.character,
            'holdings': self.holdings,
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
            
            prices_str = ", ".join([f"{inst.symbol}: {inst.get_price()}" for inst in self.instruments])
            other_days_inputs = {
                "stock_prices": prices_str,
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

    def plan_loan(self, date: int, lastday_forum_message: list) -> dict:
        """
        Decides whether to take a loan by querying the LLM.
        """
        if self.quit:
            return {"loan": "no"}

        prompt, inputs = self._get_loan_prompt_and_inputs(date, lastday_forum_message)
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

    def plan_stock(self, date: int, time: int, deals: Dict[str, dict]) -> dict:
        """
        Decides whether to buy or sell stocks by querying the LLM.
        """
        if self.quit:
            return {"action_type": "no"}

        # Base prompt and inputs, used in all conditions
        prompt_collection = [DECIDE_BUY_STOCK_PROMPT]
        
        prices_str = ", ".join([f"{inst.symbol}: {inst.get_price()}" for inst in self.instruments])
        deals_str = str(deals) # Simplify for now, maybe format better later

        inputs = {
            "date": date,
            "time": time,
            "holdings": self.holdings,
            "stock_prices": prices_str,
            "deals": deals_str,
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
            # inputs["stock_a_report"] = stock_a.gen_financial_report(index)
            # inputs["stock_b_report"] = stock_b.gen_financial_report(index)
            # TODO: Add generic report generation if needed

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

        prices = {inst.symbol: inst.get_price() for inst in self.instruments}
        action_format_check, fail_response, action = self.secretary.check_action(
            resp, self.cash, self.holdings, prices)
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
                resp, self.cash, self.holdings, prices)
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
        if self.cash < price * amount:
            log.logger.warning("ILLEGAL STOCK BUY BEHAVIOR: remain cash {}".format(self.cash))
            return False
        self.cash -= price * amount
        self.holdings[stock_name] = self.holdings.get(stock_name, 0) + amount
        return True

    def sell_stock(self, stock_name: str, amount: int, price: float) -> bool:
        """Updates agent's state after selling a stock."""
        if self.quit:
            return False
        
        current_holding = self.holdings.get(stock_name, 0)
        if current_holding < amount:
             log.logger.warning("ILLEGAL STOCK SELL BEHAVIOR: remain stock {}, amount {}".format(current_holding, amount))
             return False

        self.holdings[stock_name] = current_holding - amount
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

    def bankrupt_process(self) -> bool:
        """Handles the bankruptcy process by liquidating assets to cover negative cash."""
        if self.quit:
            return False
        
        total_value_of_stock = sum(self.holdings.get(inst.symbol, 0) * inst.get_price() for inst in self.instruments)
        if total_value_of_stock + self.cash < 0:
            log.logger.warning(f"Agent {self.order} bankrupt and is removed from the simulation.")
            return True
        
        # Liquidate stocks to cover debt
        for inst in self.instruments:
            if self.cash >= 0:
                break
            
            holding = self.holdings.get(inst.symbol, 0)
            if holding > 0:
                price = inst.get_price()
                if price * holding >= -self.cash:
                    sell_amount = math.ceil(-self.cash / price)
                    self.holdings[inst.symbol] -= sell_amount
                    self.cash += sell_amount * price
                else:
                    self.cash += price * holding
                    self.holdings[inst.symbol] = 0

        if self.cash < 0:
             # Should not happen if total value check passed, unless rounding errors or price drops?
             # But here we use current price.
             pass

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
