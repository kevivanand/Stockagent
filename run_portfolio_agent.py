from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DATA_PATH = Path("portfolio_holdings.csv")


@dataclass
class Holding:
    symbol: str
    category: str
    net_qty: int
    avg_price: float
    last_traded_price: float
    current_value: float
    day_change_pct: float
    overall_change_pct: float

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Holding":
        """Convert a CSV row into a Holding, handling commas and percent signs."""

        def parse_int(value: str) -> int:
            value = value.strip()
            return int(value.replace(",", "")) if value else 0

        def parse_float(value: str) -> float:
            value = value.strip().replace(",", "").replace("%", "")
            if not value or value == "--":
                return 0.0
            return float(value)

        return cls(
            symbol=row.get("Symbol (48)", "").strip(),
            category=row.get("Category", "").strip(),
            net_qty=parse_int(row.get("Net Qty", "0")),
            avg_price=parse_float(row.get("Avg. Price", "0")),
            last_traded_price=parse_float(row.get("LTP", "0")),
            current_value=parse_float(row.get("Current Value", "0")),
            day_change_pct=parse_float(row.get("Day %", "0")),
            overall_change_pct=parse_float(row.get("Overall %", "0")),
        )

    def decision(self) -> str:
        """
        Produce a simple qualitative decision based on performance.

        - Large unrealised loss: consider trimming.
        - Strong gain: consider profit booking.
        - Otherwise hold and monitor.
        """
        if self.overall_change_pct <= -5:
            return "Review – large unrealised loss"
        if self.overall_change_pct >= 10:
            return "Book partial profits"
        if self.day_change_pct <= -2:
            return "Monitor – sharp daily drop"
        return "Hold"


def load_holdings(path: Path) -> List[Holding]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find holdings file at {path.resolve()}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [Holding.from_row(row) for row in reader]


def summarise(holdings: Iterable[Holding]) -> None:
    holdings = list(holdings)
    if not holdings:
        print("No holdings found.")
        return

    total_value = sum(item.current_value for item in holdings)
    print(f"Total current value: ₹{total_value:,.2f}")
    print("-" * 60)

    for holding in holdings:
        print(
            f"{holding.symbol:<10} | Qty: {holding.net_qty:>4} | "
            f"P/L: {holding.overall_change_pct:+6.2f}% | Decision: {holding.decision()}"
        )


if __name__ == "__main__":
    summarise(load_holdings(DATA_PATH))
