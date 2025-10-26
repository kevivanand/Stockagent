from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

DATA_PATH = Path("batch_stockagent_exit.csv")


def _parse_int(value: str) -> int:
    value = value.strip().replace(",", "")
    return int(value) if value and value != "--" else 0


def _parse_float(value: str) -> float:
    value = value.strip().replace(",", "")
    if not value or value == "--":
        return 0.0
    return float(value)


@dataclass
class ExitCandidate:
    name: str
    ticker: str
    shares: int
    avg_buy_price: float
    current_price: float

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ExitCandidate":
        return cls(
            name=row.get("Name", "").strip(),
            ticker=row.get("Ticker", row.get("Name", "")).strip(),
            shares=_parse_int(row.get("Shares", row.get("Quantity", "0"))),
            avg_buy_price=_parse_float(row.get("Avg Buy Price Rs.", row.get("AvgPrice", "0"))),
            current_price=_parse_float(row.get("Current Price Rs.", row.get("LTP", "0"))),
        )

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_buy_price

    @property
    def proceeds(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        if self.avg_buy_price == 0:
            return 0.0
        return ((self.current_price - self.avg_buy_price) / self.avg_buy_price) * 100

    def recommendation(self) -> str:
        """Simple exit guidance based on profitability."""
        if self.pnl_pct >= 15:
            return "Exit – lock in profit"
        if self.pnl_pct <= -10:
            return "Exit – limit downside"
        return "Partial exit / review"


def load_candidates(path: Path) -> List[ExitCandidate]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data file at {path.resolve()}")

    rows: List[List[str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            rows.append([part.strip().strip('"') for part in raw.split(",")])

    if not rows:
        return []

    header = rows[0]
    candidates: List[ExitCandidate] = []
    for parts in rows[1:]:
        row_dict = {header[idx]: value for idx, value in enumerate(parts)}
        candidates.append(ExitCandidate.from_row(row_dict))
    return candidates


def summarise_exits(candidates: Iterable[ExitCandidate]) -> None:
    candidates = [c for c in candidates if c.shares > 0]
    if not candidates:
        print("No positions to exit.")
        return

    total_shares = sum(c.shares for c in candidates)
    total_cost = sum(c.cost_basis for c in candidates)
    total_proceeds = sum(c.proceeds for c in candidates)
    total_pnl = total_proceeds - total_cost

    print(f"Positions to exit: {len(candidates)} symbols ({total_shares} shares total)")
    print(f"Estimated proceeds: ₹{total_proceeds:,.2f}")
    print(f"Total P/L: ₹{total_pnl:,.2f} ({(total_pnl / total_cost * 100) if total_cost else 0:.2f}%)")
    print("-" * 70)

    for candidate in candidates:
        print(
            f"{candidate.ticker:<12} | Qty: {candidate.shares:>5} | "
            f"Avg: ₹{candidate.avg_buy_price:>8.2f} | LTP: ₹{candidate.current_price:>8.2f} | "
            f"P/L: ₹{candidate.pnl:>9.2f} ({candidate.pnl_pct:+6.2f}%) | {candidate.recommendation()}"
        )


if __name__ == "__main__":
    summarise_exits(load_candidates(DATA_PATH))
