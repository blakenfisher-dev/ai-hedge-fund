from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class BotConfig:
    symbol: str = "EURUSD=X"
    interval: str = "1h"
    period: str = "2y"
    window_size: int = 30
    sl_options: list[int] = field(default_factory=lambda: [5, 10, 15, 25, 30, 60, 90, 120])
    tp_options: list[int] = field(default_factory=lambda: [5, 10, 15, 25, 30, 60, 90, 120])
    spread_pips: float = 1.0
    commission_pips: float = 0.0
    max_slippage_pips: float = 0.2
    hold_reward_weight: float = 0.0
    open_penalty_pips: float = 0.0
    time_penalty_pips: float = 0.0
    unrealized_delta_weight: float = 0.0
    pip_value: float = 0.0001
    lot_size: float = 100_000.0
    initial_equity_usd: float = 10_000.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
