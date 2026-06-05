from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import BotConfig
from .features import FEATURE_COLUMNS


class ForexTradingEnv(gym.Env):
    """Position-persistent EUR/USD environment adapted from the referenced prototype."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        frame,
        config: BotConfig,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        random_start: bool = False,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        force_close_on_end: bool = True,
    ):
        super().__init__()
        self.frame = frame.copy()
        self.config = config
        self.window_size = config.window_size
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps
        self.episode_max_steps = episode_max_steps
        self.force_close_on_end = force_close_on_end
        self.n_steps = len(self.frame)

        if self.n_steps <= self.window_size + 1:
            raise ValueError("Dataframe is too short for the configured observation window.")
        if self.feature_mean.shape != (len(FEATURE_COLUMNS),):
            raise ValueError("Feature scaler shape does not match the feature set.")

        self.action_map: list[tuple[str, int | None, float | None, float | None]] = [
            ("HOLD", None, None, None),
            ("CLOSE", None, None, None),
        ]
        for direction in [0, 1]:
            for stop_loss in config.sl_options:
                for take_profit in config.tp_options:
                    self.action_map.append(("OPEN", direction, float(stop_loss), float(take_profit)))

        self.action_space = spaces.Discrete(len(self.action_map))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(FEATURE_COLUMNS) + 3),
            dtype=np.float32,
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = self.window_size - 1
        self.steps_in_episode = 0
        self.position = 0
        self.entry_price: float | None = None
        self.sl_price: float | None = None
        self.tp_price: float | None = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.equity_usd = self.config.initial_equity_usd
        self.last_trade_info: dict[str, Any] | None = None
        self.closed_trades: list[dict[str, Any]] = []

    @property
    def usd_per_pip(self) -> float:
        return self.config.pip_value * self.config.lot_size

    def decode_action(self, action: int) -> dict[str, Any]:
        action_type, direction, stop_loss, take_profit = self.action_map[int(action)]
        return {
            "action_index": int(action),
            "action": action_type,
            "direction": None if direction is None else ("LONG" if direction == 1 else "SHORT"),
            "stop_loss_pips": stop_loss,
            "take_profit_pips": take_profit,
        }

    def _unrealized_pips(self) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close = float(self.frame.iloc[self.current_step]["Close"])
        price_change = close - self.entry_price if self.position == 1 else self.entry_price - close
        return price_change / self.config.pip_value

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.window_size + 1
        base = self.frame.iloc[start : self.current_step + 1][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        base = (base - self.feature_mean) / self.feature_std

        state = np.array(
            [
                float(self.position),
                float(self.time_in_trade) / 1000.0,
                self._unrealized_pips() / 100.0,
            ],
            dtype=np.float32,
        )
        state_block = np.tile(state, (self.window_size, 1))
        return np.hstack([base, state_block]).astype(np.float32)

    def _slippage_pips(self) -> float:
        maximum = self.config.max_slippage_pips
        return 0.0 if maximum <= 0 else float(self.np_random.uniform(0.0, maximum))

    def _open_position(self, direction: int, stop_loss_pips: float, take_profit_pips: float) -> None:
        close = float(self.frame.iloc[self.current_step]["Close"])
        slippage = self._slippage_pips() * self.config.pip_value
        entry = close + slippage if direction == 1 else close - slippage

        self.position = 1 if direction == 1 else -1
        self.entry_price = entry
        if self.position == 1:
            self.sl_price = entry - stop_loss_pips * self.config.pip_value
            self.tp_price = entry + take_profit_pips * self.config.pip_value
        else:
            self.sl_price = entry + stop_loss_pips * self.config.pip_value
            self.tp_price = entry - take_profit_pips * self.config.pip_value
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.last_trade_info = {
            "event": "OPEN",
            "timestamp": self.frame.index[self.current_step].isoformat(),
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
        }

    def _close_position(self, reason: str, exit_price: float, step: int) -> float:
        if self.entry_price is None:
            return 0.0
        price_change = exit_price - self.entry_price if self.position == 1 else self.entry_price - exit_price
        realized_pips = price_change / self.config.pip_value
        cost_pips = self.config.spread_pips + self.config.commission_pips
        net_pips = realized_pips - cost_pips
        self.equity_usd += net_pips * self.usd_per_pip

        trade = {
            "event": "CLOSE",
            "reason": reason,
            "timestamp": self.frame.index[step].isoformat(),
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }
        self.closed_trades.append(trade)
        self.last_trade_info = trade
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        return net_pips

    def _check_next_bar(self) -> float | None:
        if self.position == 0 or self.sl_price is None or self.tp_price is None:
            return None
        next_step = self.current_step + 1
        if next_step >= self.n_steps:
            return None

        high = float(self.frame.iloc[next_step]["High"])
        low = float(self.frame.iloc[next_step]["Low"])
        if self.position == 1:
            stop_hit = low <= self.sl_price
            target_hit = high >= self.tp_price
        else:
            stop_hit = high >= self.sl_price
            target_hit = low <= self.tp_price

        if stop_hit:
            reason = "SL_AND_TP_SAME_BAR_SL_FIRST" if target_hit else "SL_HIT"
            return self._close_position(reason, self.sl_price, next_step)
        if target_hit:
            return self._close_position("TP_HIT", self.tp_price, next_step)
        return None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            latest_start = self.n_steps - max(self.min_episode_steps, 2)
            earliest_start = self.window_size - 1
            if latest_start > earliest_start:
                self.current_step = int(self.np_random.integers(earliest_start, latest_start))

        return self._observation(), {"equity_usd": self.equity_usd}

    def step(self, action: int):
        self.last_trade_info = None
        reward_pips = 0.0
        action_details = self.decode_action(int(action))
        action_type, direction, stop_loss, take_profit = self.action_map[int(action)]

        if action_type == "CLOSE" and self.position != 0:
            close = float(self.frame.iloc[self.current_step]["Close"])
            slippage = self._slippage_pips() * self.config.pip_value
            exit_price = close - slippage if self.position == 1 else close + slippage
            reward_pips += self._close_position("MANUAL_CLOSE", exit_price, self.current_step)
        elif action_type == "OPEN" and self.position == 0:
            self._open_position(int(direction), float(stop_loss), float(take_profit))
            reward_pips -= self.config.open_penalty_pips

        realized = self._check_next_bar()
        if realized is not None:
            reward_pips += realized

        if self.current_step < self.n_steps - 1:
            self.current_step += 1
        self.steps_in_episode += 1

        if self.position != 0:
            self.time_in_trade += 1
            unrealized = self._unrealized_pips()
            delta_unrealized = unrealized - self.prev_unrealized_pips
            if unrealized > 0:
                reward_pips += self.config.hold_reward_weight * unrealized
            reward_pips += self.config.unrealized_delta_weight * delta_unrealized
            reward_pips -= self.config.time_penalty_pips
            self.prev_unrealized_pips = unrealized

        terminated = self.current_step >= self.n_steps - 1
        truncated = self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps
        if (terminated or truncated) and self.force_close_on_end and self.position != 0:
            close = float(self.frame.iloc[self.current_step]["Close"])
            reward_pips += self._close_position("END_OF_EPISODE", close, self.current_step)

        info = {
            "timestamp": self.frame.index[self.current_step].isoformat(),
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "last_trade_info": self.last_trade_info,
            "decision": action_details,
        }
        return self._observation(), float(reward_pips), terminated, truncated, info

    def render(self) -> None:
        print(f"{self.frame.index[self.current_step].isoformat()} " f"equity=${self.equity_usd:,.2f} position={self.position} " f"entry={self.entry_price} sl={self.sl_price} tp={self.tp_price}")
