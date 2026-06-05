from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_trading_bot.config import BotConfig
from rl_trading_bot.environment import ForexTradingEnv
from rl_trading_bot.features import (
    FEATURE_COLUMNS,
    add_features,
    chronological_splits,
    fit_feature_scaler,
)


def synthetic_market(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    trend = np.linspace(1.08, 1.12, rows)
    wave = np.sin(np.arange(rows) / 8.0) * 0.001
    close = trend + wave
    return pd.DataFrame(
        {
            "Open": close - 0.0001,
            "High": close + 0.0006,
            "Low": close - 0.0006,
            "Close": close,
            "Volume": 100,
        },
        index=index,
    )


def test_features_and_splits_are_finite_and_chronological():
    featured = add_features(synthetic_market())
    assert np.isfinite(featured[FEATURE_COLUMNS].to_numpy()).all()

    train, validation, test, boundaries = chronological_splits(featured, window_size=30)
    assert train.index[-1] < pd.Timestamp(boundaries["validation_start"])
    assert validation.index[-1] < pd.Timestamp(boundaries["test_start"])
    assert test.index[-1] == featured.index[-1]


def test_environment_action_space_and_take_profit_close():
    featured = add_features(synthetic_market())
    mean, std = fit_feature_scaler(featured)
    config = BotConfig(
        sl_options=[5],
        tp_options=[5],
        spread_pips=1.0,
        max_slippage_pips=0.0,
    )
    env = ForexTradingEnv(featured, config, mean, std)
    env.reset(seed=7)
    entry_close = float(env.frame.iloc[env.current_step]["Close"])
    env.frame.iloc[env.current_step + 1, env.frame.columns.get_loc("Low")] = entry_close
    env.frame.iloc[env.current_step + 1, env.frame.columns.get_loc("High")] = entry_close + 0.001

    assert env.action_space.n == 4
    long_action = env.action_map.index(("OPEN", 1, 5.0, 5.0))
    _, reward, _, _, info = env.step(long_action)

    assert reward == pytest.approx(4.0)
    assert info["last_trade_info"]["reason"] == "TP_HIT"
    assert info["position"] == 0
    assert env.equity_usd == pytest.approx(10_040.0)


def test_feature_scaler_uses_expected_shape():
    featured = add_features(synthetic_market())
    mean, std = fit_feature_scaler(featured)
    assert mean.shape == (len(FEATURE_COLUMNS),)
    assert std.shape == mean.shape
    assert (std > 0).all()
