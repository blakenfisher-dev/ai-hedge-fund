from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BotConfig
from .environment import ForexTradingEnv
from .features import (
    FEATURE_COLUMNS,
    add_features,
    chronological_splits,
    download_market_data,
    fit_feature_scaler,
    load_market_csv,
)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _load_data(config: BotConfig, csv_path: str | None) -> pd.DataFrame:
    raw = (
        load_market_csv(csv_path)
        if csv_path
        else download_market_data(
            config.symbol,
            config.period,
            config.interval,
        )
    )
    return add_features(raw)


def _environment(
    frame: pd.DataFrame,
    config: BotConfig,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    random_start: bool,
    episode_max_steps: int | None = None,
    force_close_on_end: bool = True,
) -> ForexTradingEnv:
    available_steps = max(2, len(frame) - config.window_size)
    return ForexTradingEnv(
        frame,
        config,
        mean,
        std,
        random_start=random_start,
        min_episode_steps=min(1000, available_steps),
        episode_max_steps=episode_max_steps,
        force_close_on_end=force_close_on_end,
    )


def evaluate_model(model, env: ForexTradingEnv) -> tuple[dict, list[dict]]:
    observation, _ = env.reset(seed=env.config.seed)
    equity_curve = [env.equity_usd]
    total_reward = 0.0

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        equity_curve.append(env.equity_usd)
        if terminated or truncated:
            break

    trades = list(env.closed_trades)
    wins = sum(trade["net_pips"] > 0 for trade in trades)
    equity = np.asarray(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(equity)
    drawdown = np.divide(
        equity - running_peak,
        running_peak,
        out=np.zeros_like(equity),
        where=running_peak != 0,
    )
    metrics = {
        "initial_equity_usd": float(equity[0]),
        "final_equity_usd": float(equity[-1]),
        "return_pct": float((equity[-1] / equity[0] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "total_reward_pips": total_reward,
        "closed_trades": len(trades),
        "win_rate_pct": float(wins / len(trades) * 100.0) if trades else 0.0,
        "net_pips": float(sum(trade["net_pips"] for trade in trades)),
    }
    return metrics, trades


def latest_signal(model, env: ForexTradingEnv) -> dict[str, Any]:
    observation, _ = env.reset(seed=env.config.seed)
    while env.current_step < env.n_steps - 1:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    action, _ = model.predict(observation, deterministic=True)
    decision = env.decode_action(int(action))
    effective_action = decision["action"]
    if effective_action == "OPEN" and env.position != 0:
        effective_action = "HOLD"
    elif effective_action == "CLOSE" and env.position == 0:
        effective_action = "HOLD"

    return {
        "generated_from_bar": env.frame.index[env.current_step].isoformat(),
        "symbol": env.config.symbol,
        "paper_only": True,
        "effective_action": effective_action,
        "model_decision": decision,
        "current_position": "LONG" if env.position == 1 else "SHORT" if env.position == -1 else "FLAT",
        "entry_price": env.entry_price,
        "stop_loss_price": env.sl_price,
        "take_profit_price": env.tp_price,
        "equity_usd": env.equity_usd,
        "warning": "Research output only. No broker order was submitted.",
    }


def train_and_run(
    config: BotConfig,
    output_dir: str | Path,
    total_timesteps: int,
    csv_path: str | None = None,
) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    output_dir = Path(output_dir)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_data(config, csv_path)
    train, validation, test, boundaries = chronological_splits(frame, config.window_size)
    mean, std = fit_feature_scaler(train)
    train_episode_steps = min(2000, max(64, len(train) - config.window_size))

    train_env = DummyVecEnv(
        [
            lambda: _environment(
                train,
                config,
                mean,
                std,
                random_start=True,
                episode_max_steps=train_episode_steps,
            )
        ]
    )
    validation_env = DummyVecEnv([lambda: _environment(validation, config, mean, std, random_start=False)])

    rollout_steps = min(1024, max(64, total_timesteps // 4))
    batch_size = 64 if rollout_steps >= 64 else rollout_steps
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=config.seed,
        verbose=1,
        n_steps=rollout_steps,
        batch_size=batch_size,
        tensorboard_log=None,
        device="auto",
    )
    eval_callback = EvalCallback(
        validation_env,
        best_model_save_path=str(model_dir),
        log_path=str(output_dir / "evaluation"),
        eval_freq=max(64, total_timesteps // 4),
        deterministic=True,
        n_eval_episodes=1,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    best_model_path = model_dir / "best_model.zip"
    if best_model_path.exists():
        model = PPO.load(best_model_path)
    else:
        model.save(model_dir / "best_model")

    validation_metrics, _ = evaluate_model(
        model,
        _environment(validation, config, mean, std, random_start=False),
    )
    test_metrics, test_trades = evaluate_model(
        model,
        _environment(test, config, mean, std, random_start=False),
    )
    signal = latest_signal(
        model,
        _environment(
            test,
            config,
            mean,
            std,
            random_start=False,
            force_close_on_end=False,
        ),
    )

    metadata = {
        "config": config.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": mean,
        "feature_std": std,
        "split_boundaries": boundaries,
        "rows": {
            "all": len(frame),
            "train": len(train),
            "validation_with_context": len(validation),
            "test_with_context": len(test),
        },
        "training_timesteps": total_timesteps,
        "model_file": "best_model.zip",
        "source_logic": "https://github.com/ZiadFrancis/ReinforcementTrading_Part_1",
    }
    report = {
        "validation": validation_metrics,
        "test": test_metrics,
        "latest_signal": signal,
        "selection_policy": "Model selection uses validation reward; test data is evaluated once.",
    }

    _write_json(model_dir / "metadata.json", metadata)
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "latest_signal.json", signal)
    pd.DataFrame(test_trades).to_csv(output_dir / "test_trades.csv", index=False)
    return report


def signal_from_saved_model(
    model_dir: str | Path,
    output_path: str | Path,
    csv_path: str | None = None,
    period: str | None = None,
) -> dict:
    from stable_baselines3 import PPO

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    config = BotConfig(**metadata["config"])
    if period:
        config = replace(config, period=period)

    frame = _load_data(config, csv_path)
    _, _, test, _ = chronological_splits(frame, config.window_size)
    mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
    std = np.asarray(metadata["feature_std"], dtype=np.float32)
    model = PPO.load(model_dir / metadata["model_file"])
    signal = latest_signal(
        model,
        _environment(
            test,
            config,
            mean,
            std,
            random_start=False,
            force_close_on_end=False,
        ),
    )
    _write_json(Path(output_path), signal)
    return signal
