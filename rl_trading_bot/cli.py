from __future__ import annotations

import argparse
import json

from .config import BotConfig
from .pipeline import signal_from_saved_model, train_and_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EUR/USD PPO paper-trading bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Download data, train, evaluate, and emit a signal")
    run_parser.add_argument("--symbol", default="EURUSD=X")
    run_parser.add_argument("--period", default="2y")
    run_parser.add_argument("--interval", default="1h")
    run_parser.add_argument("--timesteps", type=int, default=100_000)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--csv", help="Use a local OHLC CSV instead of Yahoo Finance")
    run_parser.add_argument("--output-dir", default="outputs/rl-paper-bot")

    signal_parser = subparsers.add_parser("signal", help="Generate a signal from an existing model")
    signal_parser.add_argument("--model-dir", required=True)
    signal_parser.add_argument("--csv", help="Use a local OHLC CSV instead of Yahoo Finance")
    signal_parser.add_argument("--period", help="Override the saved download period")
    signal_parser.add_argument("--output", default="outputs/rl-paper-bot/latest_signal.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = BotConfig(
            symbol=args.symbol,
            period=args.period,
            interval=args.interval,
            seed=args.seed,
        )
        result = train_and_run(
            config,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            csv_path=args.csv,
        )
    else:
        result = signal_from_saved_model(
            model_dir=args.model_dir,
            output_path=args.output,
            csv_path=args.csv,
            period=args.period,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
