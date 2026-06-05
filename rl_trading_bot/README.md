# EUR/USD Reinforcement-Learning Paper Bot

This package converts the logic from
[`ZiadFrancis/ReinforcementTrading_Part_1`](https://github.com/ZiadFrancis/ReinforcementTrading_Part_1)
into a reproducible GitHub Actions paper-trading workflow.

It does not submit broker orders.

## What changed from the prototype

- Downloads current `EURUSD=X` hourly data from Yahoo Finance.
- Uses one action-space definition during training and inference.
- Fits feature normalization on training data only.
- Splits data chronologically into 70% train, 15% validation, and 15% test.
- Selects PPO checkpoints on validation reward, not test equity.
- Evaluates the test set once and writes machine-readable artifacts.
- Preserves the prototype's persistent position, discrete SL/TP, spread,
  slippage, and intrabar stop/target logic.

## Local run

```powershell
.\.venv\Scripts\python.exe -m pip install -r rl_trading_bot\requirements.txt
.\.venv\Scripts\python.exe -m rl_trading_bot run `
  --symbol EURUSD=X `
  --period 2y `
  --interval 1h `
  --timesteps 100000 `
  --output-dir outputs\rl-paper-bot
```

Use a CSV for reproducible research:

```powershell
.\.venv\Scripts\python.exe -m rl_trading_bot run `
  --csv C:\path\to\eurusd.csv `
  --timesteps 100000 `
  --output-dir outputs\rl-paper-bot
```

The CSV must contain a timestamp column plus `Open`, `High`, `Low`, and `Close`.

## Outputs

- `model/best_model.zip`: selected PPO model
- `model/metadata.json`: scaler, configuration, and split boundaries
- `report.json`: validation and untouched-test metrics
- `latest_signal.json`: latest paper decision and simulated position
- `test_trades.csv`: closed trades from the test replay

## GitHub schedule

`.github/workflows/rl-paper-trading.yml` runs at `07:15 Australia/Brisbane`
Monday through Friday. GitHub schedules can start late under load. Each run
uploads the model, report, signal, and trades as a workflow artifact.

## Validation boundary

The workflow proves that the software runs. It does not prove profitability.
Do not connect broker credentials until walk-forward results remain positive
after realistic costs and an independent paper-trading period.
