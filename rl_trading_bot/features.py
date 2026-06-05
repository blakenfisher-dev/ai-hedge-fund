from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
FEATURE_COLUMNS = [
    "rsi_14",
    "atr_14",
    "ma_20_slope",
    "ma_50_slope",
    "close_ma20_diff",
    "close_ma50_diff",
    "ma_spread",
    "ma_spread_slope",
]


def _flatten_yfinance_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if symbol in values:
            return frame.xs(symbol, axis=1, level=level, drop_level=True)

    flattened = frame.copy()
    flattened.columns = [str(column[0]) for column in flattened.columns]
    return flattened


def normalize_market_data(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    frame = _flatten_yfinance_columns(frame.copy(), symbol)
    frame.columns = [str(column).strip().title() for column in frame.columns]

    missing = [column for column in OHLC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        timestamp_column = next(
            (column for column in ["Datetime", "Date", "Timestamp", "Time", "Time (Eet)"] if column in frame.columns),
            None,
        )
        if timestamp_column is None:
            raise ValueError("Market data needs a DatetimeIndex or timestamp column.")
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        frame = frame.set_index(timestamp_column)
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)

    for column in OHLC_COLUMNS + ["Volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=OHLC_COLUMNS)
    if frame.empty:
        raise ValueError("No valid OHLC rows remain after normalization.")
    return frame


def download_market_data(
    symbol: str,
    period: str,
    interval: str,
    retries: int = 3,
) -> pd.DataFrame:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if frame.empty:
                raise RuntimeError("Yahoo Finance returned no rows.")
            return normalize_market_data(frame, symbol=symbol)
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            if attempt < retries:
                time.sleep(attempt * 3)
    raise RuntimeError(f"Could not download {symbol} after {retries} attempts: {last_error}")


def load_market_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_market_data(frame)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    frame["rsi_14"] = 100.0 - (100.0 / (1.0 + relative_strength))
    frame["rsi_14"] = frame["rsi_14"].fillna(100.0).where(average_gain.ne(0.0), 50.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    frame["ma_20"] = close.rolling(20, min_periods=20).mean()
    frame["ma_50"] = close.rolling(50, min_periods=50).mean()
    frame["ma_20_slope"] = frame["ma_20"].diff()
    frame["ma_50_slope"] = frame["ma_50"].diff()
    frame["close_ma20_diff"] = close - frame["ma_20"]
    frame["close_ma50_diff"] = close - frame["ma_50"]
    frame["ma_spread"] = frame["ma_20"] - frame["ma_50"]
    frame["ma_spread_slope"] = frame["ma_spread"].diff()

    frame = frame.dropna(subset=FEATURE_COLUMNS + OHLC_COLUMNS)
    if frame.empty:
        raise ValueError("Not enough market data to calculate the 50-period features.")
    return frame


def fit_feature_scaler(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def chronological_splits(
    frame: pd.DataFrame,
    window_size: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    row_count = len(frame)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    minimum_rows = window_size + 3

    if train_end < minimum_rows or validation_end - train_end < 3 or row_count - validation_end < 3:
        raise ValueError(f"Need more feature rows for chronological train/validation/test splits; received {row_count}.")

    context = window_size - 1
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[max(0, train_end - context) : validation_end].copy()
    test = frame.iloc[max(0, validation_end - context) :].copy()

    if len(validation) < minimum_rows or len(test) < minimum_rows:
        raise ValueError("Validation or test split is too short after adding observation context.")

    boundaries = {
        "train_end": frame.index[train_end - 1].isoformat(),
        "validation_start": frame.index[train_end].isoformat(),
        "validation_end": frame.index[validation_end - 1].isoformat(),
        "test_start": frame.index[validation_end].isoformat(),
        "latest_bar": frame.index[-1].isoformat(),
    }
    return train, validation, test, boundaries
