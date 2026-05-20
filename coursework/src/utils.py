"""Shared numerical and filesystem helpers.

This module provides small reusable utilities for rolling indicators, random
seed control, directory creation, and SHA256 hashing.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rolling_percentile_rank(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(10, window // 3)

    def percentile(values: np.ndarray) -> float:
        last = values[-1]
        valid = values[np.isfinite(values)]
        if len(valid) == 0 or not np.isfinite(last):
            return np.nan
        return float(np.mean(valid <= last))

    return series.rolling(window, min_periods=min_periods).apply(percentile, raw=True)


def trend_tstat(values: np.ndarray) -> float:
    if len(values) < 5 or not np.isfinite(values).all():
        return np.nan
    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    y = values - values.mean()
    denom = float(np.sum(x**2))
    if denom <= 0:
        return np.nan
    beta = float(np.sum(x * y) / denom)
    residual = y - beta * x
    dof = len(values) - 2
    if dof <= 0:
        return np.nan
    sigma2 = float(np.sum(residual**2) / dof)
    if sigma2 <= 1e-12:
        return 0.0
    se = math.sqrt(sigma2 / denom)
    return beta / se if se > 0 else np.nan


def rolling_trend_tstat(log_price: pd.Series, window: int) -> pd.Series:
    return log_price.rolling(window, min_periods=max(8, window // 2)).apply(trend_tstat, raw=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
