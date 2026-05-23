#!/usr/bin/env python3
"""Run the optional BUSI70575 strategy-construction bonus track."""

from __future__ import annotations

import json

from coursework.src.config import CONFIG
from coursework.src.strategy import run_strategy_backtest


def main() -> None:
    result = run_strategy_backtest(CONFIG)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
