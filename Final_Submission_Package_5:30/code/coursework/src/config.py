"""Central configuration for the BUSI70575 metamodel pipeline.

This module stores paths, dates, feature flags, model grids, label defaults,
and the frozen final-submission metadata used across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_STATE = 42

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_OHLCV_PATH = WORKSPACE_ROOT / "ohlcv_data.csv"
DEFAULT_PRIMARY_SIGNALS_PATH = WORKSPACE_ROOT / "primary_signals.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
DEFAULT_PREDICTION_START = "2022-01-01"
DEFAULT_PREDICTION_END = "2022-06-30"

DEFAULT_VERTICAL_BARRIER_DAYS = 10
DEFAULT_PROFIT_TAKING_MULTIPLIER = 1.5
DEFAULT_STOP_LOSS_MULTIPLIER = 1.5
DEFAULT_VOL_LOOKBACK_DAYS = 60
DEFAULT_MIN_TRAIN_EVENTS = 30

USE_SIDE_ADJUSTED_FEATURES = True
USE_SIGNAL_HISTORY_FEATURES = True
USE_VOL_STRESS_FEATURES = True
USE_TREND_SCANNING_FEATURES = True
USE_REGIME_INTERACTION_FEATURES = True
USE_HMM_EXTENSION = False

INSTRUMENTS = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]

ASSET_CLASS = {
    "es1s": "equity_index",
    "nq1s": "equity_index",
    "fesx1s": "equity_index",
    "cl1s": "energy",
    "ho1s": "energy",
    "rb1s": "energy",
    "ng1s": "energy",
    "gc1s": "metals",
    "si1s": "metals",
    "hg1s": "metals",
    "pl1s": "metals",
}

ID_COLS = {
    "date",
    "instrument",
    "asset_class",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "primary_signal",
    "label",
    "label_end_date",
    "label_reason",
    "side_return",
    "holding_days",
}

SIDE_ADJUSTED_FEATURES = [
    "side_return_1d",
    "side_return_5d",
    "side_momentum_20d",
    "side_vol_adj_momentum",
    "side_rsi",
    "side_macd",
    "side_bollinger_position",
]

SIGNAL_HISTORY_FEATURES = [
    "signal_hit_rate_20d",
    "signal_hit_rate_60d",
    "signal_hit_rate_120d",
    "signal_count_60d",
    "signal_avg_return_60d",
    "signal_win_rate_long_60d",
    "signal_win_rate_short_60d",
    "signal_recent_fail_streak",
    "signal_recent_success_streak",
]

VOL_STRESS_FEATURES = [
    "vol_percentile_63d",
    "vol_percentile_252d",
    "atr_percentile_252d",
    "range_percentile_63d",
    "drawdown_20d",
    "drawdown_60d",
    "drawdown_252d",
    "vol_of_vol_20d",
    "gap_return",
    "overnight_return",
    "intraday_reversal",
]

TREND_SCANNING_FEATURES = [
    "trend_scan_tstat_20",
    "trend_scan_tstat_60",
    "trend_scan_best_window",
    "trend_scan_sign",
    "trend_scan_abs_tstat",
    "side_trend_tstat",
]

REGIME_INTERACTION_FEATURES = [
    "gmm_state_0_x_signal",
    "gmm_state_1_x_signal",
    "gmm_state_2_x_signal",
    "pca_ohlcv_1_x_signal",
    "pca_ohlcv_2_x_signal",
    "pca_ohlcv_3_x_signal",
    "gmm_max_prob",
    "gmm_state_change",
    "regime_conditioned_hit_rate_60d",
]

HMM_EXTENSION_FEATURES = [
    "hmm_state_0_prob",
    "hmm_state_1_prob",
    "hmm_state_2_prob",
    "hmm_entropy",
    "hmm_state_id",
    "hmm_days_in_state",
    "hmm_state_change_prob",
]

CONTROLLED_FEATURE_GROUPS = {
    "side_adjusted": SIDE_ADJUSTED_FEATURES,
    "signal_history": SIGNAL_HISTORY_FEATURES,
    "vol_stress": VOL_STRESS_FEATURES,
    "trend_scanning": TREND_SCANNING_FEATURES,
    "regime_interactions": REGIME_INTERACTION_FEATURES,
    "hmm_extension": HMM_EXTENSION_FEATURES,
}

DEFAULT_MODEL_GRID = {
    "logistic": [{"C": c} for c in (0.1, 1.0)],
    "random_forest": [
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 12, "max_features": "sqrt"},
    ],
    "extra_trees": [
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 12, "max_features": "sqrt"},
    ],
    "hist_gradient_boosting": [
        {"max_iter": 120, "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 0.1},
        {"max_iter": 120, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
    ],
    "adaboost": [{"n_estimators": 120, "learning_rate": 0.05}],
    "mlp": [{"hidden_layer_sizes": (24,), "alpha": 1e-3, "learning_rate_init": 1e-3}],
}

SMALL_SAMPLE_MODEL_GRID = {
    "logistic": [{"C": 0.3}, {"C": 1.0}],
    "random_forest": [{"n_estimators": 250, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"}],
    "extra_trees": [{"n_estimators": 250, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"}],
    "hist_gradient_boosting": [{"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 0.1}],
    "adaboost": [{"n_estimators": 80, "learning_rate": 0.05}],
    "mlp": [{"hidden_layer_sizes": (16,), "alpha": 1e-3, "learning_rate_init": 1e-3}],
}

CURRENT_TRIPLE_BARRIER_DEFAULTS = {
    "vertical_barrier_days": DEFAULT_VERTICAL_BARRIER_DAYS,
    "profit_taking_multiplier": DEFAULT_PROFIT_TAKING_MULTIPLIER,
    "stop_loss_multiplier": DEFAULT_STOP_LOSS_MULTIPLIER,
    "vol_lookback_days": DEFAULT_VOL_LOOKBACK_DAYS,
    "hmm_enabled": USE_HMM_EXTENSION,
}

LABEL_SEARCH_GRID = {
    "vertical_barrier_days": [5, 10, 15, 20],
    "profit_taking_multiplier": [1.0, 1.5, 2.0],
    "stop_loss_multiplier": [1.0, 1.5, 2.0],
    "vol_lookback_days": [10, 20, 40, 60],
}

STRATEGY_METHODS = [
    "threshold_filter_050",
    "threshold_filter_055",
    "threshold_filter_060",
    "confidence_linear",
    "confidence_scaled",
    "soft_allocation",
]
STRATEGY_COST_BPS_GRID = [0, 1, 2, 5]
STRATEGY_GROSS_EXPOSURE_TARGET = 1.0
STRATEGY_MAX_ABS_WEIGHT_PER_INSTRUMENT = 0.25
STRATEGY_TARGET_VOL = 0.10
STRATEGY_TARGET_VOL_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
STRATEGY_COV_LOOKBACK = 60
STRATEGY_MAX_LEVERAGE = 2.0
TRADING_DAYS_PER_YEAR = 252

PROMOTED_FINAL_DESCRIPTION = "Calibrated 0.50 Logistic + 0.50 signal-history MLP blend"
PROMOTED_FINAL_MEAN_AUC = 0.583
PROMOTED_FINAL_MEAN_F1 = 0.626
PROMOTED_FINAL_HASH = "c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369"


@dataclass(frozen=True)
class CourseworkConfig:
    """Central configuration for the final coursework submission pipeline."""

    ohlcv_path: Path = DEFAULT_OHLCV_PATH
    primary_signals_path: Path = DEFAULT_PRIMARY_SIGNALS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prediction_start: str = DEFAULT_PREDICTION_START
    prediction_end: str = DEFAULT_PREDICTION_END
    vertical_barrier_days: int = DEFAULT_VERTICAL_BARRIER_DAYS
    profit_taking_multiplier: float = DEFAULT_PROFIT_TAKING_MULTIPLIER
    stop_loss_multiplier: float = DEFAULT_STOP_LOSS_MULTIPLIER
    vol_lookback_days: int = DEFAULT_VOL_LOOKBACK_DAYS
    min_train_events: int = DEFAULT_MIN_TRAIN_EVENTS
    seed: int = RANDOM_STATE
    enable_hmm: bool = USE_HMM_EXTENSION
    final_model_name: str = PROMOTED_FINAL_DESCRIPTION
    promoted_final_mean_auc: float = PROMOTED_FINAL_MEAN_AUC
    promoted_final_mean_f1: float = PROMOTED_FINAL_MEAN_F1
    promoted_final_hash: str = PROMOTED_FINAL_HASH
    strategy_methods: tuple[str, ...] = tuple(STRATEGY_METHODS)
    strategy_cost_bps_grid: tuple[int, ...] = tuple(STRATEGY_COST_BPS_GRID)
    strategy_gross_exposure_target: float = STRATEGY_GROSS_EXPOSURE_TARGET
    strategy_max_abs_weight_per_instrument: float = STRATEGY_MAX_ABS_WEIGHT_PER_INSTRUMENT
    strategy_target_vol: float = STRATEGY_TARGET_VOL
    strategy_target_vol_grid: tuple[float, ...] = tuple(STRATEGY_TARGET_VOL_GRID)
    strategy_cov_lookback: int = STRATEGY_COV_LOOKBACK
    strategy_max_leverage: float = STRATEGY_MAX_LEVERAGE
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR

    @property
    def final_prediction_path(self) -> Path:
        return self.output_dir / "metamodel_predictions.csv"

    @property
    def promoted_blend_source_path(self) -> Path:
        return self.output_dir / "metamodel_predictions.csv"

    @property
    def evaluation_summary_path(self) -> Path:
        return self.output_dir / "evaluation_summary.csv"

    @property
    def model_comparison_path(self) -> Path:
        return self.output_dir / "model_comparison.csv"

    @property
    def cluster_importance_path(self) -> Path:
        return self.output_dir / "cluster_importance.csv"

    @property
    def strategy_weights_path(self) -> Path:
        return self.output_dir / "strategy_weights.csv"

    @property
    def strategy_backtest_metrics_path(self) -> Path:
        return self.output_dir / "strategy_backtest_metrics.csv"

    @property
    def strategy_daily_returns_path(self) -> Path:
        return self.output_dir / "strategy_daily_returns.csv"

    @property
    def strategy_turnover_path(self) -> Path:
        return self.output_dir / "strategy_turnover.csv"

    @property
    def strategy_summary_path(self) -> Path:
        return self.output_dir / "strategy_summary.md"

    @property
    def strategy_audit_path(self) -> Path:
        return self.output_dir / "strategy_bonus_audit.txt"

    @property
    def strategy_target_vol_results_path(self) -> Path:
        return self.output_dir / "strategy_target_vol_results.csv"

    @property
    def strategy_target_vol_summary_path(self) -> Path:
        return self.output_dir / "strategy_target_vol_summary.md"

    @property
    def strategy_target_vol_audit_path(self) -> Path:
        return self.output_dir / "strategy_target_vol_audit.txt"

    @property
    def strategy_all_methods_comparison_path(self) -> Path:
        return self.output_dir / "strategy_all_methods_comparison.csv"

    @property
    def strategy_all_methods_comparison_md_path(self) -> Path:
        return self.output_dir / "strategy_all_methods_comparison.md"


CONFIG = CourseworkConfig()
