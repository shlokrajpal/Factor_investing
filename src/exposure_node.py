import os
from dotenv import load_dotenv

import pickle
import logging
import warnings
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_streamer import DataStreamer
from exposure_model import (
    KalmanFactorModel,
    KALMAN_ASSET_COLS,
    KALMAN_FACTOR_COLS,
    KALMAN_ALL_COLS,
    TOP_K_EXPOSURES,
    N_ASSETS,
    N_FACTORS,
)

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

INNOV_BUFFER_MAXLEN:    int   = 252    # ~1 trading year rolling buffer per asset
INNOV_WINDOW:           int   = 21    # short window (~1 month) for rolling mean |z|
BREAK_ZSCORE_THRESHOLD: float = 1.25  # mean|z| threshold; 3.44 SE above null mean
BREAK_PERSIST_DAYS:     int   = 5     # consecutive threshold hits → break confirmed
RETRAIN_LOOKBACK_DAYS:  int   = 252   # history window for Q/R re-optimisation

MIN_OFFLINE_DAYS: int = 252

load_dotenv()

NODE_STATE_PATH = os.getenv("KALMAN_NODE_PATH")
PREDICTION_LOG_PATH = os.getenv("KALMAN_PRED_PATH")


class _AssetInnovationMonitor:

    def __init__(self, asset_name: str) -> None:
        self.asset_name              = asset_name
        self._buffer: deque          = deque(maxlen=INNOV_BUFFER_MAXLEN)
        self.persistent_drift_counter_: int  = 0
        self.break_detected_:        bool = False
        self.break_count_:           int  = 0

    def update(self, z: float) -> bool:

        self._buffer.append(float(z))

        if len(self._buffer) < INNOV_WINDOW:
            return False

        recent       = np.asarray(self._buffer)[-INNOV_WINDOW:]
        mean_abs_z   = float(np.mean(np.abs(recent)))
        newly_confirmed = False

        if mean_abs_z > BREAK_ZSCORE_THRESHOLD:
            self.persistent_drift_counter_ += 1

            if self.persistent_drift_counter_ >= BREAK_PERSIST_DAYS:
                if not self.break_detected_:
                    newly_confirmed       = True
                    self.break_detected_  = True
                    self.break_count_    += 1
                    logger.warning(
                        f"[{self.asset_name}] Structural break confirmed "
                        f"(mean|z|={mean_abs_z:.2f} > {BREAK_ZSCORE_THRESHOLD}, "
                        f"persist={self.persistent_drift_counter_}, "
                        f"total_breaks={self.break_count_})"
                    )
        else:
            if self.persistent_drift_counter_ > 0:
                logger.info(
                    f"[{self.asset_name}] Innovation normalised "
                    f"(counter reset from {self.persistent_drift_counter_})"
                )
            self.persistent_drift_counter_ = 0
            self.break_detected_           = False

        return newly_confirmed

    def reset(self) -> None:
        """Clear buffer and counters after a successful retrain."""
        self._buffer.clear()
        self.persistent_drift_counter_ = 0
        self.break_detected_           = False

    def get_state_dict(self) -> Dict:
        return {
            'buffer':             list(self._buffer),
            'persistent_counter': self.persistent_drift_counter_,
            'break_detected':     self.break_detected_,
            'break_count':        self.break_count_,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._buffer                   = deque(state['buffer'], maxlen=INNOV_BUFFER_MAXLEN)
        self.persistent_drift_counter_ = state['persistent_counter']
        self.break_detected_           = state['break_detected']
        self.break_count_              = state['break_count']

class KalmanNode:
    """
    # First run ever — train offline
    node = KalmanNode(model=KalmanFactorModel(), data_dir='./data')
    node.fit_offline(start_date='2019-01-02', end_date='2023-12-29')
    node.save()

    # Every subsequent trading day
    node = KalmanNode.load(data_dir='./data')
    results = node.predict_and_update(target_date='2024-01-02', X=X_daily)
    node.save()
    """

    def __init__(
        self,
        model:    KalmanFactorModel,
        data_dir: str = './data',
    ) -> None:
        self.model    = model
        self.data_dir = data_dir
        self.streamer = DataStreamer(mode='daily', data_dir=data_dir)

        # Per-asset innovation monitors (one per row of B_t)
        self.monitors: Dict[str, _AssetInnovationMonitor] = {
            name: _AssetInnovationMonitor(name)
            for name in model.asset_names
        }

        # Node metadata
        self.last_prediction_date_: Optional[pd.Timestamp] = None
        self.pipeline_version_:     int = 0
        self.total_break_retrains_: int = 0

        os.makedirs('./models', exist_ok=True)
        os.makedirs('./logs',   exist_ok=True)

        logger.info(
            f"KalmanNode initialised: "
            f"{model.n_assets} assets × {model.n_factors} factors  "
            f"B_shape={model.B_.shape}  P_shape={model.P_.shape}"
        )

    @staticmethod
    def _parse_observation(
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
      
        flat = X.flatten()
        if len(flat) != len(KALMAN_ALL_COLS):
            raise ValueError(
                f"X has {len(flat)} features; expected {len(KALMAN_ALL_COLS)} "
                f"({KALMAN_ALL_COLS})"
            )
        y       = flat[:N_ASSETS].astype(float)
        factors = flat[N_ASSETS:].astype(float)
        return y, factors

    @staticmethod
    def _next_day(date_str: str) -> str:
        return (pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    def fit_offline(self, start_date: str, end_date: str) -> 'KalmanNode':
    
        logger.info("=" * 80)
        logger.info("KALMAN NODE — OFFLINE TRAINING")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 80)

        df = self.streamer.stream(
            feature_names=KALMAN_ALL_COLS,
            start_date=start_date,
            end_date=self._next_day(end_date),
        )

        if df is None or len(df) == 0:
            raise ValueError(
                f"[KalmanNode] No daily data in [{start_date}, {end_date}].  "
                "Run FeatureEngineering.process_and_save() first."
            )

        if len(df) < MIN_OFFLINE_DAYS:
            logger.warning(
                f"Offline window = {len(df)} days (<{MIN_OFFLINE_DAYS} recommended)."
            )

        returns_matrix = df[KALMAN_ASSET_COLS].values     # (T, n_assets)
        factors_matrix = df[KALMAN_FACTOR_COLS].values    # (T, n_factors)

        # Optimise Q/R 
        logger.info(
            f"Optimising (Q, R) over {len(df)} days  "
            f"(grid: {len(KALMAN_FACTOR_COLS)}×{len(KALMAN_FACTOR_COLS)} = "
            f"{len(KALMAN_FACTOR_COLS)**2} trials × 9 assets) …"
        )
        q_opt, r_opt = self.model.optimize_hyperparams(
            returns_matrix, factors_matrix
        )
        self.model.apply_hyperparams(q_opt, r_opt)

        # Warm-up pass 
        logger.info(f"Warming up B_/P_ over {len(df)} days …")
        n_processed = 0

        for t in range(len(df)):
            y_row = returns_matrix[t]
            f_row = factors_matrix[t]

            if np.any(np.isnan(y_row)) or np.any(np.isnan(f_row)):
                continue

            result = self.model.step(y_row, f_row)

            # Feed innovations into monitors (no break-trigger during warm-up)
            for i, asset_name in enumerate(self.model.asset_names):
                self.monitors[asset_name].update(result['std_innov'][i])

            n_processed += 1

        self.pipeline_version_      = 1
        self.last_prediction_date_  = df.index[-1]

        logger.info(
            f"Offline training complete: {n_processed}/{len(df)} rows processed  "
            f"(n_updates={self.model.n_updates_})"
        )
        logger.info(f"  B_t range:  [{self.model.B_.min():.4f}, {self.model.B_.max():.4f}]")
        logger.info(f"  P_t traces: {np.einsum('ijj->i', self.model.P_).round(4)}")
        logger.info(f"  Node version: v{self.pipeline_version_}")
        logger.info("=" * 80)

        return self

    def predict_and_update(
        self,
        target_date: str,
        X: np.ndarray,
    ) -> Dict:
     
        if not self.model.is_fitted_:
            raise ValueError(
                "KalmanNode: model not initialised.  "
                "Call fit_offline() first."
            )

        y, factors   = self._parse_observation(X)
        data_date    = pd.Timestamp(target_date)

        result = self.model.step(y, factors)

        std_innov   = result['std_innov']     # (n_assets,)
        confidence  = result['confidence']    # (n_assets,)
        nu          = result['nu']            # (n_assets,) raw innovations

        break_assets: List[str] = []
        for i, asset_name in enumerate(self.model.asset_names):
            newly_confirmed = self.monitors[asset_name].update(std_innov[i])
            if newly_confirmed:
                break_assets.append(asset_name)

        logger.info(
            f"[{target_date}]  "
            f"n_updates={self.model.n_updates_}  "
            f"reliable={self.model.is_reliable_}  "
            f"mean_confidence={confidence.mean():.4f}  "
            f"break_assets={break_assets or 'none'}"
        )

        retrain_triggered = False
        if break_assets:
            logger.warning(
                f"[{target_date}] Structural break in {break_assets} — "
                "triggering localised Q/R retrain."
            )
            retrain_triggered = self._localized_retrain(target_date, break_assets)

        top_exposures = self.model.get_top_exposures(top_k=TOP_K_EXPOSURES)

        n_reliable = int(np.sum(confidence > 0.5))   # assets past burn-in

        master: Dict = {
            'prediction_date':           str(data_date.date()),
            'mean_confidence':           float(confidence.mean()),
            'n_reliable_assets':         n_reliable,
            'is_reliable':               bool(self.model.is_reliable_),
            'n_updates':                 int(self.model.n_updates_),
            'structural_break_detected': len(break_assets) > 0,
            'break_assets':              ','.join(break_assets) if break_assets else 'none',
            'retrain_triggered':         retrain_triggered,
            'node_version':              self.pipeline_version_,
        }

        for i, asset in enumerate(self.model.asset_names):
            pfx = asset
            master[f'{pfx}_confidence'] = float(confidence[i])
            master[f'{pfx}_innovation'] = float(nu[i])
            master[f'{pfx}_std_innov']  = float(std_innov[i])
            master[f'{pfx}_P_trace']    = float(result['P_traces'][i])
            master[f'{pfx}_drift_count'] = int(
                self.monitors[asset].persistent_drift_counter_
            )

            for exp in top_exposures[asset]:
                k = exp['rank']
                master[f'{pfx}_top{k}_factor']   = exp['factor']
                master[f'{pfx}_top{k}_exposure']  = round(exp['exposure'], 6)
                master[f'{pfx}_top{k}_abs_exposure'] = round(exp['abs_exposure'], 6)

        self.last_prediction_date_ = data_date
        self._log_prediction(data_date, result, top_exposures, break_assets)

        return master

    def _localized_retrain(
        self,
        current_date: str,
        break_assets: List[str],
    ) -> bool:

        logger.info("=" * 80)
        logger.info("KALMAN NODE — LOCALISED RETRAIN")
        logger.info(f"  Break assets : {break_assets}")
        logger.info("=" * 80)

        try:
            end_ts   = pd.Timestamp(current_date)
            start_ts = end_ts - pd.Timedelta(days=RETRAIN_LOOKBACK_DAYS)

            df = self.streamer.stream(
                feature_names=KALMAN_ALL_COLS,
                start_date=start_ts.strftime('%Y-%m-%d'),
                end_date=self._next_day(current_date),
            )

            if df is None or len(df) < 30:
                logger.error(
                    f"Insufficient data for retrain "
                    f"({0 if df is None else len(df)} rows).  Aborting."
                )
                return False

            returns_matrix = df[KALMAN_ASSET_COLS].values
            factors_matrix = df[KALMAN_FACTOR_COLS].values

            logger.info(
                f"Re-optimising Q/R over {len(df)} days "
                f"({df.index[0].date()} → {df.index[-1].date()}) …"
            )

            q_opt, r_opt = self.model.optimize_hyperparams(returns_matrix, factors_matrix)
            self.model.apply_hyperparams(q_opt, r_opt)

            # Reset monitors for the breaking assets (they have been recalibrated)
            for asset_name in break_assets:
                if asset_name in self.monitors:
                    self.monitors[asset_name].reset()
                    logger.info(f"  Innovation monitor reset: {asset_name}")

            self.pipeline_version_       += 1
            self.total_break_retrains_   += 1

            logger.info(
                f"Localised retrain complete — Node v{self.pipeline_version_}  "
                f"(total retrains: {self.total_break_retrains_})"
            )
            logger.info("=" * 80)
            return True

        except Exception as exc:
            logger.error(f"Localised retrain failed: {exc}")
            logger.error("Continuing with existing hyperparameters.")
            return False

    def run_walk_forward(self, start_date: str, end_date: str) -> pd.DataFrame:
      
        if not self.model.is_fitted_:
            raise ValueError("Model not fitted.  Call fit_offline() first.")

        logger.info("=" * 80)
        logger.info(f"KALMAN WALK-FORWARD: {start_date} → {end_date}")
        logger.info("=" * 80)

        df = self.streamer.stream(
            feature_names=KALMAN_ALL_COLS,
            start_date=start_date,
            end_date=self._next_day(end_date),
        )

        if df is None or len(df) == 0:
            raise ValueError(f"No data in [{start_date}, {end_date}].")

        end_ts      = pd.Timestamp(end_date)
        all_results = []

        for row_date, row in df.iterrows():
            if row_date > end_ts:
                break
            X      = row[KALMAN_ALL_COLS].values.reshape(1, -1)
            result = self.predict_and_update(str(row_date.date()), X)
            all_results.append(result)

        if not all_results:
            raise ValueError(f"No predictions for [{start_date}, {end_date}].")

        results_df = pd.DataFrame(all_results)
        results_df['prediction_date'] = pd.to_datetime(results_df['prediction_date'])
        results_df = results_df.set_index('prediction_date')

        logger.info(
            f"Walk-forward complete: {len(results_df)} days  "
            f"({results_df.index[0].date()} → {results_df.index[-1].date()})  "
            f"breaks={results_df['structural_break_detected'].sum()}"
        )
        return results_df

    def retrain(self, start_date: str, end_date: str) -> 'KalmanNode':

        logger.info("=" * 80)
        logger.info("KALMAN NODE — MANUAL FULL RETRAIN")
        logger.info("=" * 80)

        # Reset model to a fresh state
        self.model = KalmanFactorModel(
            asset_names=self.model.asset_names,
            factor_names=self.model.factor_names,
        )
        # Reset all monitors
        self.monitors = {
            name: _AssetInnovationMonitor(name)
            for name in self.model.asset_names
        }
        self.total_break_retrains_ = 0

        self.fit_offline(start_date, end_date)
        return self

    def save(self, state_path: str = NODE_STATE_PATH) -> None:

        os.makedirs(os.path.dirname(state_path) or '.', exist_ok=True)

        state = {
            # Full model state (B, P, Q, R, counters)
            'model_state': self.model.get_state_dict(),
            # Per-asset monitor states
            'monitor_states': {
                name: m.get_state_dict()
                for name, m in self.monitors.items()
            },
            # Node metadata
            'last_prediction_date_':  self.last_prediction_date_,
            'pipeline_version_':      self.pipeline_version_,
            'total_break_retrains_':  self.total_break_retrains_,
            # Config for reconstruction
            'asset_names':            self.model.asset_names,
            'factor_names':           self.model.factor_names,
        }

        with open(state_path, 'wb') as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"KalmanNode saved → {state_path}  "
            f"(v{self.pipeline_version_}  "
            f"n_updates={self.model.n_updates_}  "
            f"B_shape={self.model.B_.shape}  "
            f"P_shape={self.model.P_.shape})"
        )

    @classmethod
    def load(
        cls,
        data_dir:   str = './data',
        state_path: str = NODE_STATE_PATH,
    ) -> 'KalmanNode':
       
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No saved state at: {state_path}")

        with open(state_path, 'rb') as fh:
            state = pickle.load(fh)

        # Build model shell (hyperparams overwritten by load_state_dict)
        model = KalmanFactorModel(
            asset_names=state['asset_names'],
            factor_names=state['factor_names'],
        )
        model.load_state_dict(state['model_state'])

        node = cls(model=model, data_dir=data_dir)

        # Restore per-asset innovation monitors
        for name, mon_state in state.get('monitor_states', {}).items():
            if name in node.monitors:
                node.monitors[name].load_state_dict(mon_state)

        # Restore node metadata
        node.last_prediction_date_  = state['last_prediction_date_']
        node.pipeline_version_      = state['pipeline_version_']
        node.total_break_retrains_  = state.get('total_break_retrains_', 0)

        logger.info(
            f"KalmanNode loaded ← {state_path}  "
            f"v{node.pipeline_version_}  "
            f"B_shape={model.B_.shape}  "
            f"P_shape={model.P_.shape}  "
            f"last={node.last_prediction_date_}"
        )
        return node

    def _log_prediction(
        self,
        data_date:     pd.Timestamp,
        step_result:   Dict,
        top_exposures: Dict,
        break_assets:  List[str],
    ) -> None:
        with open(PREDICTION_LOG_PATH, 'a') as fh:
            for i, asset in enumerate(self.model.asset_names):
                top3 = ','.join(e['factor'] for e in top_exposures[asset][:3])
                fh.write(
                    f"{datetime.now().isoformat()},"
                    f"{data_date.date()},"
                    f"{asset},"
                    f"z={step_result['std_innov'][i]:.3f},"
                    f"conf={step_result['confidence'][i]:.4f},"
                    f"break={'YES' if asset in break_assets else 'no'},"
                    f"drift={self.monitors[asset].persistent_drift_counter_},"
                    f"top3=[{top3}],"
                    f"v{self.pipeline_version_}\n"
                )

    def get_info(self) -> Dict:
        monitor_summary = {
            name: {
                'break_detected':    m.break_detected_,
                'break_count':       m.break_count_,
                'drift_counter':     m.persistent_drift_counter_,
                'buffer_size':       len(m._buffer),
            }
            for name, m in self.monitors.items()
        }

        return {
            'node': {
                'version':              self.pipeline_version_,
                'total_break_retrains': self.total_break_retrains_,
                'last_prediction':      (
                    str(self.last_prediction_date_.date())
                    if self.last_prediction_date_ else None
                ),
                'config': {
                    'innov_buffer_maxlen':    INNOV_BUFFER_MAXLEN,
                    'innov_window':           INNOV_WINDOW,
                    'break_zscore_threshold': BREAK_ZSCORE_THRESHOLD,
                    'break_persist_days':     BREAK_PERSIST_DAYS,
                    'retrain_lookback_days':  RETRAIN_LOOKBACK_DAYS,
                },
                'monitors': monitor_summary,
            },
            'model': self.model.get_info(),
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    node = KalmanNode(
        model=KalmanFactorModel(),
        data_dir='./data',
    )

    # 1. Train on 4 years of daily history
    node.fit_offline(start_date='2019-01-02', end_date='2023-12-29')
    node.save()

    # 2. Reload and run a walk-forward backtest
    node = KalmanNode.load(data_dir='./data')

    preds = node.run_walk_forward(
        start_date='2024-01-02',
        end_date='2025-12-31',
    )

    # Show headline stats
    print(preds[[
        'mean_confidence', 'n_reliable_assets',
        'is_reliable', 'structural_break_detected',
    ]].head(10).to_string())
    print(f"\nTotal days: {len(preds)}")
    print(f"Structural break days: {preds['structural_break_detected'].sum()}")

    # Show gold's current top exposures
    node2 = KalmanNode.load(data_dir='./data')
    gld_top = node2.model.get_top_exposures()['gld_log_return']
    print(f"\nGold (gld_log_return) top {TOP_K_EXPOSURES} factor exposures:")
    for e in gld_top:
        print(f"  #{e['rank']}  {e['factor']:30s}  β={e['exposure']:+.4f}  |β|={e['abs_exposure']:.4f}")