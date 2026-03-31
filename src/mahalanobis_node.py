import numpy as np
import pandas as pd
import pickle
from collections import deque
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from scipy import stats
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from data_streamer import DataStreamer

def compute_psi(reference_dist: np.ndarray, current_dist: np.ndarray, bins: int = 10) -> float:
    combined = np.concatenate([reference_dist, current_dist])
    bin_edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference_dist, bins=bin_edges)
    curr_counts, _ = np.histogram(current_dist, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / (len(reference_dist) + bins * 1e-6)
    curr_pct = (curr_counts + 1e-6) / (len(current_dist) + bins * 1e-6)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


def compute_ks_test(reference_dist: np.ndarray, current_dist: np.ndarray) -> Tuple[float, float]:
    ks_stat, p_value = stats.ks_2samp(reference_dist, current_dist)
    return float(ks_stat), float(p_value)

FULL_RETRAIN_PSI_THRESHOLD = 0.25
FULL_RETRAIN_KS_PVALUE = 0.001

SCORING_EFFECTIVE_DOF = 7

BASELINE_REFRESH_WEEKS = 12
BASELINE_REFRESH_STRESS_THRESHOLD = 0.33

STALE_BASELINE_WEEKS = 26

REFERENCE_UPDATE_WEEKS = 26
REFERENCE_UPDATE_PSI_THRESHOLD = 0.1

DRIFT_WINDOW_SIZE = 26
PERSISTENT_DRIFT_THRESHOLD = 5
AUTO_RETRAIN_LOOKBACK_YEARS = 4

NORMAL_HALF_LIFE_WEEKS = 12
POST_CRISIS_HALF_LIFE_WEEKS = 4
POST_STRESS_HALF_LIFE_WEEKS = 8

NODE_STATE_PATH = os.getenv("MAHA_NODE_PATH")
PREDICTION_LOG_PATH = os.getenv("MAHA_PRED_PATH")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MahalanobisNode:

    def __init__(self, model, feature_names: list, data_dir: str = "./data"):
        self.model = model
        self.feature_names = feature_names
        # DataStreamer is instantiated in weekly mode; all feature I/O uses weekly_features.csv.
        self.streamer = DataStreamer(mode='weekly', data_dir=data_dir)

        # Drift monitoring
        self.original_reference_distances_ = None
        self.reference_distances_ = None
        self.reference_update_date_ = None
        self.weeks_since_reference_update_ = 0

        self.recent_distances_buffer_ = deque(maxlen=DRIFT_WINDOW_SIZE)
        self.persistent_drift_counter_ = 0

        # Baseline refresh
        self.calm_regime_buffer_ = deque(maxlen=104)
        self.weeks_of_calm_data_ = 0
        self.weeks_since_baseline_update_ = 0
        self.previous_regime_ = 'calm'

        # Metadata
        self.last_prediction_date_ = None
        self.pipeline_version_ = 0

        os.makedirs('./models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)

        logger.info(
            f"Initialized MahalanobisNode  model={type(model).__name__}  "
            f"features={len(feature_names)}"
        )

    def _compute_alpha(self, half_life: float) -> float:
        """Convert a half-life (in weeks) to an exponential smoothing alpha."""
        return float(np.exp(-np.log(2) / half_life))

    def _get_next_week(self, date_str: str) -> str:
        return (pd.Timestamp(date_str) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')

    def _get_previous_week(self, date_str: str) -> str:
        return (pd.Timestamp(date_str) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')

    def _log_prediction(
        self,
        prediction_date: pd.Timestamp,
        stress_score: float,
        regime: str,
        contributions: np.ndarray,
        drift_info: Dict
    ) -> None:
        top3_indices = np.argsort(contributions)[-3:][::-1]
        top3_factors = ','.join(self.feature_names[i] for i in top3_indices)

        psi = drift_info.get('metrics', {}).get('psi', float('nan'))
        ks_pval = drift_info.get('metrics', {}).get('ks_pvalue', float('nan'))

        with open(PREDICTION_LOG_PATH, 'a') as f:
            f.write(
                f"{datetime.now().isoformat()},"
                f"{prediction_date.date()},"
                f"{stress_score:.6f},"
                f"{regime},"
                f"node_v{self.pipeline_version_},"
                f"top_factors=[{top3_factors}],"
                f"psi={psi:.4f},"
                f"ks_pval={ks_pval:.4f}\n"
            )

    def _check_drift(self, current_distance: float) -> Dict:
        self.recent_distances_buffer_.append(current_distance)

        if self.reference_distances_ is None:
            return {
                'drift_detected': False, 'action': 'none',
                'reason': 'No reference data available', 'metrics': {}
            }

        if len(self.recent_distances_buffer_) < DRIFT_WINDOW_SIZE:
            return {
                'drift_detected': False, 'action': 'none',
                'reason': (f'Accumulating data '
                           f'({len(self.recent_distances_buffer_)}/{DRIFT_WINDOW_SIZE})'),
                'metrics': {
                    'buffer_fill': len(self.recent_distances_buffer_) / DRIFT_WINDOW_SIZE
                }
            }

        current_dist_array = np.array(self.recent_distances_buffer_)
        psi = compute_psi(self.reference_distances_, current_dist_array)
        ks_stat, ks_pvalue = compute_ks_test(self.reference_distances_, current_dist_array)

        action = 'none'
        reason = []

        if psi > FULL_RETRAIN_PSI_THRESHOLD or ks_pvalue < FULL_RETRAIN_KS_PVALUE:
            self.persistent_drift_counter_ += 1

            if self.persistent_drift_counter_ >= PERSISTENT_DRIFT_THRESHOLD:
                action = 'full_retrain'
                reason.append(f'Persistent drift ({self.persistent_drift_counter_} consecutive checks)')
            else:
                action = 'monitor'
                reason.append(f'Drift detected ({self.persistent_drift_counter_}/{PERSISTENT_DRIFT_THRESHOLD})')

            if psi > FULL_RETRAIN_PSI_THRESHOLD:
                reason.append(f'PSI {psi:.4f} > {FULL_RETRAIN_PSI_THRESHOLD}')
            if ks_pvalue < FULL_RETRAIN_KS_PVALUE:
                reason.append(f'KS p-value {ks_pvalue:.4f} < {FULL_RETRAIN_KS_PVALUE}')
        else:
            if self.persistent_drift_counter_ > 0:
                logger.info(f"Drift subsided (counter reset from {self.persistent_drift_counter_})")
            self.persistent_drift_counter_ = 0

        if action == 'none' and (
            self.weeks_since_reference_update_ >= REFERENCE_UPDATE_WEEKS
            or psi > REFERENCE_UPDATE_PSI_THRESHOLD
        ):
            action = 'reference_update'
            if self.weeks_since_reference_update_ >= REFERENCE_UPDATE_WEEKS:
                reason.append(f'{self.weeks_since_reference_update_} weeks since last update')
            if psi > REFERENCE_UPDATE_PSI_THRESHOLD:
                reason.append(f'PSI {psi:.4f} > {REFERENCE_UPDATE_PSI_THRESHOLD}')

        drift_info = {
            'drift_detected': action in ['full_retrain', 'monitor', 'reference_update'],
            'action': action,
            'reason': '; '.join(reason) if reason else 'No significant drift',
            'metrics': {
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'weeks_since_reference_update': self.weeks_since_reference_update_,
                'persistent_drift_count': self.persistent_drift_counter_
            }
        }

        if action != 'none':
            logger.info(f"Drift check: PSI={psi:.4f}, KS_pval={ks_pvalue:.4f} → {action}")

        return drift_info

    def _accumulate_calm_data(self, date: pd.Timestamp, X: np.ndarray, stress_score: float) -> None:
        if stress_score < BASELINE_REFRESH_STRESS_THRESHOLD:
            X_flat = X.flatten() if X.ndim > 1 else X
            self.calm_regime_buffer_.append((date, X_flat))
            self.weeks_of_calm_data_ += 1

    def _check_and_refresh_baseline(self) -> None:

        has_calm_data = self.weeks_of_calm_data_ >= BASELINE_REFRESH_WEEKS
        baseline_is_stale = self.weeks_since_baseline_update_ >= STALE_BASELINE_WEEKS

        if not (has_calm_data or baseline_is_stale):
            return

        if has_calm_data:
            calm_data = np.array([x for _, x in self.calm_regime_buffer_])
            logger.info(f"Baseline refresh: {len(calm_data)} calm weeks accumulated")
        else:
            if len(self.calm_regime_buffer_) == 0:
                logger.warning("Stale baseline but no data in buffer - skipping refresh")
                return
            calm_data = np.array([x for _, x in self.calm_regime_buffer_])
            logger.warning(
                f" STALE BASELINE OVERRIDE: Forcing refresh with {len(calm_data)} "
                f"recent samples (baseline hasn't updated in "
                f"{self.weeks_since_baseline_update_} weeks)"
            )

        if baseline_is_stale and not has_calm_data:
            alpha = self._compute_alpha(POST_CRISIS_HALF_LIFE_WEEKS)
            logger.info(f"Adaptive alpha={alpha:.4f} (stale baseline override, "
                        f"half_life={POST_CRISIS_HALF_LIFE_WEEKS}w)")
        elif self.previous_regime_ == 'extreme':
            alpha = self._compute_alpha(POST_CRISIS_HALF_LIFE_WEEKS)
            logger.info(f"Adaptive alpha={alpha:.4f} (post-crisis fast-forget, "
                        f"half_life={POST_CRISIS_HALF_LIFE_WEEKS}w)")
        elif self.previous_regime_ == 'elevated':
            alpha = self._compute_alpha(POST_STRESS_HALF_LIFE_WEEKS)
            logger.info(f"Adaptive alpha={alpha:.4f} (post-stress moderate-forget, "
                        f"half_life={POST_STRESS_HALF_LIFE_WEEKS}w)")
        else:
            alpha = self._compute_alpha(NORMAL_HALF_LIFE_WEEKS)
            logger.info(f"Adaptive alpha={alpha:.4f} (normal regime, "
                        f"half_life={NORMAL_HALF_LIFE_WEEKS}w)")

        # Pass only the float; model knows nothing about regimes
        self.model.update_baseline(calm_data, alpha=alpha)

        self.weeks_of_calm_data_ = 0
        self.weeks_since_baseline_update_ = 0
        self.pipeline_version_ += 1

        logger.info(f"Baseline refreshed - Node v{self.pipeline_version_}")

    def _update_reference_window(self, data_date: pd.Timestamp, X: np.ndarray) -> None:
        """Store the 1-D distance for drift monitoring (not raw features)."""
        distance = self.model.score(X)[0]

        if self.reference_distances_ is None:
            self.reference_distances_ = np.array([distance])
        else:
            max_ref_size = 1000
            if len(self.reference_distances_) >= max_ref_size:
                keep_size = int(max_ref_size * 0.9)
                self.reference_distances_ = np.append(
                    self.reference_distances_[-keep_size:], distance
                )
            else:
                self.reference_distances_ = np.append(self.reference_distances_, distance)

        self.reference_update_date_ = data_date
        self.weeks_since_reference_update_ = 0
        logger.info(f"Reference window updated (size={len(self.reference_distances_)})")

    def _auto_retrain(self, current_date: str) -> None:
        """Full retrain triggered by persistent drift.  Fetches its own history."""
        logger.info("=" * 80)
        logger.info(" AUTOMATIC RETRAINING TRIGGERED")
        logger.info("=" * 80)

        end_date = pd.Timestamp(current_date)
        start_date = end_date - pd.DateOffset(years=AUTO_RETRAIN_LOOKBACK_YEARS)
        logger.info(f"Auto-selected training window: {start_date.date()} to {end_date.date()}")

        try:
            df = self.streamer.stream(
                feature_names=self.feature_names,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=self._get_next_week(end_date.strftime('%Y-%m-%d'))
            )

            if df is None or len(df) == 0:
                logger.error("Insufficient data for retrain — falling back to baseline refresh")
                self._check_and_refresh_baseline()
                return

            X = df.values
            logger.info(f"Retraining model with {len(X)} samples...")
            self.model.fit(X)

            distances = self.model.score(X)
            reference_size = min(len(X), 1000)
            reference_indices = np.random.choice(len(X), reference_size, replace=False)

            self.original_reference_distances_ = distances[reference_indices].copy()
            self.reference_distances_ = self.original_reference_distances_.copy()
            self.reference_update_date_ = df.index[-1]
            self.weeks_since_reference_update_ = 0

            self.recent_distances_buffer_.clear()
            self.persistent_drift_counter_ = 0

            # Rebuild calm buffer from newly trained predictions
            pred = self.model.predict(X, effective_dof=None)
            calm_mask = pred['stress_score'] < BASELINE_REFRESH_STRESS_THRESHOLD
            calm_indices = np.where(calm_mask)[0]

            self.calm_regime_buffer_ = deque(
                [(df.index[i], X[i]) for i in calm_indices],
                maxlen=104
            )
            self.weeks_of_calm_data_ = len(calm_indices)
            self.pipeline_version_ += 1

            logger.info(f" Auto-retrain complete - Node v{self.pipeline_version_}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Auto-retrain failed: {e}")
            logger.error("Continuing with existing model...")

    def fit_offline(self, start_date: str, end_date: str) -> 'MahalanobisNode':
        logger.info("=" * 80)
        logger.info("OFFLINE TRAINING")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 80)

        df = self.streamer.stream(
            feature_names=self.feature_names,
            start_date=start_date,
            end_date=self._get_next_week(end_date)
        )

        if df is None or len(df) == 0:
            raise ValueError(f"No data available for training period [{start_date}, {end_date}]")

        X = df.values
        self.model.fit(X)

        distances = self.model.score(X)
        reference_size = min(len(X), 1000)
        reference_indices = np.random.choice(len(X), reference_size, replace=False)

        self.original_reference_distances_ = distances[reference_indices].copy()
        self.reference_distances_ = self.original_reference_distances_.copy()
        self.reference_update_date_ = df.index[-1]

        pred = self.model.predict(X, effective_dof=None)
        calm_mask = pred['stress_score'] < BASELINE_REFRESH_STRESS_THRESHOLD
        calm_indices = np.where(calm_mask)[0]

        self.calm_regime_buffer_ = deque(
            [(df.index[i], X[i]) for i in calm_indices],
            maxlen=104
        )
        self.weeks_of_calm_data_ = len(calm_indices)
        self.pipeline_version_ = 1

        logger.info(f"Offline training complete - Node v{self.pipeline_version_}")
        logger.info("=" * 80)

        return self

    def predict_and_update(self, target_date: str, X: np.ndarray) -> dict:

        if not self.model.is_fitted_:
            raise ValueError("Model not fitted. Call fit_offline() first.")

        data_date = pd.Timestamp(target_date)
        prediction_date = pd.Timestamp(self._get_next_week(target_date))

        pred = self.model.predict(X, effective_dof=SCORING_EFFECTIVE_DOF)

        stress_score = float(pred['stress_score'][0])
        regime = str(pred['regime'][0])
        distance = float(pred['mahalanobis_distance'][0])
        contributions = pred['contributions'][0]
        outlier_count = int(pred['outlier_count'][0])

        logger.info(
            f"[{target_date}] → prediction for {prediction_date.date()}: "
            f"stress_score={stress_score:.6f}  regime={regime}  "
            f"outlier_count={outlier_count}"
        )

        drift_info: Dict = {'metrics': {}}

        # 1. Drift check
        drift_info = self._check_drift(distance)

        # 2. Auto-retrain on persistent severe drift
        if drift_info['action'] == 'full_retrain':
            logger.warning(" SEVERE DRIFT DETECTED - Initiating automatic retraining")
            self._auto_retrain(target_date)

            # Re-score with the new model
            pred = self.model.predict(X, effective_dof=SCORING_EFFECTIVE_DOF)
            stress_score = float(pred['stress_score'][0])
            regime = str(pred['regime'][0])
            distance = float(pred['mahalanobis_distance'][0])
            contributions = pred['contributions'][0]
            outlier_count = int(pred['outlier_count'][0])
            logger.info(
                f"Re-prediction after retrain: "
                f"stress_score={stress_score:.6f}  regime={regime}"
            )

        # 3. Reference window rotation on minor drift
        elif drift_info['action'] == 'reference_update':
            self._update_reference_window(data_date, X)

        # 4. Accumulate calm data
        self._accumulate_calm_data(data_date, X, stress_score)

        # 5. Baseline refresh (node selects alpha via half-life math)
        self._check_and_refresh_baseline()

        # 6. Regime transition tracking
        if regime != self.previous_regime_:
            logger.info(
                f"Regime transition: {self.previous_regime_} → {regime} "
                f"(previous_regime_ updated for next baseline refresh)"
            )
        self.previous_regime_ = regime

        # 7. Increment week counters
        self.weeks_since_reference_update_ += 1
        self.weeks_since_baseline_update_ += 1
        self.last_prediction_date_ = data_date

        self._log_prediction(prediction_date, stress_score, regime, contributions, drift_info)

        return {
            'prediction_date': str(prediction_date.date()),
            'stress_score': stress_score,
            'regime': regime,
            'mahalanobis_distance': distance,
            'outlier_count': outlier_count,
            'drift_action': drift_info.get('action', 'none'),
            'psi': drift_info.get('metrics', {}).get('psi', float('nan')),
            'ks_pvalue': drift_info.get('metrics', {}).get('ks_pvalue', float('nan')),
            'node_version': self.pipeline_version_,
        }

    def run_walk_forward(self, start_date: str, end_date: str) -> pd.DataFrame:

        if not self.model.is_fitted_:
            raise ValueError("Model not fitted. Call fit_offline() first.")

        logger.info("=" * 80)
        logger.info(f"WALK-FORWARD BACKTEST: {start_date} → {end_date}")
        logger.info("=" * 80)

        df = self.streamer.stream(
            feature_names=self.feature_names,
            start_date=start_date,
            end_date=self._get_next_week(end_date)
        )

        if df is None or len(df) == 0:
            raise ValueError(f"No data available for [{start_date}, {end_date}]")

        end_ts = pd.Timestamp(end_date)
        all_results = []

        for row_date, row in df.iterrows():
            if pd.Timestamp(row_date) > end_ts:
                break

            X = row.values.reshape(1, -1)
            result = self.predict_and_update(str(row_date.date()), X)
            all_results.append(result)

        if not all_results:
            raise ValueError(f"No predictions generated for range [{start_date}, {end_date}]")

        results_df = pd.DataFrame(all_results)
        results_df['prediction_date'] = pd.to_datetime(results_df['prediction_date'])
        results_df = results_df.set_index('prediction_date')

        logger.info(
            f"\nWalk-forward complete: {len(results_df)} predictions from "
            f"{results_df.index[0].date()} to {results_df.index[-1].date()}"
        )
        logger.info("=" * 80)

        return results_df

    def retrain(self, start_date: str, end_date: str) -> 'MahalanobisNode':
        logger.info("=" * 80)
        logger.info("MANUAL RETRAIN TRIGGERED")
        logger.info("=" * 80)

        df = self.streamer.stream(
            feature_names=self.feature_names,
            start_date=start_date,
            end_date=self._get_next_week(end_date)
        )

        if df is None or len(df) == 0:
            raise ValueError(f"No data available for training period [{start_date}, {end_date}]")

        X = df.values
        self.model.fit(X)

        distances = self.model.score(X)
        reference_size = min(len(X), 1000)
        reference_indices = np.random.choice(len(X), reference_size, replace=False)

        self.original_reference_distances_ = distances[reference_indices].copy()
        self.reference_distances_ = self.original_reference_distances_.copy()
        self.reference_update_date_ = df.index[-1]
        self.weeks_since_reference_update_ = 0

        self.recent_distances_buffer_.clear()
        self.persistent_drift_counter_ = 0

        pred = self.model.predict(X, effective_dof=None)
        calm_mask = pred['stress_score'] < BASELINE_REFRESH_STRESS_THRESHOLD
        calm_indices = np.where(calm_mask)[0]

        self.calm_regime_buffer_ = deque(
            [(df.index[i], X[i]) for i in calm_indices],
            maxlen=104
        )
        self.weeks_of_calm_data_ = len(calm_indices)
        self.weeks_since_baseline_update_ = 0
        self.pipeline_version_ += 1

        logger.info(f"Manual retrain complete - Node v{self.pipeline_version_}")
        logger.info("=" * 80)

        return self

    def save(self, state_path: str = NODE_STATE_PATH) -> None:
        os.makedirs(os.path.dirname(state_path) or '.', exist_ok=True)

        state = {
            'model': self.model,
            'feature_names': self.feature_names,
            'original_reference_distances_': self.original_reference_distances_,
            'reference_distances_': self.reference_distances_,
            'reference_update_date_': self.reference_update_date_,
            'weeks_since_reference_update_': self.weeks_since_reference_update_,
            'recent_distances_buffer_': list(self.recent_distances_buffer_),
            'persistent_drift_counter_': self.persistent_drift_counter_,
            'calm_regime_buffer_': list(self.calm_regime_buffer_),
            'weeks_of_calm_data_': self.weeks_of_calm_data_,
            'weeks_since_baseline_update_': self.weeks_since_baseline_update_,
            'previous_regime_': self.previous_regime_,
            'pipeline_version_': self.pipeline_version_,
            'last_prediction_date_': self.last_prediction_date_,
        }

        with open(state_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Node saved: {state_path} (v{self.pipeline_version_})")

    @classmethod
    def load(cls, data_dir: str = "./data", state_path: str = NODE_STATE_PATH) -> 'MahalanobisNode':
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        node = cls(
            model=state['model'],
            feature_names=state['feature_names'],
            data_dir=data_dir
        )

        node.original_reference_distances_ = state['original_reference_distances_']
        node.reference_distances_ = state['reference_distances_']
        node.reference_update_date_ = state['reference_update_date_']
        node.weeks_since_reference_update_ = state['weeks_since_reference_update_']

        node.recent_distances_buffer_ = deque(
            state.get('recent_distances_buffer_', []),
            maxlen=DRIFT_WINDOW_SIZE
        )
        node.persistent_drift_counter_ = state.get('persistent_drift_counter_', 0)

        node.calm_regime_buffer_ = deque(state['calm_regime_buffer_'], maxlen=104)
        node.weeks_of_calm_data_ = state['weeks_of_calm_data_']
        node.weeks_since_baseline_update_ = state.get('weeks_since_baseline_update_', 0)
        node.previous_regime_ = state.get('previous_regime_', 'calm')
        node.pipeline_version_ = state['pipeline_version_']
        node.last_prediction_date_ = state['last_prediction_date_']

        logger.info(f"Node loaded: {state_path} (v{node.pipeline_version_})")
        logger.info(f"  Last prediction: {node.last_prediction_date_}")

        return node

    def get_info(self) -> Dict:
        return {
            'node': {
                'version': self.pipeline_version_,
                'model_type': type(self.model).__name__,
                'n_features': len(self.feature_names),
                'last_prediction': str(self.last_prediction_date_.date()) if self.last_prediction_date_ else None,
                'weeks_since_reference_update': self.weeks_since_reference_update_,
                'weeks_since_baseline_update': self.weeks_since_baseline_update_,
                'weeks_of_calm_data': self.weeks_of_calm_data_,
                'previous_regime': self.previous_regime_,
                'reference_update_date': str(self.reference_update_date_.date()) if self.reference_update_date_ else None,
                'calm_buffer_size': len(self.calm_regime_buffer_),
                'recent_distances_buffer_size': len(self.recent_distances_buffer_),
                'reference_distances_size': len(self.reference_distances_) if self.reference_distances_ is not None else 0,
                'persistent_drift_counter': self.persistent_drift_counter_,
                'half_life_config': {
                    'normal_weeks': NORMAL_HALF_LIFE_WEEKS,
                    'post_stress_weeks': POST_STRESS_HALF_LIFE_WEEKS,
                    'post_crisis_weeks': POST_CRISIS_HALF_LIFE_WEEKS,
                }
            },
            'model': self.model.get_info()
        }


if __name__ == "__main__":
    from mahalanobis_model import MahalanobisStressModel, MAHALANOBIS_FEATURES

    node = MahalanobisNode(
        model=MahalanobisStressModel(),
        feature_names=MAHALANOBIS_FEATURES,
        data_dir="./data"
    )

    # 1. Train on 4 years of history
    node.fit_offline(start_date='2019-06-02', end_date='2023-05-28')
    node.save()

    # 2. Reload and run a walk-forward backtest using real CSV data
    node = MahalanobisNode.load(data_dir="./data")

    predictions = node.run_walk_forward(
        start_date='2023-06-04',
        end_date='2025-12-28'
    )

    print(predictions.head(10).to_string())
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Regime counts:\n{predictions['regime'].value_counts()}")