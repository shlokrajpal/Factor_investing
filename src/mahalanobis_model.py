import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
import warnings
import logging
from typing import Tuple, Dict, Optional
import pickle

warnings.filterwarnings('ignore')

# Statistical parameters
MIN_SAMPLES_COVARIANCE = 50  # ~1 year of weekly data
CALM_REGIME_PERCENTILE = 0.50  # Bottom % of the composite proxy for baseline estimation

REGIME_ELEVATED_THRESHOLD = 0.75
REGIME_EXTREME_THRESHOLD = 0.95

MAHALANOBIS_FEATURES = [
    'spy_log_return',
    'vix_pct_change',
    'credit_spread',
    'tlt_log_return',
    'yield_curve_change',
    'tip_ief_spread',
    'uup_log_return',
    'gld_tlt_spread',
    'spy_vix_interaction',
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MahalanobisStressModel:
    def __init__(self):

        # Core model parameters
        self.baseline_mean_ = None
        self.baseline_std_ = None   # per-feature std of calm baseline
        self.baseline_cov_ = None
        self.precision_ = None
        self.feature_names_ = MAHALANOBIS_FEATURES

        # Metadata
        self.is_fitted_ = False

        logger.info(f"Initialized MahalanobisStressModel with {len(self.feature_names_)} features")

    def _estimate_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if X.shape[0] < MIN_SAMPLES_COVARIANCE:
            raise ValueError(
                f"Insufficient samples for covariance estimation: "
                f"{X.shape[0]} < {MIN_SAMPLES_COVARIANCE}"
            )

        # Fail fast on degenerate inputs that no estimator can fix
        if np.any(np.std(X, axis=0) == 0):
            zero_cols = np.where(np.std(X, axis=0) == 0)[0]
            raise ValueError(
                f"Zero-variance features detected (columns {zero_cols.tolist()}). "
                f"Fix the data pipeline — pseudo-inverse cannot recover from this."
            )

        cov_estimator = LedoitWolf(assume_centered=False)
        cov_estimator.fit(X)
        cov_matrix = cov_estimator.covariance_

        # Attempt standard inversion; fall back to pseudo-inverse for near-singular matrices
        try:
            precision = np.linalg.inv(cov_matrix)
            logger.debug("Precision matrix: standard inverse (full rank)")
        except np.linalg.LinAlgError:
            min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
            precision = np.linalg.pinv(cov_matrix, rcond=1e-10)
            logger.warning(
                f"Near-singular covariance (min eigenvalue {min_eig:.2e}); "
                f"pseudo-inverse used. Check feature correlations if this persists."
            )

        return cov_matrix, precision

    def _compute_mahalanobis_distances(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.baseline_mean_
        distances = np.sqrt(np.sum(X_centered @ self.precision_ * X_centered, axis=1))
        return distances

    def _distances_to_scores(
        self,
        distances: np.ndarray,
        effective_dof: Optional[int] = None
    ) -> np.ndarray:

        distances_squared = distances ** 2
        true_dof = len(self.feature_names_)
        scoring_dof = effective_dof if effective_dof is not None else true_dof

        if scoring_dof != true_dof:
            logger.debug(
                f"Scoring with effective_dof={scoring_dof} (structural DoF={true_dof}). "
                f"Mahalanobis geometry is unchanged; only CDF sensitivity is adjusted."
            )

        scores = chi2.cdf(distances_squared, df=scoring_dof)
        return scores

    def _compute_contributions(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.baseline_mean_
        contributions = np.abs(X_centered @ self.precision_)

        # Normalize to sum to 1 per sample
        row_sums = contributions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        contributions = contributions / row_sums

        return contributions

    def fit(
        self,
        X: np.ndarray,
        calm_regime_mask: Optional[np.ndarray] = None
    ) -> 'MahalanobisStressModel':

        logger.info(f"Training samples: {X.shape[0]}")

        if calm_regime_mask is None:
            # 1. Identify indices for the three Macro pillars
            vix_idx    = self.feature_names_.index('vix_pct_change')
            credit_idx = self.feature_names_.index('credit_spread')
            rates_idx  = self.feature_names_.index('yield_curve_change')

            # 2. Extract absolute magnitudes of shocks per pillar
            vix_mag    = np.abs(X[:, vix_idx])
            credit_mag = np.abs(X[:, credit_idx])
            rates_mag  = np.abs(X[:, rates_idx])

            # 3. Convert to percentile ranks (0–1) to normalise across different scales
            vix_rank    = stats.rankdata(vix_mag)    / len(vix_mag)
            credit_rank = stats.rankdata(credit_mag) / len(credit_mag)
            rates_rank  = stats.rankdata(rates_mag)  / len(rates_mag)

            # 4. Create the equal-weighted Macro-Composite Stress Proxy
            composite_proxy = (vix_rank + credit_rank + rates_rank) / 3.0

            # 5. Generate the calm mask: bottom CALM_REGIME_PERCENTILE of the composite
            calm_threshold_val = np.percentile(composite_proxy, CALM_REGIME_PERCENTILE * 100)
            calm_regime_mask   = composite_proxy <= calm_threshold_val

        X_calm = X[calm_regime_mask]
        logger.info(f"Calm regime: {len(X_calm)} samples ({len(X_calm)/len(X)*100:.0f}%)")

        if len(X_calm) < MIN_SAMPLES_COVARIANCE:
            raise ValueError(
                f"Insufficient calm samples: {len(X_calm)} < {MIN_SAMPLES_COVARIANCE}"
            )

        # Estimate baseline statistics
        self.baseline_mean_ = np.mean(X_calm, axis=0)
        self.baseline_std_ = np.std(X_calm, axis=0)
        self.baseline_cov_, self.precision_ = self._estimate_covariance(X_calm)

        self.is_fitted_ = True

        logger.info("Model fitted successfully")
        logger.info("=" * 80)

        return self

    def predict(
        self,
        X: np.ndarray,
        effective_dof: Optional[int] = None
    ) -> Dict[str, np.ndarray]:

        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # per-feature Z-scores and outlier count
        safe_std = np.where(self.baseline_std_ == 0, 1e-6, self.baseline_std_)
        z_scores = np.abs(X - self.baseline_mean_) / safe_std
        outlier_count = np.sum(z_scores >= 2.0, axis=1)   # features with |Z| >= 2

        distances = self._compute_mahalanobis_distances(X)
        stress_scores = self._distances_to_scores(distances, effective_dof=effective_dof)

        regimes = np.full(len(stress_scores), 'calm', dtype=object)
        regimes[stress_scores >= REGIME_ELEVATED_THRESHOLD] = 'elevated'
        regimes[stress_scores >= REGIME_EXTREME_THRESHOLD] = 'extreme'

        contributions = self._compute_contributions(X)

        return {
            'stress_score': stress_scores,
            'mahalanobis_distance': distances,
            'contributions': contributions,
            'regime': regimes,
            'outlier_count': outlier_count,
        }

    def update_baseline(
        self,
        X_calm: np.ndarray,
        alpha: float
    ) -> 'MahalanobisStressModel':
        """
        Exponentially smooth the baseline mean, std, and covariance.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        new_mean = np.mean(X_calm, axis=0)
        new_std = np.std(X_calm, axis=0)

        # Always smooth the mean and std
        self.baseline_mean_ = alpha * self.baseline_mean_ + (1 - alpha) * new_mean
        self.baseline_std_ = alpha * self.baseline_std_ + (1 - alpha) * new_std

        if len(X_calm) < MIN_SAMPLES_COVARIANCE:
            logger.warning(
                f"Insufficient calm samples for covariance update: "
                f"{len(X_calm)} < {MIN_SAMPLES_COVARIANCE}. Using mean/std-only update."
            )
            return self

        new_cov, _ = self._estimate_covariance(X_calm)

        smoothed_cov = alpha * self.baseline_cov_ + (1 - alpha) * new_cov

        ENTROPY_FLOOR_WEIGHT = 0.05
        n_features = smoothed_cov.shape[0]
        floor_scale = np.trace(new_cov) / n_features
        identity_floor = floor_scale * np.eye(n_features)
        self.baseline_cov_ = (
            (1 - ENTROPY_FLOOR_WEIGHT) * smoothed_cov
            + ENTROPY_FLOOR_WEIGHT * identity_floor
        )

        logger.info(
            f"  Covariance updated: trace_smoothed={np.trace(smoothed_cov):.2f}  "
            f"trace_after_floor={np.trace(self.baseline_cov_):.2f}  "
            f"floor_scale={floor_scale:.4f}"
        )

        try:
            self.precision_ = np.linalg.inv(self.baseline_cov_)
        except np.linalg.LinAlgError:
            min_eig = np.min(np.linalg.eigvalsh(self.baseline_cov_))
            self.precision_ = np.linalg.pinv(self.baseline_cov_, rcond=1e-10)
            logger.warning(
                f"Smoothed covariance near-singular (min eig {min_eig:.2e}); "
                f"pseudo-inverse used for precision matrix."
            )

        logger.info(f"Baseline updated: alpha={alpha:.4f}  n_samples={len(X_calm)}")

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._compute_mahalanobis_distances(X)

    def get_info(self) -> Dict:
        if not self.is_fitted_:
            return {'fitted': False}

        return {
            'fitted': True,
            'n_features': len(self.feature_names_),
            'features': self.feature_names_,
            'scoring_method': 'chi_squared_cdf',
            'degrees_of_freedom': len(self.feature_names_),
            'baseline_mean_range': {
                'min': float(self.baseline_mean_.min()),
                'max': float(self.baseline_mean_.max()),
                'mean': float(self.baseline_mean_.mean())
            },
            'baseline_std_range': {
                'min': float(self.baseline_std_.min()),
                'max': float(self.baseline_std_.max()),
                'mean': float(self.baseline_std_.mean())
            }
        }