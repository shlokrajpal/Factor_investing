import warnings
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

KALMAN_ASSET_COLS: List[str] = [
    'gld_log_return',
    'slv_log_return',
    'mtum_log_return',
    'size_log_return',
    'qual_log_return',
    'vlue_log_return',
    'usmv_log_return',
    'shy_log_return',
    'tlt_log_return',
]

KALMAN_FACTOR_COLS: List[str] = [
    'spy_log_return',
    'spy_return_minus_ma5',
    'vix_pct_change',
    'credit_spread',
    'tnx_change',
    'yield_curve_change',
    'tip_ief_spread',
    'uup_log_return',
    'tlt_orth_return',
]

KALMAN_ALL_COLS: List[str] = KALMAN_ASSET_COLS + KALMAN_FACTOR_COLS

N_ASSETS:        int   = len(KALMAN_ASSET_COLS)   # 9
N_FACTORS:       int   = len(KALMAN_FACTOR_COLS)  # 9
N_STATE:         int   = N_FACTORS + 1            # 10  (intercept + 9 betas)

TOP_K_EXPOSURES: int   = 4     # top-k factor exposures stored per asset
MIN_BURN_IN:     int   = 63    # ~3 months; reliability flag is False before this

DEFAULT_Q_SCALE: float = 1e-4  # process noise  — controls beta drift speed
DEFAULT_R_SCALE: float = 1e-3  # observation noise — per-asset residual variance
P_INIT_SCALE:    float = 1.0   # high P allows rapid burn-in

# Optimisation grid bounds
Q_GRID = np.logspace(-6, -2, 9)   # 9 candidate Q scales
R_GRID = np.logspace(-5, -1, 9)   # 9 candidate R scales

class KalmanFactorModel:

    def __init__(
        self,
        asset_names:  Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        q_scale:      float = DEFAULT_Q_SCALE,
        r_scale:      float = DEFAULT_R_SCALE,
        p_init_scale: float = P_INIT_SCALE,
    ) -> None:

        self.asset_names  = asset_names  or KALMAN_ASSET_COLS
        self.factor_names = factor_names or KALMAN_FACTOR_COLS
        self.n_assets     = len(self.asset_names)
        self.n_factors    = len(self.factor_names)
        self.n_state      = self.n_factors + 1    # +1 for time-varying intercept

        self.B_: np.ndarray = np.zeros((self.n_assets, self.n_state))
        self.P_: np.ndarray = (
            p_init_scale
            * np.tile(np.eye(self.n_state), (self.n_assets, 1, 1))
        )  # shape: (n_assets, n_state, n_state)

        #   Q_: shared (n_state × n_state) — broadcast over the asset dimension
        #   R_: per-asset (n_assets,)       — individual return residual variance
        self.Q_: np.ndarray = q_scale * np.eye(self.n_state)
        self.R_: np.ndarray = np.full(self.n_assets, r_scale)

        self.n_updates_:   int  = 0
        self.is_fitted_:   bool = False
        self.is_reliable_: bool = False   # True after MIN_BURN_IN steps

        logger.info(
            f"KalmanFactorModel  {self.n_assets} assets × {self.n_factors} factors  "
            f"n_state={self.n_state}  "
            f"B shape={self.B_.shape}  P shape={self.P_.shape}  "
            f"Q_scale={q_scale:.0e}  R_scale={r_scale:.0e}"
        )

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:

        B_pred = self.B_.copy()
        P_pred = self.P_ + self.Q_[np.newaxis, :, :]   # broadcast Q
        return B_pred, P_pred

    def update(
        self,
        y:      np.ndarray,   # (n_assets,)   actual daily log-returns
        H:      np.ndarray,   # (n_state,)    [1, f1, f2, …, fK]
        B_pred: np.ndarray,   # (n_assets, n_state)
        P_pred: np.ndarray,   # (n_assets, n_state, n_state)
    ) -> Dict:

        y_hat = B_pred @ H                     # (n_assets,)
        nu    = y - y_hat                      # Innovation Vector (n_assets,)

        #   P_H[i] = P_pred[i] @ H
        P_H = np.einsum('ijk,k->ij', P_pred, H)   # (n_assets, n_state)
        #   S[i]   = H @ P_pred[i] @ H + R[i]
        S   = np.einsum('ij,j->i', P_H, H) + self.R_   # (n_assets,)

        #   K[i] = P_pred[i] @ H / S[i]
        K = P_H / S[:, np.newaxis]             # (n_assets, n_state)

        #   B_t[i] = B_pred[i] + K[i] * ν[i]
        self.B_ = B_pred + K * nu[:, np.newaxis]   # (n_assets, n_state)

        # Joseph-Stabilised Covariance Update
        #   I_KH[i]  = I  −  outer(K[i], H)
        #   P_t[i]   = I_KH[i] P_pred[i] I_KH[i]^T + R[i] outer(K[i],K[i])
        I_KH = (
            np.eye(self.n_state)[np.newaxis]        # (1, n_state, n_state)
            - np.einsum('ij,k->ijk', K, H)          # (n_assets, n_state, n_state)
        )
        tmp          = np.einsum('ijk,ikl->ijl', I_KH, P_pred)    # I_KH @ P_pred
        P_joseph     = np.einsum('ijm,ikm->ijk', tmp, I_KH)       # @ I_KH^T
        P_noise      = (
            self.R_[:, np.newaxis, np.newaxis]
            * np.einsum('ij,ik->ijk', K, K)
        )
        self.P_ = P_joseph + P_noise

        # Enforce symmetry (guards against floating-point drift accumulation)
        self.P_ = 0.5 * (self.P_ + self.P_.transpose(0, 2, 1))

        self.n_updates_   += 1
        self.is_fitted_    = True
        self.is_reliable_  = self.n_updates_ >= MIN_BURN_IN

        traces       = np.einsum('ijj->i', self.P_)              # trace per asset
        confidence   = 1.0 / np.maximum(traces, 1e-12)           # (n_assets,)
        std_innov    = nu / np.sqrt(np.maximum(S, 1e-12))        # (n_assets,)

        return {
            'nu':         nu,           # Innovation Vector       (n_assets,)
            'S':          S,            # Innovation variance     (n_assets,)
            'std_innov':  std_innov,    # Standardised innovation (n_assets,)
            'K_norm':     np.linalg.norm(K, axis=1),   # Gain norms (n_assets,)
            'confidence': confidence,   # 1/trace(P[i])           (n_assets,)
            'P_traces':   traces,       # trace(P[i])             (n_assets,)
        }

    def step(
        self,
        y: np.ndarray,     # (n_assets,)  today's asset returns
        factors: np.ndarray,  # (n_factors,) today's macro factor values
    ) -> Dict:
        """
        Combined predict + update for one trading day
        """
        H              = np.concatenate([[1.0], factors])    # prepend intercept
        B_pred, P_pred = self.predict()
        result         = self.update(y, H, B_pred, P_pred)

        result['B']            = self.B_.copy()     # (n_assets, n_state)
        result['P']            = self.P_.copy()     # (n_assets, n_state, n_state)
        result['n_updates']    = self.n_updates_
        result['is_reliable']  = self.is_reliable_

        return result

    def get_exposure_matrix(self) -> np.ndarray:
        """
        Return the pure factor beta sub-matrix (excludes the intercept column)
        """
        return self.B_[:, 1:].copy()    # strip intercept column

    def get_top_exposures(
        self,
        top_k: int = TOP_K_EXPOSURES,
    ) -> Dict[str, List[Dict]]:

        E = self.get_exposure_matrix()    # (n_assets, n_factors)
        result = {}

        for i, asset in enumerate(self.asset_names):
            abs_betas  = np.abs(E[i])
            top_idx    = np.argsort(abs_betas)[::-1][:top_k]
            result[asset] = [
                {
                    'rank':         rank + 1,
                    'factor':       self.factor_names[j],
                    'exposure':     float(E[i, j]),
                    'abs_exposure': float(abs_betas[j]),
                }
                for rank, j in enumerate(top_idx)
            ]

        return result

    def get_confidence_vector(self) -> np.ndarray:
        """
        Confidence scalar for each asset = 1 / trace(P_t[i]).
        Shape: (n_assets,)
        """
        traces = np.einsum('ijj->i', self.P_)
        return 1.0 / np.maximum(traces, 1e-12)

    def optimize_hyperparams(
        self,
        returns_matrix: np.ndarray,     # (T, n_assets)
        factors_matrix: np.ndarray,     # (T, n_factors)
        q_grid: Optional[np.ndarray] = None,
        r_grid: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:

        if q_grid is None:
            q_grid = Q_GRID
        if r_grid is None:
            r_grid = R_GRID

        best_ll         = -np.inf
        best_q, best_r  = DEFAULT_Q_SCALE, DEFAULT_R_SCALE

        for q in q_grid:
            for r in r_grid:
                ll = self._trial_log_likelihood(returns_matrix, factors_matrix, q, r)
                if ll > best_ll:
                    best_ll        = ll
                    best_q, best_r = q, r

        logger.info(
            f"Hyperparameter optimisation: "
            f"Q={best_q:.2e}  R={best_r:.2e}  "
            f"avg_log_lik={best_ll:.4f}"
        )
        return best_q, best_r

    def _trial_log_likelihood(
        self,
        returns_matrix: np.ndarray,   # (T, n_assets)
        factors_matrix: np.ndarray,   # (T, n_factors)
        q_scale: float,
        r_scale: float,
    ) -> float:
       
        T        = returns_matrix.shape[0]
        B_trial  = np.zeros((self.n_assets, self.n_state))
        P_trial  = P_INIT_SCALE * np.tile(np.eye(self.n_state), (self.n_assets, 1, 1))
        Q_trial  = q_scale * np.eye(self.n_state)
        R_trial  = np.full(self.n_assets, r_scale)

        total_ll = 0.0
        n_valid  = 0

        for t in range(T):
            y_row = returns_matrix[t]
            f_row = factors_matrix[t]

            # Skip rows with any NaN
            if np.any(np.isnan(y_row)) or np.any(np.isnan(f_row)):
                continue

            H = np.concatenate([[1.0], f_row])

            # Predict
            P_pred = P_trial + Q_trial[np.newaxis, :, :]

            # Innovation
            y_hat = B_trial @ H                            # (n_assets,)
            nu    = y_row - y_hat                          # (n_assets,)
            P_H   = np.einsum('ijk,k->ij', P_pred, H)     # (n_assets, n_state)
            S     = np.einsum('ij,j->i', P_H, H) + R_trial  # (n_assets,)

            # Gaussian log-likelihood contribution (sum over assets)
            total_ll += float(
                np.sum(-0.5 * (np.log(np.maximum(2 * np.pi * S, 1e-30))
                               + nu ** 2 / np.maximum(S, 1e-12)))
            )

            # Update (simplified for speed — skip Joseph form in trial)
            K      = P_H / S[:, np.newaxis]
            I_KH   = np.eye(self.n_state) - np.einsum('ij,k->ijk', K, H)
            tmp    = np.einsum('ijk,ikl->ijl', I_KH, P_pred)
            P_trial = np.einsum('ijm,ikm->ijk', tmp, I_KH)
            P_trial = 0.5 * (P_trial + P_trial.transpose(0, 2, 1))
            B_trial = B_trial + K * nu[:, np.newaxis]

            n_valid += 1

        return total_ll / max(n_valid, 1)

    def apply_hyperparams(self, q_scale: float, r_scale: float) -> None:
     
        self.Q_ = q_scale * np.eye(self.n_state)
        self.R_ = np.full(self.n_assets, r_scale)
        logger.info(
            f"Hyperparams applied:  Q_scale={q_scale:.2e}  R_scale={r_scale:.2e}"
        )

    def get_state_dict(self) -> Dict:
      
        return {
            # State matrix and uncertainty tensor
            'B_':           self.B_.copy(),
            'P_':           self.P_.copy(),
            # Noise matrices
            'Q_':           self.Q_.copy(),
            'R_':           self.R_.copy(),
            # Config
            'asset_names':  self.asset_names,
            'factor_names': self.factor_names,
            # Counters
            'n_updates_':   self.n_updates_,
            'is_fitted_':   self.is_fitted_,
            'is_reliable_': self.is_reliable_,
        }

    def load_state_dict(self, state: Dict) -> None:
      
        self.B_           = state['B_'].copy()
        self.P_           = state['P_'].copy()
        self.Q_           = state['Q_'].copy()
        self.R_           = state['R_'].copy()
        self.n_updates_   = state['n_updates_']
        self.is_fitted_   = state['is_fitted_']
        self.is_reliable_ = state['is_reliable_']

    def get_info(self) -> Dict:
  
        E          = self.get_exposure_matrix()     # (n_assets, n_factors)
        traces     = np.einsum('ijj->i', self.P_)
        confidence = 1.0 / np.maximum(traces, 1e-12)

        return {
            'fitted':       self.is_fitted_,
            'reliable':     self.is_reliable_,
            'n_updates':    self.n_updates_,
            'n_assets':     self.n_assets,
            'n_factors':    self.n_factors,
            'n_state':      self.n_state,
            'B_shape':      list(self.B_.shape),
            'P_shape':      list(self.P_.shape),
            'Q_scale':      float(self.Q_[0, 0]),
            'R_scale':      float(self.R_.mean()),
            'assets':       self.asset_names,
            'factors':      self.factor_names,
            'per_asset': {
                asset: {
                    'confidence':    float(confidence[i]),
                    'P_trace':       float(traces[i]),
                    'intercept':     float(self.B_[i, 0]),
                    'top_exposures': self.get_top_exposures()[asset],
                }
                for i, asset in enumerate(self.asset_names)
            },
        }