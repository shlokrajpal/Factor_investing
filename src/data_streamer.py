import os
import warnings
from typing import Optional, List
import numpy as np
import pandas as pd
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def _log_return(series: pd.Series) -> pd.Series:
    """Log return: ln(P_t / P_{t-1})."""
    return np.log(series / series.shift(1))


def _orthogonalize(y: pd.Series, x: pd.Series) -> pd.Series:
    """
    OLS residuals of regressing y on x (with intercept).
    Returns a Series aligned to y's index, preserving NaN where inputs are NaN.
    """
    mask = y.notna() & x.notna()
    if mask.sum() < 2:
        return y.copy()

    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta, *_ = np.linalg.lstsq(X, y[mask].values, rcond=None)

    fitted = pd.Series(np.nan, index=y.index)
    fitted[mask] = beta[0] + beta[1] * x[mask]
    return y - fitted


class FeatureEngineering:
    """
    Computes and persists two feature sets from raw_daily_data.csv:

        daily_features.csv   - daily-frequency features, all orthogonalised
                               against SPY log-return.
        weekly_features.csv  - weekly-frequency features, all orthogonalised
                               against weekly SPY log-return.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        raw_data_filename: str = "raw_daily_data.csv",
    ):
        self.data_dir = data_dir
        self.raw_data_filename = raw_data_filename
        self.raw_data_path = os.path.join(data_dir, raw_data_filename)
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineering")

    def _load_raw_data(self) -> pd.DataFrame:

        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(
                f"Raw data not found: {self.raw_data_path}\n"
                "Run data_ingestion.py first."
            )
        self.logger.info(f"Loading raw data from: {self.raw_data_path}")
        daily = pd.read_csv(self.raw_data_path, index_col=0, parse_dates=True)
        daily.sort_index(inplace=True)
        if len(daily) == 0:
            raise ValueError("Raw data file is empty.")
        self.logger.info(
            f"Loaded {len(daily)} days "
            f"({daily.index[0].date()} to {daily.index[-1].date()})"
        )
        return daily

    def _convert_to_weekly(self, data: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Converting daily data to weekly (W-SUN)...")
        agg_dict = {}
        for col in data.columns:
            if col.endswith('_volume'):
                agg_dict[col] = 'sum'
            elif col.endswith('_open'):
                agg_dict[col] = 'first'
            elif col.endswith('_high'):
                agg_dict[col] = 'max'
            elif col.endswith('_low'):
                agg_dict[col] = 'min'
            else:
                agg_dict[col] = 'last'

        weekly = data.resample('W-SUN').agg(agg_dict)
        self.logger.info(f"Converted to {len(weekly)} weekly periods.")
        return weekly

    def _orthogonalize_all(
        self,
        features: pd.DataFrame,
        spy_col: str = 'spy_log_return',
    ) -> pd.DataFrame:

        if spy_col not in features.columns:
            self.logger.warning(
                f"'{spy_col}' not in features; skipping orthogonalisation."
            )
            return features.copy()

        result = features.copy()
        spy = features[spy_col]
        for col in features.columns:
            if col == spy_col:
                continue
            result[col] = _orthogonalize(features[col], spy)
        return result

    def _compute_daily_features(self, daily: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Computing daily features...")
        f = pd.DataFrame(index=daily.index)

        f['gld_log_return'] = _log_return(daily['gld_close'])
        f['slv_log_return'] = _log_return(daily['slv_close'])
        f['mtum_log_return'] = _log_return(daily['mtum_close'])
        f['size_log_return'] = _log_return(daily['size_close'])
        f['qual_log_return'] = _log_return(daily['qual_close'])
        f['vlue_log_return'] = _log_return(daily['vlue_close'])
        f['usmv_log_return'] = _log_return(daily['usmv_close'])
        f['shy_log_return'] = _log_return(daily['shy_close'])
        f['tlt_log_return'] = _log_return(daily['tlt_close'])

        if 'spy_close' in daily.columns:
            spy_lr = _log_return(daily['spy_close'])
            f['spy_log_return'] = spy_lr
        else:
            self.logger.warning("spy_close missing; spy_log_return = NaN")
            f['spy_log_return'] = np.nan
            spy_lr = pd.Series(np.nan, index=daily.index)

        f['spy_return_minus_ma5'] = spy_lr - spy_lr.rolling(5).mean()

        if 'vix_close' in daily.columns:
            f['vix_pct_change'] = daily['vix_close'].pct_change()
        else:
            self.logger.warning("vix_close missing; vix_pct_change = NaN")
            f['vix_pct_change'] = np.nan

        hyg_lr = _log_return(daily['hyg_close']) if 'hyg_close' in daily.columns else None
        ief_lr = _log_return(daily['ief_close']) if 'ief_close' in daily.columns else None
        if hyg_lr is not None and ief_lr is not None:
            f['credit_spread'] = hyg_lr - ief_lr
        else:
            self.logger.warning("hyg_close or ief_close missing; credit_spread = NaN")
            f['credit_spread'] = np.nan

        if 'tnx_close' in daily.columns:
            tnx_chg = daily['tnx_close'].diff()
            f['tnx_change'] = tnx_chg
        else:
            self.logger.warning("tnx_close missing; tnx_change = NaN")
            tnx_chg = pd.Series(np.nan, index=daily.index)
            f['tnx_change'] = np.nan

        if 'tnx_close' in daily.columns and 'irx_close' in daily.columns:
            spread = daily['tnx_close'] - daily['irx_close']
            f['yield_curve_change'] = spread.diff()
        else:
            self.logger.warning(
                "tnx_close or irx_close missing; yield_curve_change = NaN"
            )
            f['yield_curve_change'] = np.nan

        tip_lr = _log_return(daily['tip_close']) if 'tip_close' in daily.columns else None
        if tip_lr is not None and ief_lr is not None:
            f['tip_ief_spread'] = tip_lr - ief_lr
        else:
            self.logger.warning(
                "tip_close or ief_close missing; tip_ief_spread = NaN"
            )
            f['tip_ief_spread'] = np.nan

        if 'uup_close' in daily.columns:
            f['uup_log_return'] = _log_return(daily['uup_close'])
        else:
            self.logger.warning("uup_close missing; uup_log_return = NaN")
            f['uup_log_return'] = np.nan

        if 'gld_close' in daily.columns:
            f['gld_log_return'] = _log_return(daily['gld_close'])
        else:
            self.logger.warning("gld_close missing; gld_log_return = NaN")
            f['gld_log_return'] = np.nan

        if 'tlt_close' in daily.columns:
            tlt_lr = _log_return(daily['tlt_close'])
            f['tlt_orth_return'] = _orthogonalize(tlt_lr, tnx_chg)
        else:
            self.logger.warning("tlt_close missing; tlt_orth_return = NaN")
            f['tlt_orth_return'] = np.nan

        f = f.iloc[1:]  # drop first row (NaN from shift-based returns)
        f = self._orthogonalize_all(f, spy_col='spy_log_return')

        self.logger.info(
            f"Daily features: {len(f.columns)} columns, {len(f)} rows "
            f"({f.index[0].date()} to {f.index[-1].date()})"
        )
        return f

    def _compute_weekly_features(self, weekly: pd.DataFrame) -> pd.DataFrame:

        self.logger.info("Computing weekly features...")
        f = pd.DataFrame(index=weekly.index)

        if 'spy_close' in weekly.columns:
            spy_lr = _log_return(weekly['spy_close'])
            f['spy_log_return'] = spy_lr
        else:
            self.logger.warning("spy_close missing; spy_log_return = NaN")
            f['spy_log_return'] = np.nan
            spy_lr = pd.Series(np.nan, index=weekly.index)

        if 'vix_close' in weekly.columns:
            vix_pct = weekly['vix_close'].pct_change()
            f['vix_pct_change'] = vix_pct
        else:
            self.logger.warning("vix_close missing; vix_pct_change = NaN")
            vix_pct = pd.Series(np.nan, index=weekly.index)
            f['vix_pct_change'] = np.nan

        hyg_lr = _log_return(weekly['hyg_close']) if 'hyg_close' in weekly.columns else None
        ief_lr = _log_return(weekly['ief_close']) if 'ief_close' in weekly.columns else None
        if hyg_lr is not None and ief_lr is not None:
            f['credit_spread'] = hyg_lr - ief_lr
        else:
            self.logger.warning("hyg_close or ief_close missing; credit_spread = NaN")
            f['credit_spread'] = np.nan

        if 'tlt_close' in weekly.columns:
            tlt_lr = _log_return(weekly['tlt_close'])
            f['tlt_log_return'] = tlt_lr
        else:
            self.logger.warning("tlt_close missing; tlt_log_return = NaN")
            tlt_lr = pd.Series(np.nan, index=weekly.index)
            f['tlt_log_return'] = np.nan

        if 'tnx_close' in weekly.columns and 'irx_close' in weekly.columns:
            spread = weekly['tnx_close'] - weekly['irx_close']
            f['yield_curve_change'] = spread.diff()
        else:
            self.logger.warning(
                "tnx_close or irx_close missing; yield_curve_change = NaN"
            )
            f['yield_curve_change'] = np.nan

        tip_lr = _log_return(weekly['tip_close']) if 'tip_close' in weekly.columns else None
        if tip_lr is not None and ief_lr is not None:
            f['tip_ief_spread'] = tip_lr - ief_lr
        else:
            self.logger.warning(
                "tip_close or ief_close missing; tip_ief_spread = NaN"
            )
            f['tip_ief_spread'] = np.nan

        if 'uup_close' in weekly.columns:
            f['uup_log_return'] = _log_return(weekly['uup_close'])
        else:
            self.logger.warning("uup_close missing; uup_log_return = NaN")
            f['uup_log_return'] = np.nan

        if 'gld_close' in weekly.columns:
            gld_lr = _log_return(weekly['gld_close'])
            f['gld_tlt_spread'] = gld_lr - tlt_lr
        else:
            self.logger.warning("gld_close missing; gld_tlt_spread = NaN")
            f['gld_tlt_spread'] = np.nan

        f['spy_vix_interaction'] = spy_lr * vix_pct

        f = f.iloc[1:]  # drop first row (NaN from shift-based returns)
        f = self._orthogonalize_all(f, spy_col='spy_log_return')

        self.logger.info(
            f"Weekly features: {len(f.columns)} columns, {len(f)} rows "
            f"({f.index[0].date()} to {f.index[-1].date()})"
        )
        return f

    def process_and_save(
        self,
        daily_filename:  str = "daily_features.csv",
        weekly_filename: str = "weekly_features.csv",
    ) -> tuple:

        daily_raw = self._load_raw_data()

        self.logger.info("--- Daily feature computation ---")
        daily_features = self._compute_daily_features(daily_raw)
        daily_path = os.path.join(self.data_dir, daily_filename)
        daily_features.to_csv(daily_path)
        self.logger.info(f"Saved daily features  -> {daily_path}  {daily_features.shape}")

        self.logger.info("--- Weekly feature computation ---")
        weekly_raw      = self._convert_to_weekly(daily_raw)
        weekly_features = self._compute_weekly_features(weekly_raw)
        weekly_path = os.path.join(self.data_dir, weekly_filename)
        weekly_features.to_csv(weekly_path)
        self.logger.info(f"Saved weekly features -> {weekly_path}  {weekly_features.shape}")

        self.logger.info("=" * 70)
        self.logger.info("Feature engineering complete.")
        self.logger.info("=" * 70)

        return daily_features, weekly_features


class DataStreamer:

    def __init__(
        self,
        mode: str = 'weekly',
        data_dir: str = "./data",
        daily_features_filename:  str = "daily_features.csv",
        weekly_features_filename: str = "weekly_features.csv",
    ):
        if mode not in ('daily', 'weekly'):
            raise ValueError(f"mode must be 'daily' or 'weekly', got '{mode}'.")
        self.mode = mode
        self.data_dir = data_dir
        self._filenames: dict = {
            'daily':  os.path.join(data_dir, daily_features_filename),
            'weekly': os.path.join(data_dir, weekly_features_filename),
        }
        self.logger = logging.getLogger(f"{__name__}.DataStreamer")

    def _load_features(self) -> pd.DataFrame:
        path = self._filenames[self.mode]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{self.mode.capitalize()} features file not found: {path}\n"
                "Run FeatureEngineering.process_and_save() first."
            )

        self.logger.info(f"Loading {self.mode} features from: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)

        if len(df) == 0:
            raise ValueError(f"{self.mode.capitalize()} features file is empty.")

        self.logger.info(
            f"Loaded {len(df)} {self.mode} rows "
            f"({df.index[0].date()} to {df.index[-1].date()}), "
            f"{len(df.columns)} features."
        )
        return df

    def _resolve_date(
        self,
        target_date: pd.Timestamp,
        df: pd.DataFrame,
    ) -> Optional[pd.Timestamp]:
        """
        Return the nearest index entry at or before ``target_date``
        """
        valid = df.index[df.index <= target_date]
        if len(valid) == 0:
            return None
        resolved = valid[-1]
        if resolved != target_date:
            self.logger.warning(
                f"[{self.mode}] Requested date '{target_date.date()}' not in index; "
                f"resolved to '{resolved.date()}'."
            )
        return resolved

    def _is_week_complete(self, week_end: pd.Timestamp) -> bool:
        """True if the calendar Sunday for this week has already passed."""
        return pd.Timestamp.now().normalize() > week_end

    def stream(
        self,
        feature_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:

        df = self._load_features()

        missing = set(feature_names) - set(df.columns)
        if missing:
            raise ValueError(
                f"Feature(s) not found in '{self.mode}' features: {sorted(missing)}\n"
                f"Available: {sorted(df.columns.tolist())}"
            )

        # Single-period request
        if end_date is None:
            if start_date is None:
                raise ValueError("start_date is required when end_date is not provided.")

            resolved = self._resolve_date(pd.Timestamp(start_date), df)

            if resolved is None:
                self.logger.warning(
                    f"[{self.mode}] '{start_date}' precedes all available data."
                )
                return None

            if self.mode == 'weekly' and not self._is_week_complete(resolved):
                self.logger.warning(
                    f"[weekly] Week ending {resolved.date()} is not yet complete."
                )
                return None

            return df.loc[[resolved], feature_names]

        # Date-range request [start_date, end_date)
        start_ts = pd.Timestamp(start_date) if start_date else df.index[0]
        end_ts   = pd.Timestamp(end_date)

        mask   = (df.index >= start_ts) & (df.index < end_ts)
        result = df.loc[mask, feature_names]

        if len(result) == 0:
            self.logger.warning(
                f"[{self.mode}] No data found in [{start_date}, {end_date})."
            )
            return None

        self.logger.info(
            f"[{self.mode}] Returning {len(result)} rows "
            f"({result.index[0].date()} to {result.index[-1].date()}) "
            f"for {len(feature_names)} feature(s)."
        )
        return result

    def get_latest_available_date(self) -> pd.Timestamp:
        """Return the most recent index entry in the feature set."""
        return self._load_features().index[-1]

    def get_data_info(self) -> dict:
        try:
            df   = self._load_features()
            info = {
                'available':  True,
                'mode':       self.mode,
                'file_path':  self._filenames[self.mode],
                'n_rows':     len(df),
                'start_date': str(df.index[0].date()),
                'end_date':   str(df.index[-1].date()),
                'features':   sorted(df.columns.tolist()),
            }
            if self.mode == 'weekly':
                info['latest_week_complete'] = self._is_week_complete(df.index[-1])
            return info
        except Exception as exc:
            return {
                'available': False,
                'mode':      self.mode,
                'file_path': self._filenames[self.mode],
                'error':     str(exc),
            }


if __name__ == "__main__":
    # Compute and persist both feature sets
    fe = FeatureEngineering(data_dir="./data")
    daily_df, weekly_df = fe.process_and_save()

    print(f"\nDaily  features : {daily_df.shape}")
    print(f"  columns: {list(daily_df.columns)}")
    print(f"\nWeekly features : {weekly_df.shape}")
    print(f"  columns: {list(weekly_df.columns)}")

    """
    daily_streamer  = DataStreamer(mode='daily',  data_dir="./data")
    weekly_streamer = DataStreamer(mode='weekly', data_dir="./data")

    daily_slice = daily_streamer.stream(
        feature_names=[
            'spy_log_return', 'spy_return_minus_ma5', 'vix_pct_change',
            'credit_spread', 'tnx_change', 'yield_curve_change',
            'tip_ief_spread', 'uup_log_return', 'gld_log_return',
            'tlt_orth_return',
        ],
        start_date='2023-01-01',
        end_date='2024-01-01',
    )

    weekly_slice = weekly_streamer.stream(
        feature_names=[
            'spy_log_return', 'vix_pct_change', 'credit_spread',
            'tlt_log_return', 'yield_curve_change', 'tip_ief_spread',
            'uup_log_return', 'gld_tlt_spread', 'spy_vix_interaction',
        ],
        start_date='2023-01-01',
        end_date='2024-01-01',
    )
    """