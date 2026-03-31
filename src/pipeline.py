import logging
import os
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_streamer import DataStreamer
from mahalanobis_model import MAHALANOBIS_FEATURES, MahalanobisStressModel
from mahalanobis_node import MahalanobisNode
from exposure_model import KALMAN_ALL_COLS, KalmanFactorModel
from exposure_node import KalmanNode

logger = logging.getLogger(__name__)

load_dotenv()

_MAHA_STATE_PATH = os.getenv("MAHA_STATE_PATH")
_KALMAN_STATE_PATH = os.getenv("KALMAN_STATE_PATH")

class CentralPipeline:

    def __init__(self, data_dir: str = './data') -> None:
        self.data_dir = data_dir

        self._weekly_streamer = DataStreamer(mode='weekly', data_dir=data_dir)
        self._daily_streamer  = DataStreamer(mode='daily',  data_dir=data_dir)

        # Node registry 
        self.nodes: Dict = {}

        # Stores the resolved CSV index timestamp for the last weekly run that actually processed data
        self._last_weekly_resolved_date_: Optional[pd.Timestamp] = None
        self._last_weekly_results_:       Optional[Dict]          = None

        os.makedirs('./models', exist_ok=True)
        os.makedirs('./logs',   exist_ok=True)

    def register_node(self, name: str, node) -> 'CentralPipeline':
        """Register an inference node under ``name``.  Returns self for chaining."""
        self.nodes[name] = node
        logger.info(f"Registered node: '{name}'  ({type(node).__name__})")
        return self

    def _fetch_weekly_features(
        self,
        target_date: str,
        feature_names,
    ) -> Tuple[pd.Timestamp, np.ndarray]:

        df = self._weekly_streamer.stream(
            feature_names=list(feature_names),
            start_date=target_date,
            end_date=None,   # single-week fetch
        )

        if df is None or len(df) == 0:
            raise ValueError(
                f"[pipeline] No weekly features available for target_date={target_date}."
            )

        resolved_date = df.index[0]

        logger.info(
            f"[pipeline] Weekly features fetched: {df.shape}  "
            f"resolved={resolved_date.date()}  "
            f"(requested={target_date})"
        )
        return resolved_date, df.values   # (pd.Timestamp, ndarray (1, n_features))

    def _fetch_daily_features(
        self,
        target_date: str,
        feature_names,
    ) -> Optional[np.ndarray]:

        next_day = (
            pd.Timestamp(target_date) + pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')

        df = self._daily_streamer.stream(
            feature_names=list(feature_names),
            start_date=target_date,
            end_date=next_day,
        )

        if df is None or len(df) == 0:
            logger.warning(
                f"[pipeline] No daily features for target_date={target_date}.  "
                "Skipping daily nodes."
            )
            return None

        logger.info(
            f"[pipeline] Daily features fetched: {df.shape}  "
            f"date={df.index[0].date()}"
        )
        return df.values   # shape: (1, n_all_cols)

    def run_weekly(self, target_date: str) -> Dict:
      
        logger.info("=" * 80)
        logger.info(f"Central Pipeline — run_weekly({target_date})")
        logger.info("=" * 80)

        resolved_date: Optional[pd.Timestamp] = None
        X_weekly:      Optional[np.ndarray]   = None

        has_weekly_nodes = any(
            not isinstance(node, KalmanNode) for node in self.nodes.values()
        )

        if has_weekly_nodes:
            resolved_date, X_weekly = self._fetch_weekly_features(
                target_date=target_date,
                feature_names=MAHALANOBIS_FEATURES,
            )

            if (
                self._last_weekly_resolved_date_ is not None
                and resolved_date == self._last_weekly_resolved_date_
                and self._last_weekly_results_ is not None
            ):
                logger.info(
                    f"[dedup] Weekly index {resolved_date.date()} already processed "
                    f"— returning cached result.  "
                    f"(target_date={target_date}, 0 node calls)"
                )
 
                cached = dict(self._last_weekly_results_)
                cached['date']             = target_date
                cached['weekly_cache_hit'] = True
                return cached

        logger.info(f"Registered nodes: {list(self.nodes.keys())}")

        master_results: Dict = {
            'date':             target_date,
            'weekly_cache_hit': False,
        }

        for name, node in self.nodes.items():
            # Skip daily-only nodes in the weekly run
            if isinstance(node, KalmanNode):
                logger.info(
                    f"Skipping daily node '{name}' in run_weekly "
                    "(use run_daily or run_all)"
                )
                continue

            logger.info(f"Routing to node: '{name}'")

            if X_weekly is None:
                logger.warning(f"Skipping node '{name}': X_weekly is None.")
                continue

            if name == 'mahalanobis':
                node_results = node.predict_and_update(target_date, X_weekly)
            else:
                logger.warning(
                    f"Node '{name}' has no explicit weekly routing rule; "
                    "passing X_weekly as default."
                )
                node_results = node.predict_and_update(target_date, X_weekly)

            for k, v in node_results.items():
                master_results[f"{name}_{k}"] = v

            save_path = f'./models/{name}_node_state.pkl'
            node.save(save_path)
            logger.info(f"Node '{name}' state saved → {save_path}")

        # Update deduplication cache 
        if resolved_date is not None:
            self._last_weekly_resolved_date_ = resolved_date
            self._last_weekly_results_       = dict(master_results)

        logger.info(
            f"Weekly routing complete.  "
            f"Resolved week: {resolved_date.date() if resolved_date else 'n/a'}  "
            f"Keys: {len(master_results)}"
        )
        return master_results

    def run_daily(self, target_date: str) -> Dict:
      
        logger.info("=" * 80)
        logger.info(f"Central Pipeline — run_daily({target_date})")
        logger.info(f"Registered nodes: {list(self.nodes.keys())}")
        logger.info("=" * 80)

        master_results: Dict = {'date': target_date}
        X_daily: Optional[np.ndarray] = None

        if 'kalman' in self.nodes:
            X_daily = self._fetch_daily_features(
                target_date=target_date,
                feature_names=KALMAN_ALL_COLS,
            )

        for name, node in self.nodes.items():
            # Skip weekly-only nodes in the daily run
            if isinstance(node, MahalanobisNode):
                logger.info(
                    f"Skipping weekly node '{name}' in run_daily "
                    "(use run_weekly or run_all)"
                )
                continue

            logger.info(f"Routing to node: '{name}'")

            if name == 'kalman':
                if X_daily is None:
                    logger.warning("Skipping kalman node: X_daily is None.")
                    continue
                node_results = node.predict_and_update(target_date, X_daily)
            else:
                logger.warning(
                    f"Node '{name}' has no explicit daily routing rule; "
                    "passing X_daily as default."
                )
                if X_daily is None:
                    logger.warning(f"Skipping node '{name}': X_daily is None.")
                    continue
                node_results = node.predict_and_update(target_date, X_daily)

            for k, v in node_results.items():
                master_results[f"{name}_{k}"] = v

            save_path = f'./models/{name}_node_state.pkl'
            node.save(save_path)
            logger.info(f"Node '{name}' state saved → {save_path}")

        logger.info(
            f"Daily routing complete.  Keys: {len(master_results)}"
        )
        return master_results

    def run_all(self, target_date: str) -> Dict:
    
        logger.info("=" * 80)
        logger.info(f"Central Pipeline — run_all({target_date})")
        logger.info("=" * 80)

        master_results: Dict = {'date': target_date}

        weekly_results = self.run_weekly(target_date)
        daily_results  = self.run_daily(target_date)

        # Merge — 'date' always reflects the actual trading day (target_date)
        for k, v in weekly_results.items():
            if k != 'date':
                master_results[k] = v

        for k, v in daily_results.items():
            if k != 'date':
                master_results[k] = v

        logger.info(
            f"run_all complete.  "
            f"weekly_cache_hit={weekly_results.get('weekly_cache_hit', False)}  "
            f"Total keys: {len(master_results)}"
        )
        return master_results

    def invalidate_weekly_cache(self) -> None:
        """
        Force the next run_weekly() call to re-process regardless of whether
        the resolved weekly index has changed.

        Use after: manual retrain of the Mahalanobis node, unit tests, or any
        external change to node state.
        """
        self._last_weekly_resolved_date_ = None
        self._last_weekly_results_       = None
        logger.info(
            "Weekly dedup cache invalidated — next run_weekly() will re-process."
        )

    def get_info(self) -> Dict:
        return {
            'data_dir':                  self.data_dir,
            'registered_nodes':          list(self.nodes.keys()),
            'last_weekly_resolved_date': (
                str(self._last_weekly_resolved_date_.date())
                if self._last_weekly_resolved_date_ else None
            ),
            'nodes': {
                name: node.get_info()
                for name, node in self.nodes.items()
            },
        }

def build_default_pipeline(data_dir: str = './data') -> CentralPipeline:
    """
    Each node is loaded from its saved state if one exists; otherwise a fresh
    (unfitted) node is created and will require fit_offline() before use.
    """
    pipeline = CentralPipeline(data_dir=data_dir)

    # Mahalanobis node (weekly)
    if os.path.exists(_MAHA_STATE_PATH):
        logger.info(f"Loading Mahalanobis node state: {_MAHA_STATE_PATH}")
        maha_node = MahalanobisNode.load(
            data_dir=data_dir,
            state_path=_MAHA_STATE_PATH,
        )
    else:
        logger.info("No saved Mahalanobis state — creating fresh node (needs fit_offline).")
        maha_node = MahalanobisNode(
            model=MahalanobisStressModel(),
            feature_names=MAHALANOBIS_FEATURES,
            data_dir=data_dir,
        )

    pipeline.register_node('mahalanobis', maha_node)

    # Kalman node (daily)
    if os.path.exists(_KALMAN_STATE_PATH):
        logger.info(f"Loading Kalman node state: {_KALMAN_STATE_PATH}")
        kalman_node = KalmanNode.load(
            data_dir=data_dir,
            state_path=_KALMAN_STATE_PATH,
        )
    else:
        logger.info("No saved Kalman state — creating fresh node (needs fit_offline).")
        kalman_node = KalmanNode(
            model=KalmanFactorModel(),
            data_dir=data_dir,
        )

    pipeline.register_node('kalman', kalman_node)

    return pipeline

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    pipeline = build_default_pipeline(data_dir='./data')

    # Weekly stress score (Mahalanobis)
    weekly_results = pipeline.run_weekly(target_date='2026-01-04')

    print("\n=== Weekly Results (Mahalanobis) ===")
    for k, v in sorted(weekly_results.items()):
        print(f"  {k}: {v}")

    # Daily factor exposure (Kalman)
    daily_results = pipeline.run_daily(target_date='2026-01-06')

    print("\n=== Daily Results (Kalman) ===")
    summary_keys = [
        k for k in sorted(daily_results.keys())
        if any(tag in k for tag in [
            'date', 'confidence', 'reliable', 'break', 'retrain',
            'top1_factor', 'top1_exposure',
        ])
    ]
    for k in summary_keys:
        print(f"  {k}: {daily_results[k]}")