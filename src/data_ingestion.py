import os
import warnings
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
import logging
import time
import numpy as np

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DataIngestion:

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _fetch_ticker_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        columns: list,
        prefix: str
    ) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching {ticker}...")
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if df.empty:
                    self.logger.warning(f"{ticker} returned no data")
                    return pd.DataFrame()

                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Select requested columns
                available_cols = [col for col in columns if col in df.columns]
                if not available_cols:
                    self.logger.warning(f"{ticker} missing requested columns")
                    return pd.DataFrame()

                df_selected = df[available_cols].copy()
                df_selected.columns = [f"{prefix}_{c.lower()}" for c in available_cols]

                self.logger.info(f"{ticker}: {len(df)} rows")
                return df_selected

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}/{max_retries} for {ticker} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"Failed to fetch {ticker} after {max_retries} attempts")
                    return pd.DataFrame()

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:

        if end_date is None:
            today = datetime.now()
            days_since_sunday = (today.weekday() + 1) % 7
            last_sunday = today - timedelta(days=days_since_sunday)
            end_date = last_sunday.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching data: {start_date} to {end_date}")

        tickers_config = [
            # Core equity / volatility
            ('SPY',   ['Open', 'High', 'Low', 'Close', 'Volume'], 'spy'),
            ('^VIX',  ['Open', 'High', 'Low', 'Close'], 'vix'),

            # Rates (yield levels)
            ('^TNX',  ['Close'], 'tnx'),
            ('^IRX',  ['Close'], 'irx'),

            # Credit & bond ETFs
            ('HYG',   ['Close', 'Volume'], 'hyg'),
            ('IEF',   ['Close'], 'ief'),
            ('TIP',   ['Close'], 'tip'),
            ('SHY',   ['Close'], 'shy'),   # short-term bond
            ('TLT',   ['Close', 'Volume'], 'tlt'),   # long-term bond

            # Currency
            ('UUP',   ['Close'], 'uup'),

            # Commodity ETFs
            ('GLD',   ['Close'], 'gld'),
            ('SLV',   ['Close'], 'slv'),

            # Factor ETFs
            ('MTUM',  ['Close'], 'mtum'),
            ('SIZE',  ['Close'], 'size'),
            ('QUAL',  ['Close'], 'qual'),
            ('VLUE',  ['Close'], 'vlue'),
            ('USMV',  ['Close'], 'usmv'),
        ]

        data_frames = []

        for ticker, columns, prefix in tickers_config:
            df = self._fetch_ticker_data(ticker, start_date, end_date, columns, prefix)
            if not df.empty:
                data_frames.append(df)

        if not data_frames:
            raise RuntimeError("No data fetched from any ticker")

        data = pd.concat(data_frames, axis=1)

        if len(data) == 0:
            raise ValueError("No valid data after combining tickers")

        self.logger.info(
            f"Combined data: {len(data)} rows, {len(data.columns)} columns"
        )
        return data

    def save_data(self, data: pd.DataFrame, filename: str = "raw_daily_data.csv"):
     
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        self.logger.info(f"Saved to: {filepath}")
        self.logger.info(
            f"Shape: {data.shape}, "
            f"Range: {data.index[0]} to {data.index[-1]}"
        )

    def load_data(self, filename: str = "raw_daily_data.csv") -> pd.DataFrame:
        
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded: {filepath}, Shape: {data.shape}")
        return data

if __name__ == "__main__":
    ingestion = DataIngestion(data_dir="./data")
    data = ingestion.fetch_data(start_date='2019-01-01')
    ingestion.save_data(data, "raw_daily_data.csv")
    print(f"\nComplete: {len(data)} days, Columns: {len(data.columns)}")
    print(f"Columns: {list(data.columns)}")