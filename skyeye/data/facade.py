from __future__ import annotations

import os
import pickle
from typing import Optional, Union, Sequence

import h5py
import numpy as np
import pandas as pd


class DataFacade:
    def __init__(self):
        self.provider = None
        source = os.environ.get("SKYEYE_DATA_SOURCE", "rqdatac").strip().lower()
        if source not in ("bundle", "local"):
            try:
                from skyeye.data.provider import RQDataProvider
                self.provider = RQDataProvider()
            except Exception:
                self.provider = None

    def get_daily_bars(
        self,
        order_book_ids: Union[str, Sequence[str]],
        start_date,
        end_date,
        fields: Optional[Sequence[str]] = None,
        adjust_type: str = "pre",
    ) -> Optional[pd.DataFrame]:
        if self.provider is not None:
            try:
                df = self.provider.get_price(
                    order_book_ids, start_date, end_date, frequency="1d", adjust_type=adjust_type, fields=list(fields) if fields else None
                )
                if df is not None:
                    return df
            except Exception:
                pass
        if str(adjust_type).lower() not in ("none", "pre"):
            return None
        ids = [order_book_ids] if isinstance(order_book_ids, str) else list(order_book_ids)
        frames = []
        for ob_id in ids:
            df = self._read_bundle_daily(ob_id, start_date, end_date)
            if df is None:
                continue
            if fields is not None:
                keep = [c for c in df.columns if c in set(fields)]
                if keep:
                    df = df[keep]
            df = df.loc[str(start_date):str(end_date)]
            df.insert(0, "order_book_id", ob_id)
            frames.append(df)
        if not frames:
            return None
        result = pd.concat(frames, axis=0, sort=False)
        result.index.name = "date"
        return result

    def get_trading_dates(self, start_date, end_date):
        if self.provider is not None:
            try:
                return self.provider.get_trading_dates(start_date, end_date)
            except Exception:
                pass
        path = os.path.join(self._bundle_path(), "trading_dates.npy")
        if not os.path.exists(path):
            return []
        arr = np.load(path)
        dates = [pd.Timestamp(str(int(x))) for x in arr]
        return [d for d in dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    def all_instruments(self, type: Optional[str] = None, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[pd.DataFrame]:
        if self.provider is not None:
            try:
                return self.provider.get_instruments(type=type or "CS", date=date)
            except Exception:
                pass
        pkl = os.path.join(self._bundle_path(), "instruments.pk")
        if not os.path.exists(pkl):
            return None
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        if type:
            df = df[df.get("type") == type]
        return df

    def index_components(self, index_code: str, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[list[str]]:
        if self.provider is None:
            return None
        try:
            return self.provider.get_index_components(index_code, date=date)
        except Exception:
            return None

    def index_weights(self, index_code: str, date: Optional[Union[str, int, pd.Timestamp]] = None) -> Optional[pd.Series]:
        if self.provider is None:
            return None
        try:
            return self.provider.get_index_weights(index_code, date=date)
        except Exception:
            return None

    def get_factor(
        self,
        order_book_ids: Union[str, Sequence[str]],
        factors: Union[str, Sequence[str]],
        start_date,
        end_date,
    ) -> Optional[pd.DataFrame]:
        if self.provider is None:
            return None
        try:
            return self.provider.get_factors(order_book_ids, factors, start_date, end_date)
        except Exception:
            return None

    def _bundle_path(self) -> str:
        return os.path.expanduser("~/.rqalpha/bundle")

    def _read_bundle_daily(self, order_book_id: str, start_date, end_date) -> Optional[pd.DataFrame]:
        base = self._bundle_path()
        for name in ["stocks.h5", "indexes.h5", "funds.h5"]:
            path = os.path.join(base, name)
            if not os.path.exists(path):
                continue
            with h5py.File(path, "r") as h5:
                if order_book_id not in h5:
                    continue
                rec = h5[order_book_id][:]
                df = pd.DataFrame(rec)
                if "datetime" in df.columns:
                    if np.issubdtype(df["datetime"].dtype, np.integer):
                        s = df["datetime"].astype(str).str[:8]
                        df["date"] = pd.to_datetime(s, format="%Y%m%d")
                    else:
                        df["date"] = pd.to_datetime(df["datetime"]).normalize()
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                else:
                    return None
                df = df.set_index("date").sort_index()
                return df
        return None
