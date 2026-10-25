"""Pembagian data kronologis ketat (Zero-Leakage Temporal Split) & RobustScaler Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class TemporalSplitResult:
    """Hasil pembagian kronologis Train, Validation, dan Test."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_end_ts: pd.Timestamp
    val_end_ts: pd.Timestamp


class ChronologicalTemporalSplitter:
    """Pembagi deret waktu transaksi secara kronologis ketat tanpa kebocoran masa depan."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        timestamp_col: str = "timestamp",
    ) -> None:
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Total rasio train + val + test harus sama dengan 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.timestamp_col = timestamp_col

    def split(self, df: pd.DataFrame) -> TemporalSplitResult:
        """Membagi DataFrame secara berurutan berdasarkan timestamp dengan batas waktu tegas."""
        if df is None or len(df) < 10:
            raise ValueError("DataFrame terlalu kecil untuk dibagi secara kronologis.")

        work_df = df.copy()
        work_df["_parsed_ts"] = pd.to_datetime(work_df[self.timestamp_col])
        work_df = work_df.sort_values(["_parsed_ts", "tx_id"], kind="mergesort").reset_index(drop=True)

        n = len(work_df)
        train_idx = int(n * self.train_ratio)
        val_idx = int(n * (self.train_ratio + self.val_ratio))

        # Pastikan tidak ada timestamp yang sama di perbatasan train/val maupun val/test
        train_boundary_ts = work_df["_parsed_ts"].iloc[train_idx - 1]
        while train_idx < val_idx - 1 and work_df["_parsed_ts"].iloc[train_idx] <= train_boundary_ts:
            train_idx += 1

        val_boundary_ts = work_df["_parsed_ts"].iloc[val_idx - 1]
        while val_idx < n - 1 and work_df["_parsed_ts"].iloc[val_idx] <= val_boundary_ts:
            val_idx += 1

        train_df = work_df.iloc[:train_idx].drop(columns=["_parsed_ts"]).reset_index(drop=True)
        val_df = work_df.iloc[train_idx:val_idx].drop(columns=["_parsed_ts"]).reset_index(drop=True)
        test_df = work_df.iloc[val_idx:].drop(columns=["_parsed_ts"]).reset_index(drop=True)

        max_train_ts = pd.to_datetime(train_df[self.timestamp_col]).max()
        min_val_ts = pd.to_datetime(val_df[self.timestamp_col]).min()
        max_val_ts = pd.to_datetime(val_df[self.timestamp_col]).max()
        min_test_ts = pd.to_datetime(test_df[self.timestamp_col]).min()

        if not (max_train_ts < min_val_ts and max_val_ts < min_test_ts):
            raise RuntimeError(
                f"Pelanggaran batas kronologis: max_train={max_train_ts}, "
                f"min_val={min_val_ts}, max_val={max_val_ts}, min_test={min_test_ts}"
            )

        return TemporalSplitResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_end_ts=max_train_ts,
            val_end_ts=max_val_ts,
        )


class ZeroLeakageScalerPipeline:
    """Membungkus RobustScaler yang di-fit murni pada data normal di train split."""

    def __init__(self, feature_columns: list[str], fit_only_normal: bool = True) -> None:
        self.feature_columns = list(feature_columns)
        self.fit_only_normal = fit_only_normal
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.is_fitted: bool = False
        self.fitted_max_timestamp: Optional[pd.Timestamp] = None
        self.clip_value: float = 15.0

    def fit(self, train_df: pd.DataFrame, timestamp_col: str = "timestamp") -> "ZeroLeakageScalerPipeline":
        """Fit RobustScaler hanya pada train split (dan khusus transaksi normal bila diaktifkan)."""
        if self.fit_only_normal and "is_anomaly" in train_df.columns:
            fit_subset = train_df[train_df["is_anomaly"] == 0]
            if len(fit_subset) == 0:
                fit_subset = train_df
        else:
            fit_subset = train_df

        x_train = fit_subset[self.feature_columns].to_numpy(dtype=np.float64)
        self.scaler.fit(x_train)
        self.is_fitted = True
        if timestamp_col in train_df.columns:
            self.fitted_max_timestamp = pd.to_datetime(train_df[timestamp_col]).max()
        return self

    def transform(self, df_or_array: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transformasi vektor fitur ke ruang terstandardisasi RobustScaler."""
        if not self.is_fitted:
            raise RuntimeError("ZeroLeakageScalerPipeline belum di-fit pada train split!")

        if isinstance(df_or_array, pd.DataFrame):
            x = df_or_array[self.feature_columns].to_numpy(dtype=np.float64)
        else:
            x = np.asarray(df_or_array, dtype=np.float64)

        scaled = self.scaler.transform(x)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        return np.clip(scaled, -self.clip_value, self.clip_value).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Menyimpan artefak scaler ke file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "fit_only_normal": self.fit_only_normal,
                "fitted_max_timestamp": str(self.fitted_max_timestamp) if self.fitted_max_timestamp else None,
                "clip_value": self.clip_value,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ZeroLeakageScalerPipeline":
        """Memuat artefak scaler dari file."""
        payload = joblib.load(path)
        instance = cls(
            feature_columns=payload["feature_columns"],
            fit_only_normal=payload.get("fit_only_normal", True),
        )
        instance.scaler = payload["scaler"]
        instance.is_fitted = True
        ts_str = payload.get("fitted_max_timestamp")
        instance.fitted_max_timestamp = pd.to_datetime(ts_str) if ts_str else None
        instance.clip_value = payload.get("clip_value", 15.0)
        return instance
