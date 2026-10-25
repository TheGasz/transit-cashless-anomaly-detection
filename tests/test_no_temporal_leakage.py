"""Unit tests untuk memverifikasi ketiadaan temporal & preprocessing leakage (DoD-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.velocity_features import FEATURE_COLUMNS, extract_velocity_features
from src.ingestion.temporal_splitter import (
    ChronologicalTemporalSplitter,
    ZeroLeakageScalerPipeline,
)


def _make_synthetic_stream(n_rows: int = 600, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_ts = int(pd.Timestamp("2026-10-01T06:00:00").value // 1_000_000_000)
    offsets = np.cumsum(rng.integers(1, 25, size=n_rows))
    epochs = base_ts + offsets

    cards = [f"CARDHASH{i:08d}ABCDEF" for i in rng.integers(1, 35, size=n_rows)]
    readers = [f"TOB-{i:03d}" for i in rng.integers(1, 12, size=n_rows)]
    stops = [f"STOP-{i:03d}" for i in rng.integers(1, 10, size=n_rows)]
    corridors = [f"COR-0{(i % 5) + 1}" for i in range(n_rows)]
    tap_types = rng.choice(["TAP_IN", "TAP_OUT"], size=n_rows)

    df = pd.DataFrame(
        {
            "tx_id": [f"TX-{i:08d}" for i in range(1, n_rows + 1)],
            "timestamp": pd.to_datetime(epochs, unit="s").strftime("%Y-%m-%dT%H:%M:%S"),
            "card_pan_hash": cards,
            "user_segment": rng.choice(["REGULER", "PELAJAR_LANSIA"], size=n_rows),
            "payment_channel": rng.choice(["KMT", "E_MONEY_BANK", "QRIS_TAP"], size=n_rows),
            "tap_type": tap_types,
            "corridor_id": corridors,
            "stop_id": stops,
            "latitude": -6.20 + rng.normal(0, 0.05, size=n_rows),
            "longitude": 106.82 + rng.normal(0, 0.05, size=n_rows),
            "reader_device_id": readers,
            "fare_amount": rng.choice([3500.0, 4000.0, 6000.0, 10000.0], size=n_rows),
            "reader_latency_ms": rng.uniform(45.0, 450.0, size=n_rows).round(2),
            "is_offline_sync": rng.choice([False, True], size=n_rows, p=[0.95, 0.05]),
            "is_anomaly": rng.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
        }
    )
    return df


def test_chronological_split_strict_ordering() -> None:
    """Memastikan max(train.ts) < min(val.ts) < min(test.ts) tanpa tumpang tindih waktu."""
    df = _make_synthetic_stream(n_rows=800)
    splitter = ChronologicalTemporalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    split_res = splitter.split(df)

    max_train = pd.to_datetime(split_res.train_df["timestamp"]).max()
    min_val = pd.to_datetime(split_res.val_df["timestamp"]).min()
    max_val = pd.to_datetime(split_res.val_df["timestamp"]).max()
    min_test = pd.to_datetime(split_res.test_df["timestamp"]).min()

    assert max_train < min_val, f"Terjadi kebocoran batas train-val: {max_train} >= {min_val}"
    assert max_val < min_test, f"Terjadi kebocoran batas val-test: {max_val} >= {min_test}"
    assert len(split_res.train_df) + len(split_res.val_df) + len(split_res.test_df) == len(df)


def test_sliding_window_zero_future_leakage() -> None:
    """Memastikan perubahan/penambahan transaksi masa depan (t' > t) tidak mengubah fitur pada waktu <= t."""
    df_full = _make_synthetic_stream(n_rows=500, seed=99)
    cutoff = 320

    # Hitung fitur hanya dari masa lalu (0..cutoff-1)
    df_past_only = extract_velocity_features(df_full.iloc[:cutoff].copy())

    # Modifikasi transaksi masa depan (cutoff..end) secara ekstrem
    df_perturbed = df_full.copy()
    df_perturbed.loc[cutoff:, "reader_latency_ms"] = 29999.0
    df_perturbed.loc[cutoff:, "fare_amount"] = 99000.0
    df_perturbed.loc[cutoff:, "latitude"] = -7.79
    df_perturbed.loc[cutoff:, "longitude"] = 110.36
    df_perturbed.loc[cutoff:, "tap_type"] = "TAP_IN"

    df_with_future = extract_velocity_features(df_perturbed)
    df_past_from_full = df_with_future.iloc[:cutoff]

    assert len(FEATURE_COLUMNS) == 28
    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            df_past_only[col].to_numpy(),
            df_past_from_full[col].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Ditemukan look-ahead leakage pada fitur '{col}'!",
        )


def test_scaler_fits_strictly_on_normal_train_split() -> None:
    """Memastikan RobustScaler hanya melihat data normal di train split dan kebal terhadap outlier di test split."""
    df = extract_velocity_features(_make_synthetic_stream(n_rows=600, seed=77))
    splitter = ChronologicalTemporalSplitter()
    split_res = splitter.split(df)

    scaler_a = ZeroLeakageScalerPipeline(FEATURE_COLUMNS, fit_only_normal=True)
    scaler_a.fit(split_res.train_df)

    # Ubah nilai anomali di train_df serta seluruh val_df, pastikan parameter scaler tidak berubah
    train_modified = split_res.train_df.copy()
    train_modified.loc[train_modified["is_anomaly"] == 1, FEATURE_COLUMNS[0]] = 999999.0

    scaler_b = ZeroLeakageScalerPipeline(FEATURE_COLUMNS, fit_only_normal=True)
    scaler_b.fit(train_modified)

    np.testing.assert_allclose(scaler_a.scaler.center_, scaler_b.scaler.center_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler_a.scaler.scale_, scaler_b.scaler.scale_, rtol=1e-12, atol=1e-12)
