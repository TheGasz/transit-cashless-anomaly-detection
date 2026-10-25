"""Unit tests untuk memverifikasi akurasi Haversine, deteksi double-tap <= 8 detik, dan kecepatan ekstraksi fitur."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from src.features.geodesic import (
    compute_velocity_kmh,
    flag_impossible_travel,
    haversine_distance_km,
)
from src.features.velocity_features import (
    FEATURE_COLUMNS,
    OnlineVelocityFeatureExtractor,
    extract_velocity_features,
)


def test_haversine_geodesic_accuracy() -> None:
    """Memverifikasi akurasi perhitungan jarak geodesik Haversine pada koordinat nyata."""
    # Blok M (-6.2442, 106.8006) ke Jakarta Kota (-6.1376, 106.8133) ~ 11.93 km
    dist_blokm_kota = haversine_distance_km(-6.2442, 106.8006, -6.1376, 106.8133)
    assert 11.5 <= dist_blokm_kota <= 12.3

    # Titik identik harus menghasilkan jarak 0.0 km
    assert haversine_distance_km(-6.2000, 106.8000, -6.2000, 106.8000) == pytest.approx(0.0, abs=1e-9)

    # Uji vektorisasi array NumPy dan simetri
    lats1 = np.array([-6.2442, -6.5950])
    lons1 = np.array([106.8006, 106.7905])
    lats2 = np.array([-6.1376, -6.1376])
    lons2 = np.array([106.8133, 106.8133])

    d_forward = haversine_distance_km(lats1, lons1, lats2, lons2)
    d_backward = haversine_distance_km(lats2, lons2, lats1, lons1)
    np.testing.assert_allclose(d_forward, d_backward, rtol=1e-12)
    assert d_forward[1] > 48.0  # Bogor ke Jakarta Kota > 48 km


def test_impossible_travel_velocity_flag() -> None:
    """Memastikan perpindahan > 2.5 km dalam < 60 detik (v > 120 km/jam) ditandai sebagai impossible travel."""
    dist_km = 12.0
    delta_t_sec = 30.0  # 12 km dalam 30 detik = 1.440 km/jam
    vel = compute_velocity_kmh(dist_km, delta_t_sec)
    assert vel == pytest.approx(1440.0, rel=1e-6)
    assert flag_impossible_travel(dist_km, delta_t_sec, vel) == 1.0

    # Perjalanan normal BRT: 3.5 km dalam 600 detik (10 menit) = 21 km/jam
    vel_normal = compute_velocity_kmh(3.5, 600.0)
    assert vel_normal == pytest.approx(21.0, rel=1e-6)
    assert flag_impossible_travel(3.5, 600.0, vel_normal) == 0.0


def test_double_tap_under_8_seconds_and_online_consistency() -> None:
    """Memastikan pola double-tap <= 8 detik pada reader yang sama terdeteksi akurat oleh batch & online extractor."""
    events = [
        {
            "tx_id": "TX-00000001",
            "timestamp": "2026-10-05T07:30:00",
            "card_pan_hash": "CARD9988776655443322",
            "user_segment": "REGULER",
            "payment_channel": "QRIS_TAP",
            "tap_type": "TAP_IN",
            "corridor_id": "COR-03",
            "stop_id": "STOP-020",
            "latitude": -7.4200,
            "longitude": 109.1500,
            "reader_device_id": "TOB-045",
            "fare_amount": 4000.0,
            "reader_latency_ms": 140.0,
            "is_offline_sync": False,
        },
        {
            "tx_id": "TX-00000002",
            "timestamp": "2026-10-05T07:30:04",  # 4 detik kemudian (<= 8 detik)
            "card_pan_hash": "CARD9988776655443322",
            "user_segment": "REGULER",
            "payment_channel": "QRIS_TAP",
            "tap_type": "TAP_IN",
            "corridor_id": "COR-03",
            "stop_id": "STOP-020",
            "latitude": -7.4200,
            "longitude": 109.1500,
            "reader_device_id": "TOB-045",
            "fare_amount": 4000.0,
            "reader_latency_ms": 920.0,
            "is_offline_sync": False,
        },
    ]
    df = extract_velocity_features(pd.DataFrame(events))
    second_row = df.iloc[1]

    assert second_row["delta_t_card_sec"] == pytest.approx(4.0)
    assert second_row["is_same_reader"] == 1.0
    assert second_row["is_same_stop"] == 1.0
    assert second_row["is_same_tap_type"] == 1.0
    assert second_row["card_tap_count_15s"] == 2.0

    # Bandingkan dengan OnlineVelocityFeatureExtractor
    online_ext = OnlineVelocityFeatureExtractor()
    vec1 = online_ext.extract_single(events[0])
    vec2 = online_ext.extract_single(events[1])

    np.testing.assert_allclose(vec1, df.iloc[0][FEATURE_COLUMNS].to_numpy(dtype=np.float64), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vec2, df.iloc[1][FEATURE_COLUMNS].to_numpy(dtype=np.float64), rtol=1e-6, atol=1e-6)


def test_feature_extraction_batch_latency_under_10ms() -> None:
    """Memastikan waktu komputasi ekstraksi 28 fitur untuk micro-batch (< 10 ms/batch sesuai F-03)."""
    rng = np.random.default_rng(42)
    n_batch = 250
    base_ts = int(pd.Timestamp("2026-10-02T08:00:00").value // 1_000_000_000)
    df_batch = pd.DataFrame(
        {
            "tx_id": [f"TX-{i:08d}" for i in range(1, n_batch + 1)],
            "timestamp": pd.to_datetime(base_ts + np.arange(n_batch) * 2, unit="s").strftime("%Y-%m-%dT%H:%M:%S"),
            "card_pan_hash": [f"CARDHASH{i % 40:08d}" for i in range(n_batch)],
            "user_segment": ["REGULER"] * n_batch,
            "payment_channel": ["E_MONEY_BANK"] * n_batch,
            "tap_type": ["TAP_IN" if i % 2 == 0 else "TAP_OUT" for i in range(n_batch)],
            "corridor_id": ["COR-01"] * n_batch,
            "stop_id": [f"STOP-00{(i % 5) + 1}" for i in range(n_batch)],
            "latitude": -6.20 + rng.normal(0, 0.01, size=n_batch),
            "longitude": 106.81 + rng.normal(0, 0.01, size=n_batch),
            "reader_device_id": [f"TOB-00{(i % 8) + 1}" for i in range(n_batch)],
            "fare_amount": [3500.0] * n_batch,
            "reader_latency_ms": rng.uniform(50.0, 180.0, size=n_batch),
            "is_offline_sync": [False] * n_batch,
        }
    )

    # Warm-up
    _ = extract_velocity_features(df_batch)

    timings_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = extract_velocity_features(df_batch)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(timings_ms))
    assert res.shape[1] >= 28
    assert median_ms < 10.0, f"Komputasi batch fitur melebihi 10 ms: {median_ms:.2f} ms"
