"""Ekstraktor 28 fitur kausal sliding-window spasio-temporal & telemetri perangkat TOB."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.geodesic import (
    compute_velocity_kmh,
    flag_impossible_travel,
    haversine_distance_km,
)

FEATURE_COLUMNS: List[str] = [
    # Keluarga A: Kecepatan & Jarak Spasio-Temporal Kartu
    "delta_t_card_sec",
    "delta_d_km",
    "velocity_kmh",
    "is_impossible_velocity_flag",
    # Keluarga B: Frekuensi & Pola Double-Tap (Velocity Counters)
    "card_tap_count_15s",
    "card_tap_count_30s",
    "card_tap_count_1m",
    "card_tap_count_5m",
    "card_tap_count_1h",
    "card_tap_count_24h",
    "is_same_reader",
    "is_same_stop",
    "is_same_tap_type",
    "incomplete_tapout_ratio_7d",
    "consecutive_tapin_count",
    "passback_same_stop_count_5m",
    # Keluarga C: Telemetri Kesehatan Perangkat TOB / Gate
    "reader_latency_ms",
    "reader_rolling_latency_mean_5m",
    "reader_rolling_latency_std_5m",
    "reader_latency_zscore",
    "reader_retry_ratio_5m",
    "sync_delay_seconds",
    "is_offline_sync_int",
    # Keluarga D: Deviasi Nilai Tarif & Siklus Waktu
    "hour_sin",
    "hour_cos",
    "day_sin",
    "is_peak_hour",
    "fare_to_corridor_std_ratio",
]

DEFAULT_CORRIDOR_STD_FARE: Dict[str, float] = {
    "COR-01": 3500.0,
    "COR-02": 6000.0,
    "COR-03": 4000.0,
    "COR-04": 3600.0,
    "COR-05": 10000.0,
}

DEFAULT_FIRST_TAP_DELTA_SEC: float = 86400.0
KEY_STRIDE_INT: int = 10_000_000_000  # > 300 tahun dalam detik (int64 exact)


def _compute_consecutive_tapin(card_codes: np.ndarray, is_tap_in: np.ndarray) -> np.ndarray:
    """Menghitung jumlah TAP_IN beruntun tanpa TAP_OUT secara kausal per kartu.

    Karena diurutkan per kartu secara kronologis, setiap TAP_OUT me-reset penghitung ke 0.
    """
    n = len(card_codes)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Reset terjadi saat kartu berganti atau saat transaksi adalah TAP_OUT (is_tap_in == 0)
    card_change = np.empty(n, dtype=bool)
    card_change[0] = True
    card_change[1:] = card_codes[1:] != card_codes[:-1]

    # Gunakan trik cumsum dengan nilai 1 untuk TAP_IN, dan reset pada TAP_OUT / pergantian kartu
    reset_mask = card_change | (is_tap_in == 0)
    idx = np.arange(n, dtype=np.int64)
    # Nilai cumsum dari is_tap_in
    csum = np.cumsum(is_tap_in, dtype=np.int64)
    # Propagasi maksimum indeks reset per segmen
    reset_positions = np.where(reset_mask, idx, 0)
    last_reset_pos = np.maximum.accumulate(reset_positions)
    base_val = csum[last_reset_pos] - is_tap_in[last_reset_pos]
    consec = np.where(is_tap_in == 1, csum - base_val, 0)
    return consec.astype(np.float64)


def extract_velocity_features(
    df: pd.DataFrame,
    corridor_fares: Optional[Dict[str, float]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Mengekstrak 28 fitur spasio-temporal dan telemetri perangkat secara kausal (t' <= t).

    Seluruh perhitungan jendela waktu (15s, 30s, 1m, 5m, 1h, 24h, 7d) dijamin bebas
    dari look-ahead leakage: perubahan atau penambahan transaksi di masa depan (t' > t)
    tidak akan pernah mengubah nilai fitur pada waktu t.
    """
    if df is None or df.empty:
        out_empty = df.copy() if df is not None else pd.DataFrame()
        for col in FEATURE_COLUMNS:
            out_empty[col] = pd.Series(dtype=np.float64)
        return out_empty

    fares_map = corridor_fares or DEFAULT_CORRIDOR_STD_FARE
    work = df if inplace else df.copy()

    n = len(work)
    orig_index = np.arange(n, dtype=np.int64)
    ts_series = pd.to_datetime(work["timestamp"])
    ts_sec_raw = ts_series.dt.as_unit("s").astype("int64").to_numpy().astype(np.float64)

    # Urutkan secara kronologis ketat terlebih dahulu (stabil terhadap urutan asli)
    chrono_order = np.lexsort((orig_index, ts_sec_raw))

    ts_sec = ts_sec_raw[chrono_order]
    ts_sec_int = ts_sec.astype(np.int64)
    rel_ts = ts_sec

    # Ambil kolom-kolom dalam urutan kronologis menggunakan hash table Cython pd.factorize
    card_codes = pd.factorize(work["card_pan_hash"], sort=False)[0][chrono_order]
    reader_codes = pd.factorize(work["reader_device_id"], sort=False)[0][chrono_order]
    stop_codes = pd.factorize(work["stop_id"], sort=False)[0][chrono_order]
    is_tap_in_chrono = (work["tap_type"].to_numpy() == "TAP_IN").astype(np.int64)[chrono_order]
    corridors_chrono = work["corridor_id"].to_numpy()[chrono_order]
    lat_chrono = work["latitude"].to_numpy(dtype=np.float64)[chrono_order]
    lon_chrono = work["longitude"].to_numpy(dtype=np.float64)[chrono_order]
    fare_chrono = work["fare_amount"].to_numpy(dtype=np.float64)[chrono_order]
    latency_chrono = work["reader_latency_ms"].to_numpy(dtype=np.float64)[chrono_order]
    offline_chrono = work["is_offline_sync"].to_numpy(dtype=np.float64)[chrono_order]

    if "server_received_timestamp" in work.columns and work["server_received_timestamp"].notna().any():
        srv_ts = pd.to_datetime(work["server_received_timestamp"]).fillna(ts_series)
        srv_sec = srv_ts.dt.as_unit("s").astype("int64").to_numpy().astype(np.float64)[chrono_order]
        sync_delay_chrono = np.maximum(0.0, srv_sec - ts_sec)
    else:
        # Jika tidak ada kolom server_received_timestamp eksplisit, estimasi kausal dari flag offline & latensi
        sync_delay_chrono = np.where(
            offline_chrono > 0.5,
            3600.0 + latency_chrono * 0.5,
            latency_chrono / 1000.0,
        )

    # =========================================================================
    # 1. KELUARGA A & B: FITUR BERBASIS KARTU (CARD-SORTED CAUSAL WINDOWS)
    # =========================================================================
    card_sort_idx = np.argsort(card_codes, kind="mergesort")
    c_codes = card_codes[card_sort_idx]
    c_rel_ts = rel_ts[card_sort_idx]
    c_lat = lat_chrono[card_sort_idx]
    c_lon = lon_chrono[card_sort_idx]
    c_reader = reader_codes[card_sort_idx]
    c_stop = stop_codes[card_sort_idx]
    c_tap_in = is_tap_in_chrono[card_sort_idx]

    same_prev_card = np.zeros(n, dtype=bool)
    if n > 1:
        same_prev_card[1:] = c_codes[1:] == c_codes[:-1]

    # Delta t & Delta d dari tap sebelumnya oleh kartu yang sama
    prev_ts = np.roll(c_rel_ts, 1)
    prev_lat = np.roll(c_lat, 1)
    prev_lon = np.roll(c_lon, 1)
    prev_reader = np.roll(c_reader, 1)
    prev_stop = np.roll(c_stop, 1)
    prev_tap_in = np.roll(c_tap_in, 1)

    c_delta_t = np.where(same_prev_card, np.maximum(0.0, c_rel_ts - prev_ts), DEFAULT_FIRST_TAP_DELTA_SEC)
    raw_dist = haversine_distance_km(
        np.where(same_prev_card, prev_lat, c_lat),
        np.where(same_prev_card, prev_lon, c_lon),
        c_lat,
        c_lon,
    )
    c_delta_d = np.where(same_prev_card, raw_dist, 0.0)
    c_velocity = np.where(
        same_prev_card,
        compute_velocity_kmh(c_delta_d, c_delta_t, min_delta_t_sec=1.0),
        0.0,
    )
    c_imp_flag = np.where(
        same_prev_card,
        flag_impossible_travel(c_delta_d, c_delta_t, c_velocity),
        0.0,
    )

    c_same_reader = np.where(same_prev_card & (c_reader == prev_reader) & (c_stop == prev_stop), 1.0, 0.0)
    c_same_stop = np.where(same_prev_card & (c_stop == prev_stop), 1.0, 0.0)
    c_same_tap_type = np.where(same_prev_card & (c_tap_in == prev_tap_in), 1.0, 0.0)

    # Key gabungan int64 untuk searchsorted per kartu: dijamin non-decreasing dan exact
    c_key = c_codes.astype(np.int64) * KEY_STRIDE_INT + ts_sec_int[card_sort_idx]
    idx_seq = np.arange(n, dtype=np.int64)

    def _window_count(key_arr: np.ndarray, window_sec: int) -> np.ndarray:
        left = np.searchsorted(key_arr, key_arr - window_sec, side="left")
        return (idx_seq - left + 1).astype(np.float64)

    c_cnt_15s = _window_count(c_key, 15)
    c_cnt_30s = _window_count(c_key, 30)
    c_cnt_1m = _window_count(c_key, 60)
    c_cnt_5m = _window_count(c_key, 300)
    c_cnt_1h = _window_count(c_key, 3600)
    c_cnt_24h = _window_count(c_key, 86400)

    # Incomplete tap-out ratio dalam 7 hari (604800 detik)
    left_7d = np.searchsorted(c_key, c_key - 604800, side="left")
    prefix_tapin = np.concatenate(([0], np.cumsum(c_tap_in, dtype=np.int64)))
    prefix_tapout = np.concatenate(([0], np.cumsum(1 - c_tap_in, dtype=np.int64)))
    tapin_7d = (prefix_tapin[idx_seq + 1] - prefix_tapin[left_7d]).astype(np.float64)
    tapout_7d = (prefix_tapout[idx_seq + 1] - prefix_tapout[left_7d]).astype(np.float64)
    safe_tapin_7d = np.maximum(tapin_7d, 1.0)
    c_inc_ratio_7d = np.where(
        tapin_7d > 0.0,
        np.clip((tapin_7d - tapout_7d) / safe_tapin_7d, 0.0, 1.0),
        0.0,
    )

    c_consec_tapin = _compute_consecutive_tapin(c_codes, c_tap_in)

    # Kembalikan fitur kartu ke urutan kronologis
    inv_card_sort = np.empty(n, dtype=np.int64)
    inv_card_sort[card_sort_idx] = idx_seq

    delta_t_card_sec = c_delta_t[inv_card_sort]
    delta_d_km = c_delta_d[inv_card_sort]
    velocity_kmh = c_velocity[inv_card_sort]
    is_impossible_velocity_flag = c_imp_flag[inv_card_sort]
    card_tap_count_15s = c_cnt_15s[inv_card_sort]
    card_tap_count_30s = c_cnt_30s[inv_card_sort]
    card_tap_count_1m = c_cnt_1m[inv_card_sort]
    card_tap_count_5m = c_cnt_5m[inv_card_sort]
    card_tap_count_1h = c_cnt_1h[inv_card_sort]
    card_tap_count_24h = c_cnt_24h[inv_card_sort]
    is_same_reader = c_same_reader[inv_card_sort]
    is_same_stop = c_same_stop[inv_card_sort]
    is_same_tap_type = c_same_tap_type[inv_card_sort]
    incomplete_tapout_ratio_7d = c_inc_ratio_7d[inv_card_sort]
    consecutive_tapin_count = c_consec_tapin[inv_card_sort]

    # Pass-back same stop count dalam 5 menit terakhir per (card, stop)
    card_stop_code = card_codes.astype(np.int64) * (int(stop_codes.max()) + 2) + stop_codes.astype(np.int64)
    cs_sort_idx = np.argsort(card_stop_code, kind="mergesort")
    cs_key = card_stop_code[cs_sort_idx] * KEY_STRIDE_INT + ts_sec_int[cs_sort_idx]
    cs_cnt_5m = _window_count(cs_key, 300)
    inv_cs_sort = np.empty(n, dtype=np.int64)
    inv_cs_sort[cs_sort_idx] = idx_seq
    passback_same_stop_count_5m = cs_cnt_5m[inv_cs_sort]

    # =========================================================================
    # 2. KELUARGA C: TELEMETRI PERANGKAT TOB / GATE (READER-SORTED CAUSAL)
    # =========================================================================
    is_retry_event = ((delta_t_card_sec <= 15.0) & (is_same_reader > 0.5)).astype(np.int64)

    reader_sort_idx = np.argsort(reader_codes, kind="mergesort")
    r_codes = reader_codes[reader_sort_idx]
    r_lat_ms = latency_chrono[reader_sort_idx]
    r_retry = is_retry_event[reader_sort_idx]

    r_key = r_codes.astype(np.int64) * KEY_STRIDE_INT + ts_sec_int[reader_sort_idx]
    r_left_5m = np.searchsorted(r_key, r_key - 300, side="left")
    r_count_5m = (idx_seq - r_left_5m + 1).astype(np.float64)

    # Gunakan int64 scaled (0.01 ms) untuk prefix sum agar 100% bebas floating-point drift
    r_lat_scaled = np.round(r_lat_ms * 100.0).astype(np.int64)
    r_lat_sq_scaled = r_lat_scaled * r_lat_scaled

    pref_lat = np.concatenate(([0], np.cumsum(r_lat_scaled, dtype=np.int64)))
    pref_lat_sq = np.concatenate(([0], np.cumsum(r_lat_sq_scaled, dtype=np.int64)))
    pref_retry = np.concatenate(([0], np.cumsum(r_retry, dtype=np.int64)))

    win_sum = (pref_lat[idx_seq + 1] - pref_lat[r_left_5m]).astype(np.float64) / 100.0
    win_sq_sum = (pref_lat_sq[idx_seq + 1] - pref_lat_sq[r_left_5m]).astype(np.float64) / 10000.0
    win_retry_sum = (pref_retry[idx_seq + 1] - pref_retry[r_left_5m]).astype(np.float64)

    r_mean_5m = win_sum / r_count_5m
    r_var_5m = np.maximum(0.0, (win_sq_sum / r_count_5m) - (r_mean_5m * r_mean_5m))
    r_std_5m = np.sqrt(r_var_5m)
    r_zscore = (r_lat_ms - r_mean_5m) / np.maximum(r_std_5m, 5.0)
    r_retry_ratio_5m = win_retry_sum / r_count_5m

    inv_reader_sort = np.empty(n, dtype=np.int64)
    inv_reader_sort[reader_sort_idx] = idx_seq

    reader_rolling_latency_mean_5m = r_mean_5m[inv_reader_sort]
    reader_rolling_latency_std_5m = r_std_5m[inv_reader_sort]
    reader_latency_zscore = r_zscore[inv_reader_sort]
    reader_retry_ratio_5m = r_retry_ratio_5m[inv_reader_sort]

    # =========================================================================
    # 3. KELUARGA D: SIKLUS WAKTU & DEVIASI TARIF KORIDOR (VEKTORISASI NUMPY)
    # =========================================================================
    sec_of_day = np.mod(ts_sec_int, 86400)
    fractional_hour = (sec_of_day // 3600).astype(np.float64) + ((sec_of_day % 3600) // 60).astype(np.float64) / 60.0
    day_of_week = (((ts_sec_int // 86400) + 3) % 7).astype(np.float64)

    hour_sin = np.sin(2.0 * np.pi * fractional_hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * fractional_hour / 24.0)
    day_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
    is_peak_hour = (
        ((fractional_hour >= 6.0) & (fractional_hour <= 9.0))
        | ((fractional_hour >= 16.5) & (fractional_hour <= 19.5))
    ).astype(np.float64)

    std_fares = np.array([fares_map.get(str(c), 4000.0) for c in corridors_chrono], dtype=np.float64)
    fare_to_corridor_std_ratio = fare_chrono / np.maximum(std_fares, 1.0)

    # Kembalikan seluruh fitur dari urutan kronologis ke urutan baris asli DataFrame
    inv_chrono = np.empty(n, dtype=np.int64)
    inv_chrono[chrono_order] = idx_seq

    features_dict = {
        "delta_t_card_sec": delta_t_card_sec[inv_chrono].astype(np.float64),
        "delta_d_km": delta_d_km[inv_chrono].astype(np.float64),
        "velocity_kmh": velocity_kmh[inv_chrono].astype(np.float64),
        "is_impossible_velocity_flag": is_impossible_velocity_flag[inv_chrono].astype(np.float64),
        "card_tap_count_15s": card_tap_count_15s[inv_chrono].astype(np.float64),
        "card_tap_count_30s": card_tap_count_30s[inv_chrono].astype(np.float64),
        "card_tap_count_1m": card_tap_count_1m[inv_chrono].astype(np.float64),
        "card_tap_count_5m": card_tap_count_5m[inv_chrono].astype(np.float64),
        "card_tap_count_1h": card_tap_count_1h[inv_chrono].astype(np.float64),
        "card_tap_count_24h": card_tap_count_24h[inv_chrono].astype(np.float64),
        "is_same_reader": is_same_reader[inv_chrono].astype(np.float64),
        "is_same_stop": is_same_stop[inv_chrono].astype(np.float64),
        "is_same_tap_type": is_same_tap_type[inv_chrono].astype(np.float64),
        "incomplete_tapout_ratio_7d": incomplete_tapout_ratio_7d[inv_chrono].astype(np.float64),
        "consecutive_tapin_count": consecutive_tapin_count[inv_chrono].astype(np.float64),
        "passback_same_stop_count_5m": passback_same_stop_count_5m[inv_chrono].astype(np.float64),
        "reader_latency_ms": latency_chrono[inv_chrono].astype(np.float64),
        "reader_rolling_latency_mean_5m": reader_rolling_latency_mean_5m[inv_chrono].astype(np.float64),
        "reader_rolling_latency_std_5m": reader_rolling_latency_std_5m[inv_chrono].astype(np.float64),
        "reader_latency_zscore": reader_latency_zscore[inv_chrono].astype(np.float64),
        "reader_retry_ratio_5m": reader_retry_ratio_5m[inv_chrono].astype(np.float64),
        "sync_delay_seconds": sync_delay_chrono[inv_chrono].astype(np.float64),
        "is_offline_sync_int": offline_chrono[inv_chrono].astype(np.float64),
        "hour_sin": hour_sin[inv_chrono].astype(np.float64),
        "hour_cos": hour_cos[inv_chrono].astype(np.float64),
        "day_sin": day_sin[inv_chrono].astype(np.float64),
        "is_peak_hour": is_peak_hour[inv_chrono].astype(np.float64),
        "fare_to_corridor_std_ratio": fare_to_corridor_std_ratio[inv_chrono].astype(np.float64),
    }

    feat_mat = np.column_stack([features_dict[col] for col in FEATURE_COLUMNS])
    feat_df = pd.DataFrame(feat_mat, index=work.index, columns=FEATURE_COLUMNS)
    overlap_cols = [c for c in FEATURE_COLUMNS if c in work.columns]
    base_df = work.drop(columns=overlap_cols) if overlap_cols else work
    return pd.concat([base_df, feat_df], axis=1)


class OnlineVelocityFeatureExtractor:
    """Ekstraktor fitur waktu nyata (< 0.5 ms/event) untuk layanan FastAPI."""

    def __init__(self, corridor_fares: Optional[Dict[str, float]] = None) -> None:
        self.corridor_fares = corridor_fares or DEFAULT_CORRIDOR_STD_FARE
        # Riwayat per kartu: deque of (ts_sec, lat, lon, reader_id, stop_id, tap_type)
        self.card_history: Dict[str, Deque[Tuple[float, float, float, str, str, str]]] = {}
        self.card_consec_tapin: Dict[str, int] = {}
        # Riwayat per perangkat TOB: deque of (ts_sec, latency_ms, is_retry)
        self.reader_history: Dict[str, Deque[Tuple[float, float, int]]] = {}

    def reset(self) -> None:
        """Mengosongkan state in-memory."""
        self.card_history.clear()
        self.card_consec_tapin.clear()
        self.reader_history.clear()

    def extract_single(self, event: Dict[str, Any], update_state: bool = True) -> np.ndarray:
        """Menghitung vektor 28 fitur untuk satu transaksi masuk secara kausal."""
        ts_dt = datetime.fromisoformat(str(event["timestamp"]).replace("Z", "+00:00"))
        ts_sec = ts_dt.timestamp()

        card_id = str(event["card_pan_hash"])
        reader_id = str(event["reader_device_id"])
        stop_id = str(event["stop_id"])
        tap_type = str(event["tap_type"])
        corridor_id = str(event["corridor_id"])
        lat = float(event["latitude"])
        lon = float(event["longitude"])
        fare = float(event["fare_amount"])
        latency_ms = float(event["reader_latency_ms"])
        is_offline = 1.0 if bool(event.get("is_offline_sync", False)) else 0.0

        c_hist = self.card_history.get(card_id)
        if c_hist and len(c_hist) > 0:
            last_ts, last_lat, last_lon, last_reader, last_stop, last_tap = c_hist[-1]
            delta_t = max(0.0, ts_sec - last_ts)
            delta_d = float(haversine_distance_km(last_lat, last_lon, lat, lon))
            vel_kmh = float(compute_velocity_kmh(delta_d, delta_t, min_delta_t_sec=1.0))
            imp_flag = float(flag_impossible_travel(delta_d, delta_t, vel_kmh))
            same_reader = 1.0 if (reader_id == last_reader and stop_id == last_stop) else 0.0
            same_stop = 1.0 if stop_id == last_stop else 0.0
            same_tap = 1.0 if tap_type == last_tap else 0.0
        else:
            delta_t = DEFAULT_FIRST_TAP_DELTA_SEC
            delta_d = 0.0
            vel_kmh = 0.0
            imp_flag = 0.0
            same_reader = 0.0
            same_stop = 0.0
            same_tap = 0.0

        # Hitung frekuensi jendela waktu kartu (termasuk transaksi saat ini)
        cnt_15s = 1.0
        cnt_30s = 1.0
        cnt_1m = 1.0
        cnt_5m = 1.0
        cnt_1h = 1.0
        cnt_24h = 1.0
        passback_5m = 1.0
        tapin_7d = 1.0 if tap_type == "TAP_IN" else 0.0
        tapout_7d = 1.0 if tap_type == "TAP_OUT" else 0.0

        if c_hist:
            for h_ts, _, _, _, h_stop, h_tap in reversed(c_hist):
                age = ts_sec - h_ts
                if age < 0:
                    continue
                if age <= 15.0:
                    cnt_15s += 1.0
                if age <= 30.0:
                    cnt_30s += 1.0
                if age <= 60.0:
                    cnt_1m += 1.0
                if age <= 300.0:
                    cnt_5m += 1.0
                    if h_stop == stop_id:
                        passback_5m += 1.0
                if age <= 3600.0:
                    cnt_1h += 1.0
                if age <= 86400.0:
                    cnt_24h += 1.0
                if age <= 604800.0:
                    if h_tap == "TAP_IN":
                        tapin_7d += 1.0
                    else:
                        tapout_7d += 1.0
                else:
                    break

        inc_ratio_7d = float(np.clip((tapin_7d - tapout_7d) / tapin_7d, 0.0, 1.0)) if tapin_7d > 0 else 0.0
        prev_consec = self.card_consec_tapin.get(card_id, 0)
        consec_tapin = float(prev_consec + 1) if tap_type == "TAP_IN" else 0.0

        # Telemetri perangkat TOB dalam 5 menit terakhir
        is_retry = 1 if (delta_t <= 15.0 and same_reader > 0.5) else 0
        r_hist = self.reader_history.get(reader_id)
        lat_vals = [latency_ms]
        retry_vals = [is_retry]
        if r_hist:
            for r_ts, r_lat, r_ret in reversed(r_hist):
                age = ts_sec - r_ts
                if 0.0 <= age <= 300.0:
                    lat_vals.append(r_lat)
                    retry_vals.append(r_ret)
                elif age > 300.0:
                    break

        lat_arr = np.asarray(lat_vals, dtype=np.float64)
        r_mean_5m = float(np.mean(lat_arr))
        r_std_5m = float(np.std(lat_arr))
        r_zscore = (latency_ms - r_mean_5m) / max(r_std_5m, 5.0)
        r_retry_ratio = float(np.mean(retry_vals))

        sync_delay = (3600.0 + latency_ms * 0.5) if is_offline > 0.5 else (latency_ms / 1000.0)

        frac_hour = ts_dt.hour + ts_dt.minute / 60.0
        hour_sin = float(np.sin(2.0 * np.pi * frac_hour / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * frac_hour / 24.0))
        day_sin = float(np.sin(2.0 * np.pi * ts_dt.weekday() / 7.0))
        is_peak = 1.0 if ((6.0 <= frac_hour <= 9.0) or (16.5 <= frac_hour <= 19.5)) else 0.0

        std_fare = self.corridor_fares.get(corridor_id, 4000.0)
        fare_ratio = fare / max(std_fare, 1.0)

        if update_state:
            if card_id not in self.card_history:
                self.card_history[card_id] = deque(maxlen=64)
            self.card_history[card_id].append((ts_sec, lat, lon, reader_id, stop_id, tap_type))
            self.card_consec_tapin[card_id] = int(consec_tapin)

            if reader_id not in self.reader_history:
                self.reader_history[reader_id] = deque(maxlen=128)
            self.reader_history[reader_id].append((ts_sec, latency_ms, is_retry))

        return np.array(
            [
                delta_t,
                delta_d,
                vel_kmh,
                imp_flag,
                cnt_15s,
                cnt_30s,
                cnt_1m,
                cnt_5m,
                cnt_1h,
                cnt_24h,
                same_reader,
                same_stop,
                same_tap,
                inc_ratio_7d,
                consec_tapin,
                passback_5m,
                latency_ms,
                r_mean_5m,
                r_std_5m,
                r_zscore,
                r_retry_ratio,
                sync_delay,
                is_offline,
                hour_sin,
                hour_cos,
                day_sin,
                is_peak,
                fare_ratio,
            ],
            dtype=np.float64,
        )
