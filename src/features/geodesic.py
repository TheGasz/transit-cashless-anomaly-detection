"""Perhitungan jarak geodesik Haversine & kecepatan perpindahan implisit ter-vektorisasi NumPy."""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np

EARTH_RADIUS_KM: float = 6371.0088
ArrayOrFloat = Union[float, np.ndarray]


def haversine_distance_km(
    lat1: ArrayOrFloat,
    lon1: ArrayOrFloat,
    lat2: ArrayOrFloat,
    lon2: ArrayOrFloat,
) -> ArrayOrFloat:
    """Menghitung jarak lingkaran besar (great-circle distance) Haversine dalam kilometer.

    Args:
        lat1: Garis lintang titik awal dalam derajat desimal.
        lon1: Garis bujur titik awal dalam derajat desimal.
        lat2: Garis lintang titik akhir dalam derajat desimal.
        lon2: Garis bujur titik akhir dalam derajat desimal.

    Returns:
        Jarak geodesik dalam kilometer (float atau np.ndarray).
    """
    is_scalar = np.isscalar(lat1) and np.isscalar(lon1) and np.isscalar(lat2) and np.isscalar(lon2)
    lat1_arr = np.asarray(lat1, dtype=np.float64)
    lon1_arr = np.asarray(lon1, dtype=np.float64)
    lat2_arr = np.asarray(lat2, dtype=np.float64)
    lon2_arr = np.asarray(lon2, dtype=np.float64)

    valid_mask = ~(np.isnan(lat1_arr) | np.isnan(lon1_arr) | np.isnan(lat2_arr) | np.isnan(lon2_arr))

    phi1 = np.radians(np.where(valid_mask, lat1_arr, 0.0))
    phi2 = np.radians(np.where(valid_mask, lat2_arr, 0.0))
    dphi = np.radians(np.where(valid_mask, lat2_arr - lat1_arr, 0.0))
    dlambda = np.radians(np.where(valid_mask, lon2_arr - lon1_arr, 0.0))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist_km = np.where(valid_mask, EARTH_RADIUS_KM * c, 0.0)

    if is_scalar:
        return float(dist_km)
    return dist_km


def compute_velocity_kmh(
    distance_km: ArrayOrFloat,
    delta_t_sec: ArrayOrFloat,
    min_delta_t_sec: float = 1.0,
    max_cap_kmh: float = 5000.0,
) -> ArrayOrFloat:
    """Menghitung kecepatan perpindahan implisit dalam km/jam: v = d_km / (max(dt, 1.0) / 3600).

    Args:
        distance_km: Jarak Haversine antar lokasi tap (km).
        delta_t_sec: Selisih waktu sejak tap terakhir (detik).
        min_delta_t_sec: Batas bawah selisih waktu (default 1.0 detik sesuai PRD).
        max_cap_kmh: Batas atas numerik untuk mencegah nilai ekstrem tak hingga.

    Returns:
        Kecepatan implisit dalam km/jam.
    """
    is_scalar = np.isscalar(distance_km) and np.isscalar(delta_t_sec)
    dist_arr = np.asarray(distance_km, dtype=np.float64)
    dt_arr = np.asarray(delta_t_sec, dtype=np.float64)

    valid_mask = (~np.isnan(dist_arr)) & (~np.isnan(dt_arr)) & (dt_arr >= 0.0) & (dist_arr > 0.0)
    safe_dt_hours = np.maximum(np.where(valid_mask, dt_arr, min_delta_t_sec), min_delta_t_sec) / 3600.0
    vel_kmh = np.where(valid_mask, dist_arr / safe_dt_hours, 0.0)
    vel_kmh = np.clip(vel_kmh, 0.0, max_cap_kmh)

    if is_scalar:
        return float(vel_kmh)
    return vel_kmh


def flag_impossible_travel(
    distance_km: ArrayOrFloat,
    delta_t_sec: ArrayOrFloat,
    velocity_kmh: ArrayOrFloat | None = None,
    velocity_threshold_kmh: float = 120.0,
    min_distance_km: float = 2.5,
) -> ArrayOrFloat:
    """Menandai apakah perpindahan antar-tap melampaui batas fisika transportasi perkotaan.

    Sesuai PRD Bagian 1 & 3.1:
    - Kecepatan implisit v > 120 km/jam dengan jarak >= 2.5 km, atau
    - Jarak > 2.5 km dalam waktu < 60 detik.
    """
    is_scalar = np.isscalar(distance_km) and np.isscalar(delta_t_sec)
    dist_arr = np.asarray(distance_km, dtype=np.float64)
    dt_arr = np.asarray(delta_t_sec, dtype=np.float64)
    if velocity_kmh is None:
        vel_arr = np.asarray(compute_velocity_kmh(dist_arr, dt_arr), dtype=np.float64)
    else:
        vel_arr = np.asarray(velocity_kmh, dtype=np.float64)

    cond_vel = (dist_arr >= min_distance_km) & (vel_arr > velocity_threshold_kmh)
    cond_short = (dist_arr > min_distance_km) & (dt_arr >= 0.0) & (dt_arr < 60.0)
    flagged = (cond_vel | cond_short).astype(np.float64)

    if is_scalar:
        return float(flagged)
    return flagged
