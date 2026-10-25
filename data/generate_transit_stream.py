"""Generator simulasi aliran transaksi multi-koridor (120.000 transaksi) & batch settlement harian."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.features.velocity_features import extract_velocity_features
from src.ingestion.schemas_event import validate_transaction_dataframe


CORRIDOR_SPECS: Dict[str, Dict[str, float | str]] = {
    "COR-01": {
        "name": "Koridor 1 TransJakarta (Blok M - Kota)",
        "standard_fare": 3500.0,
        "base_lat": -6.2442,
        "base_lon": 106.8006,
        "end_lat": -6.1376,
        "end_lon": 106.8133,
    },
    "COR-02": {
        "name": "Koridor 2 KRL Commuter (Bogor - Jakarta Kota)",
        "standard_fare": 6000.0,
        "base_lat": -6.5950,
        "base_lon": 106.7905,
        "end_lat": -6.1376,
        "end_lon": 106.8133,
    },
    "COR-03": {
        "name": "Koridor 3 Trans Banyumas QRIS Tap (Ajibarang - Purwokerto)",
        "standard_fare": 4000.0,
        "base_lat": -7.4167,
        "base_lon": 109.0667,
        "end_lat": -7.4244,
        "end_lon": 109.2396,
    },
    "COR-04": {
        "name": "Koridor 4 Trans Jogja (Adisutjipto - Malioboro)",
        "standard_fare": 3600.0,
        "base_lat": -7.7882,
        "base_lon": 110.4317,
        "end_lat": -7.7926,
        "end_lon": 110.3658,
    },
    "COR-05": {
        "name": "Koridor 5 LRT & Parkir Integrasi (Harjamukti - Dukuh Atas)",
        "standard_fare": 10000.0,
        "base_lat": -6.3735,
        "base_lon": 106.8962,
        "end_lat": -6.2008,
        "end_lon": 106.8228,
    },
}


def build_stops_and_readers(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Membangun topologi 45 halte GPS nyata (9 halte per koridor) dan 120 perangkat TOB/Gate."""
    rng = np.random.default_rng(seed)
    stops_rows = []
    corridor_ids = list(CORRIDOR_SPECS.keys())

    stop_counter = 1
    for cid in corridor_ids:
        spec = CORRIDOR_SPECS[cid]
        lats = np.linspace(float(spec["base_lat"]), float(spec["end_lat"]), 9)
        lons = np.linspace(float(spec["base_lon"]), float(spec["end_lon"]), 9)
        for idx in range(9):
            stop_id = f"STOP-{stop_counter:03d}"
            stops_rows.append(
                {
                    "stop_id": stop_id,
                    "corridor_id": cid,
                    "stop_sequence": idx + 1,
                    "stop_name": f"Halte {cid} #{idx + 1}",
                    "latitude": round(float(lats[idx] + rng.normal(0, 0.0008)), 6),
                    "longitude": round(float(lons[idx] + rng.normal(0, 0.0008)), 6),
                    "standard_fare": float(spec["standard_fare"]),
                }
            )
            stop_counter += 1

    stops_df = pd.DataFrame(stops_rows)

    readers_rows = []
    banks = ["MANDIRI", "BRI", "BCA", "BNI", "BSI_QRIS"]
    for r_idx in range(1, 121):
        cid = corridor_ids[(r_idx - 1) % len(corridor_ids)]
        corridor_stops = stops_df[stops_df["corridor_id"] == cid]["stop_id"].tolist()
        home_stop = corridor_stops[(r_idx - 1) % len(corridor_stops)]
        readers_rows.append(
            {
                "reader_device_id": f"TOB-{r_idx:03d}",
                "corridor_id": cid,
                "default_stop_id": home_stop,
                "acquirer_bank": banks[(r_idx - 1) % len(banks)],
            }
        )
    readers_df = pd.DataFrame(readers_rows)
    return stops_df, readers_df


def _sample_commuter_seconds_of_day(rng: np.random.Generator, size: int) -> np.ndarray:
    """Membangkitkan waktu kejadian dalam sehari (detik) dengan pola bimodal jam sibuk komuter."""
    comp = rng.choice([0, 1, 2], size=size, p=[0.42, 0.38, 0.20])
    hours = np.empty(size, dtype=np.float64)
    m0 = comp == 0
    m1 = comp == 1
    m2 = comp == 2
    hours[m0] = rng.normal(loc=7.4, scale=1.05, size=int(m0.sum()))
    hours[m1] = rng.normal(loc=17.6, scale=1.15, size=int(m1.sum()))
    hours[m2] = rng.uniform(low=5.2, high=22.4, size=int(m2.sum()))
    hours = np.clip(hours, 5.0, 23.2)
    return np.round(hours * 3600.0).astype(np.int64)


def generate_transit_dataset(
    num_transactions: int = 120000,
    num_cards: int = 15000,
    num_days: int = 14,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membangkitkan dataset 120.000 transaksi pembayaran nirsentuh beserta 5 subtipe anomali."""
    rng = np.random.default_rng(seed)
    stops_df, readers_df = build_stops_and_readers(seed=seed)

    # Buat 15.000 kartu unik dengan hash SHA-256 (panjang 24 hex karakter)
    card_hashes = [
        hashlib.sha256(f"AINO-CARD-{i:06d}-{seed}".encode("utf-8")).hexdigest()[:24].upper()
        for i in range(1, num_cards + 1)
    ]
    channels = np.array(["E_MONEY_BANK", "KMT", "QRIS_TAP"])
    segments = np.array(["REGULER", "PELAJAR_LANSIA"])

    card_channel = rng.choice(channels, size=num_cards, p=[0.52, 0.28, 0.20])
    card_segment = rng.choice(segments, size=num_cards, p=[0.82, 0.18])
    corridor_ids = list(CORRIDOR_SPECS.keys())
    card_home_corridor = rng.choice(corridor_ids, size=num_cards, p=[0.30, 0.25, 0.18, 0.15, 0.12])

    # Pemetaan halte dan reader per koridor
    corridor_stops: Dict[str, List[Dict]] = {}
    corridor_readers: Dict[str, List[str]] = {}
    for cid in corridor_ids:
        corridor_stops[cid] = stops_df[stops_df["corridor_id"] == cid].to_dict("records")
        corridor_readers[cid] = readers_df[readers_df["corridor_id"] == cid]["reader_device_id"].tolist()

    base_epoch = int(pd.Timestamp("2026-10-01T00:00:00").value // 1_000_000_000)

    # Target jumlah anomali (~1.850 transaksi = ~1.54%)
    n_double_tap = 560
    n_impossible = 460
    n_fare_evasion_events = 480  # sekitar 120 rangkaian x 4 tap_in beruntun
    n_settlement_events = 260
    n_novel_zero_day = 90
    total_anomaly_rows = (
        n_double_tap + n_impossible + n_fare_evasion_events + n_settlement_events + n_novel_zero_day
    )

    normal_target = num_transactions - total_anomaly_rows
    num_normal_trips = normal_target // 2

    records: List[Dict] = []

    # -------------------------------------------------------------------------
    # 1. BANGKITKAN PERJALANAN NORMAL (BERPASANGAN TAP_IN -> TAP_OUT)
    # -------------------------------------------------------------------------
    trip_card_idxs = rng.integers(0, num_cards, size=num_normal_trips)
    trip_days = rng.integers(0, num_days, size=num_normal_trips)
    trip_sod = _sample_commuter_seconds_of_day(rng, size=num_normal_trips)
    trip_start_epochs = base_epoch + trip_days * 86400 + trip_sod

    for i in range(num_normal_trips):
        c_idx = int(trip_card_idxs[i])
        c_hash = card_hashes[c_idx]
        chan = str(card_channel[c_idx])
        seg = str(card_segment[c_idx])
        cid = str(card_home_corridor[c_idx])

        c_stops = corridor_stops[cid]
        c_readers = corridor_readers[cid]

        s1_idx = int(rng.integers(0, len(c_stops) - 1))
        hop_count = int(rng.integers(1, len(c_stops) - s1_idx))
        s2_idx = s1_idx + hop_count

        stop_in = c_stops[s1_idx]
        stop_out = c_stops[s2_idx]
        reader_in = c_readers[(s1_idx * 3 + i) % len(c_readers)]
        reader_out = c_readers[(s2_idx * 3 + i + 1) % len(c_readers)]

        std_fare = float(stop_in["standard_fare"])
        fare = std_fare if seg == "REGULER" else round(std_fare * 0.85, 0)

        t_in = int(trip_start_epochs[i])
        # Waktu tempuh realistis: 240 - 360 detik per halte (~22 - 35 km/jam)
        travel_sec = int(hop_count * rng.integers(240, 360) + rng.integers(30, 120))
        t_out = t_in + travel_sec

        lat_in_ms = round(float(np.clip(rng.lognormal(mean=4.45, sigma=0.25), 35.0, 240.0)), 2)
        lat_out_ms = round(float(np.clip(rng.lognormal(mean=4.45, sigma=0.25), 35.0, 240.0)), 2)

        records.append(
            {
                "_epoch": t_in,
                "card_pan_hash": c_hash,
                "user_segment": seg,
                "payment_channel": chan,
                "tap_type": "TAP_IN",
                "corridor_id": cid,
                "stop_id": stop_in["stop_id"],
                "latitude": float(stop_in["latitude"]),
                "longitude": float(stop_in["longitude"]),
                "reader_device_id": reader_in,
                "fare_amount": fare,
                "reader_latency_ms": lat_in_ms,
                "is_offline_sync": False,
                "server_received_epoch": t_in + 1,
                "is_anomaly": 0,
                "anomaly_category": "NORMAL",
            }
        )
        records.append(
            {
                "_epoch": t_out,
                "card_pan_hash": c_hash,
                "user_segment": seg,
                "payment_channel": chan,
                "tap_type": "TAP_OUT",
                "corridor_id": cid,
                "stop_id": stop_out["stop_id"],
                "latitude": float(stop_out["latitude"]),
                "longitude": float(stop_out["longitude"]),
                "reader_device_id": reader_out,
                "fare_amount": fare,
                "reader_latency_ms": lat_out_ms,
                "is_offline_sync": False,
                "server_received_epoch": t_out + 1,
                "is_anomaly": 0,
                "anomaly_category": "NORMAL",
            }
        )

    # -------------------------------------------------------------------------
    # 2. INJEKSI ANOMALI 1: DOUBLE_TAP_GLITCH (1 - 7 detik pada reader yang sama)
    # -------------------------------------------------------------------------
    anchor_indices = rng.choice(len(records), size=n_double_tap, replace=False)
    for idx in anchor_indices:
        base_rec = records[int(idx)]
        dt_glitch = int(rng.integers(1, 8))  # 1 s.d. 7 detik (<= 8 detik)
        glitch_epoch = int(base_rec["_epoch"]) + dt_glitch
        # Latensi perangkat melonjak saat terjadi retry glitch
        glitch_latency = round(float(rng.uniform(480.0, 2400.0)), 2)
        records.append(
            {
                "_epoch": glitch_epoch,
                "card_pan_hash": base_rec["card_pan_hash"],
                "user_segment": base_rec["user_segment"],
                "payment_channel": base_rec["payment_channel"],
                "tap_type": base_rec["tap_type"],
                "corridor_id": base_rec["corridor_id"],
                "stop_id": base_rec["stop_id"],
                "latitude": base_rec["latitude"],
                "longitude": base_rec["longitude"],
                "reader_device_id": base_rec["reader_device_id"],
                "fare_amount": base_rec["fare_amount"],
                "reader_latency_ms": glitch_latency,
                "is_offline_sync": False,
                "server_received_epoch": glitch_epoch + 2,
                "is_anomaly": 1,
                "anomaly_category": "DOUBLE_TAP_GLITCH",
            }
        )

    # -------------------------------------------------------------------------
    # 3. INJEKSI ANOMALI 2: IMPOSSIBLE_TRAVEL_CLONE (v > 150 km/jam, d > 8 km)
    # -------------------------------------------------------------------------
    clone_anchors = rng.choice(len(records) - n_double_tap, size=n_impossible, replace=False)
    for idx in clone_anchors:
        base_rec = records[int(idx)]
        base_cid = base_rec["corridor_id"]
        # Pilih koridor lain atau ujung halte yang berjarak > 10 km dalam selisih 12 - 55 detik
        other_cid = "COR-02" if base_cid == "COR-01" else "COR-01"
        far_stop = corridor_stops[other_cid][-1]
        far_reader = corridor_readers[other_cid][-1]
        dt_clone = int(rng.integers(12, 55))
        clone_epoch = int(base_rec["_epoch"]) + dt_clone
        records.append(
            {
                "_epoch": clone_epoch,
                "card_pan_hash": base_rec["card_pan_hash"],
                "user_segment": base_rec["user_segment"],
                "payment_channel": base_rec["payment_channel"],
                "tap_type": "TAP_IN",
                "corridor_id": other_cid,
                "stop_id": far_stop["stop_id"],
                "latitude": float(far_stop["latitude"]),
                "longitude": float(far_stop["longitude"]),
                "reader_device_id": far_reader,
                "fare_amount": float(far_stop["standard_fare"]),
                "reader_latency_ms": round(float(rng.uniform(180.0, 620.0)), 2),
                "is_offline_sync": False,
                "server_received_epoch": clone_epoch + 1,
                "is_anomaly": 1,
                "anomaly_category": "IMPOSSIBLE_TRAVEL_CLONE",
            }
        )

    # -------------------------------------------------------------------------
    # 4. INJEKSI ANOMALI 3: FARE_EVASION_INCOMPLETE_OUT (Pass-back & Repeated TAP_IN)
    # -------------------------------------------------------------------------
    n_chains = n_fare_evasion_events // 4
    evader_cards = [
        hashlib.sha256(f"AINO-EVADER-{k:04d}-{seed}".encode("utf-8")).hexdigest()[:24].upper()
        for k in range(n_chains)
    ]
    for k in range(n_chains):
        day_k = int(k % num_days)
        base_t = base_epoch + day_k * 86400 + int(rng.integers(25200, 68400))
        cid = corridor_ids[k % len(corridor_ids)]
        st = corridor_stops[cid][0]
        rd = corridor_readers[cid][0]
        std_f = float(st["standard_fare"])
        for step in range(4):
            ev_t = base_t + step * int(rng.integers(22, 48))
            # Hanya tap-in beruntun di halte yang sama (pass-back / incomplete tap-out) dengan tarif terpotong parsial
            records.append(
                {
                    "_epoch": ev_t,
                    "card_pan_hash": evader_cards[k],
                    "user_segment": "REGULER",
                    "payment_channel": "KMT",
                    "tap_type": "TAP_IN",
                    "corridor_id": cid,
                    "stop_id": st["stop_id"],
                    "latitude": float(st["latitude"]),
                    "longitude": float(st["longitude"]),
                    "reader_device_id": rd,
                    "fare_amount": round(std_f * 0.25, 0) if step > 0 else std_f,
                    "reader_latency_ms": round(float(rng.uniform(110.0, 340.0)), 2),
                    "is_offline_sync": False,
                    "server_received_epoch": ev_t + 1,
                    "is_anomaly": 1,
                    "anomaly_category": "FARE_EVASION_INCOMPLETE_OUT",
                }
            )

    # -------------------------------------------------------------------------
    # 5. INJEKSI ANOMALI 4: SETTLEMENT_BATCH_ANOMALY (Late Offline Sync & Fare Spike)
    # -------------------------------------------------------------------------
    for k in range(n_settlement_events):
        day_k = int(k % num_days)
        ev_t = base_epoch + day_k * 86400 + int(rng.integers(72000, 82800))
        cid = corridor_ids[k % len(corridor_ids)]
        st = corridor_stops[cid][k % len(corridor_stops[cid])]
        rd = corridor_readers[cid][k % len(corridor_readers[cid])]
        std_f = float(st["standard_fare"])
        delay_sec = int(rng.integers(9000, 28800))  # 2.5 - 8 jam terlambat sinkronisasi
        records.append(
            {
                "_epoch": ev_t,
                "card_pan_hash": card_hashes[(k * 37) % num_cards],
                "user_segment": "REGULER",
                "payment_channel": "QRIS_TAP",
                "tap_type": "TAP_IN",
                "corridor_id": cid,
                "stop_id": st["stop_id"],
                "latitude": float(st["latitude"]),
                "longitude": float(st["longitude"]),
                "reader_device_id": rd,
                "fare_amount": round(std_f * float(rng.uniform(2.8, 4.5)), 0),
                "reader_latency_ms": round(float(rng.uniform(1400.0, 5800.0)), 2),
                "is_offline_sync": True,
                "server_received_epoch": ev_t + delay_sec,
                "is_anomaly": 1,
                "anomaly_category": "SETTLEMENT_BATCH_ANOMALY",
            }
        )

    # -------------------------------------------------------------------------
    # 6. INJEKSI ANOMALI 5: NOVEL_ZERO_DAY_PATTERN (Untuk Evaluasi VAE vs Rule Statis)
    # -------------------------------------------------------------------------
    for k in range(n_novel_zero_day):
        # Sebagian besar di hari 11-14 (Val & Test) untuk menguji deteksi zero-day
        day_k = int(10 + (k % 4))
        ev_t = base_epoch + day_k * 86400 + int(rng.integers(18000, 21000))  # subuh/dini hari
        cid = corridor_ids[k % len(corridor_ids)]
        st = corridor_stops[cid][(k * 2) % len(corridor_stops[cid])]
        rd = corridor_readers[cid][(k * 2) % len(corridor_readers[cid])]
        std_f = float(st["standard_fare"])
        records.append(
            {
                "_epoch": ev_t,
                "card_pan_hash": card_hashes[(k * 97) % num_cards],
                "user_segment": "PELAJAR_LANSIA",
                "payment_channel": "QRIS_TAP",
                "tap_type": "TAP_OUT",
                "corridor_id": cid,
                "stop_id": st["stop_id"],
                "latitude": float(st["latitude"]),
                "longitude": float(st["longitude"]),
                "reader_device_id": rd,
                "fare_amount": round(std_f * float(rng.uniform(3.2, 5.2)), 0),
                "reader_latency_ms": round(float(rng.uniform(2200.0, 6800.0)), 2),
                "is_offline_sync": True,
                "server_received_epoch": ev_t + int(rng.integers(4500, 12000)),
                "is_anomaly": 1,
                "anomaly_category": "NOVEL_ZERO_DAY_PATTERN",
            }
        )

    # Jika jumlah baris kurang/lebih sedikit dari num_transactions, sesuaikan secara eksak
    df = pd.DataFrame(records)
    df = df.sort_values("_epoch", kind="mergesort").reset_index(drop=True)
    if len(df) > num_transactions:
        df = df.iloc[:num_transactions].reset_index(drop=True)

    # Buat ID unik berurutan kronologis dan format ISO-8601
    df["tx_id"] = [f"TX-{i:08d}" for i in range(1, len(df) + 1)]
    df["timestamp"] = pd.to_datetime(df["_epoch"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["server_received_timestamp"] = pd.to_datetime(df["server_received_epoch"], unit="s").dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    df = df.drop(columns=["_epoch", "server_received_epoch"])

    # Susun urutan kolom utama sebelum ekstraksi 28 fitur
    ordered_base_cols = [
        "tx_id",
        "timestamp",
        "card_pan_hash",
        "user_segment",
        "payment_channel",
        "tap_type",
        "corridor_id",
        "stop_id",
        "latitude",
        "longitude",
        "reader_device_id",
        "fare_amount",
        "reader_latency_ms",
        "is_offline_sync",
        "server_received_timestamp",
        "is_anomaly",
        "anomaly_category",
    ]
    df = df[ordered_base_cols]
    validate_transaction_dataframe(df)

    # Hitung 28 fitur kausal sliding-window
    df = extract_velocity_features(df, inplace=True)

    # Bangkitkan tabel batch settlement harian (14 hari x 120 perangkat TOB = 1.680 batch)
    settlement_df = generate_daily_settlement_batches(df, readers_df, seed=seed)

    return df, settlement_df, stops_df


def generate_daily_settlement_batches(
    tx_df: pd.DataFrame,
    readers_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Membangkitkan data rekonsiliasi settlement harian per TOB & bank mitra dengan injeksi anomali."""
    rng = np.random.default_rng(seed + 101)
    work = tx_df.copy()
    work["settlement_date"] = work["timestamp"].str.slice(0, 10)

    grouped = (
        work.groupby(["settlement_date", "reader_device_id", "corridor_id"], as_index=False)
        .agg(
            device_tx_count=("tx_id", "count"),
            device_total_idr=("fare_amount", "sum"),
            offline_tx_count=("is_offline_sync", "sum"),
            mean_latency_ms=("reader_latency_ms", "mean"),
        )
        .sort_values(["settlement_date", "reader_device_id"])
        .reset_index(drop=True)
    )

    reader_bank_map = dict(zip(readers_df["reader_device_id"], readers_df["acquirer_bank"]))
    grouped["acquirer_bank"] = grouped["reader_device_id"].map(reader_bank_map).fillna("MANDIRI")

    n_batches = len(grouped)
    grouped["batch_id"] = [f"BATCH-{i:05d}" for i in range(1, n_batches + 1)]

    # Secara normal, selisih pencairan bank = 0 atau pembulatan sangat kecil (<= Rp 500)
    small_noise_idr = rng.choice([0.0, 0.0, 0.0, 100.0, -100.0], size=n_batches)
    grouped["settled_tx_count"] = grouped["device_tx_count"]
    grouped["acquirer_settled_idr"] = np.maximum(0.0, grouped["device_total_idr"] + small_noise_idr)
    grouped["sync_delay_minutes"] = np.round(rng.uniform(2.0, 18.0, size=n_batches), 1)
    grouped["is_settlement_anomaly"] = 0
    grouped["discrepancy_type"] = "RECONCILED_NORMAL"

    # Injeksi 45 batch anomali (selisih > Rp50.000 atau missing packet > 0.5% atau late offline batch)
    anom_indices = rng.choice(n_batches, size=min(45, n_batches // 10), replace=False)
    for k, b_idx in enumerate(anom_indices):
        idx = int(b_idx)
        dev_cnt = int(grouped.loc[idx, "device_tx_count"])
        dev_idr = float(grouped.loc[idx, "device_total_idr"])

        if k % 3 == 0:
            # Missing offline packet (> 0.5% paket hilang & selisih > Rp50.000)
            missing_cnt = max(int(np.ceil(dev_cnt * 0.08)), 15)
            settled_cnt = max(0, dev_cnt - missing_cnt)
            discrepancy_idr = max(missing_cnt * 4500.0, 85000.0)
            grouped.loc[idx, "settled_tx_count"] = settled_cnt
            grouped.loc[idx, "acquirer_settled_idr"] = max(0.0, dev_idr - discrepancy_idr)
            grouped.loc[idx, "sync_delay_minutes"] = round(float(rng.uniform(180.0, 480.0)), 1)
            grouped.loc[idx, "is_settlement_anomaly"] = 1
            grouped.loc[idx, "discrepancy_type"] = "MISSING_OFFLINE_PACKET"
        elif k % 3 == 1:
            # Duplicate batch upload (pencairan ganda / selisih positif besar > Rp50.000)
            dup_cnt = max(int(np.ceil(dev_cnt * 0.07)), 14)
            dup_idr = max(dup_cnt * 5000.0, 95000.0)
            grouped.loc[idx, "settled_tx_count"] = dev_cnt + dup_cnt
            grouped.loc[idx, "acquirer_settled_idr"] = dev_idr + dup_idr
            grouped.loc[idx, "sync_delay_minutes"] = round(float(rng.uniform(95.0, 260.0)), 1)
            grouped.loc[idx, "is_settlement_anomaly"] = 1
            grouped.loc[idx, "discrepancy_type"] = "DUPLICATE_BATCH_UPLOAD"
        else:
            # Acquirer under-settlement discrepancy (> Rp50.000) + missing packet
            missing_cnt = max(int(np.ceil(dev_cnt * 0.06)), 12)
            shortfall_idr = float(rng.integers(75000, 240000))
            grouped.loc[idx, "settled_tx_count"] = max(0, dev_cnt - missing_cnt)
            grouped.loc[idx, "acquirer_settled_idr"] = max(0.0, dev_idr - shortfall_idr)
            grouped.loc[idx, "sync_delay_minutes"] = round(float(rng.uniform(140.0, 420.0)), 1)
            grouped.loc[idx, "is_settlement_anomaly"] = 1
            grouped.loc[idx, "discrepancy_type"] = "ACQUIRER_CASH_SHORTFALL"

    return grouped


def main() -> None:
    """Menjalankan pembangkitan data transaksi dan menyimpan ke direktori data/."""
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[AinoSentinel] Membangkitkan 120.000 transaksi multi-koridor & 28 fitur spasio-temporal...")
    tx_df, settlement_df, stops_df = generate_transit_dataset(
        num_transactions=120000,
        num_cards=15000,
        num_days=14,
        seed=42,
    )

    parquet_raw_path = raw_dir / "transactions_stream.parquet"
    parquet_compat_path = root / "data" / "transactions_stream.parquet"
    settlement_path = raw_dir / "daily_settlement_batches.csv"
    stops_path = raw_dir / "stops_metadata.csv"

    tx_df.to_parquet(parquet_raw_path, index=False)
    tx_df.to_parquet(parquet_compat_path, index=False)
    settlement_df.to_csv(settlement_path, index=False)
    stops_df.to_csv(stops_path, index=False)

    print(f"[AinoSentinel] Tersimpan: {parquet_raw_path} ({len(tx_df):,} baris, {tx_df.shape[1]} kolom)")
    print(f"[AinoSentinel] Tersimpan: {settlement_path} ({len(settlement_df):,} batch harian)")
    print("\nDistribusi Kategori Transaksi:")
    dist = tx_df["anomaly_category"].value_counts()
    for cat, cnt in dist.items():
        pct = 100.0 * cnt / len(tx_df)
        print(f"  - {cat:<30}: {cnt:>7,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
