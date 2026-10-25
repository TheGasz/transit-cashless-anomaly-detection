"""Skema kontrak event transaksi pembayaran nirsentuh (TapEvent) dan validasi DataFrame."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator


ANOMALY_CATEGORIES = (
    "NORMAL",
    "DOUBLE_TAP_GLITCH",
    "IMPOSSIBLE_TRAVEL_CLONE",
    "FARE_EVASION_INCOMPLETE_OUT",
    "SETTLEMENT_BATCH_ANOMALY",
    "NOVEL_ZERO_DAY_PATTERN",
)

REQUIRED_STREAM_COLUMNS = [
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
]


class TapEvent(BaseModel):
    """Representasi satu peristiwa penempelan kartu/QRIS Tap pada TOB atau Gate."""

    tx_id: str = Field(..., pattern=r"^TX-\d{8,12}$", description="ID unik transaksi tap")
    timestamp: str = Field(..., description="Waktu kejadian ISO-8601 (YYYY-MM-DDTHH:MM:SS)")
    card_pan_hash: str = Field(
        ...,
        min_length=12,
        max_length=64,
        description="Hash SHA-256 nomor kartu atau token QRIS",
    )
    user_segment: Literal["REGULER", "PELAJAR_LANSIA"] = Field(
        default="REGULER",
        description="Segmen tarif pengguna transportasi publik",
    )
    payment_channel: Literal["KMT", "E_MONEY_BANK", "QRIS_TAP"] = Field(
        ...,
        description="Kanal instrumen pembayaran nirsentuh",
    )
    tap_type: Literal["TAP_IN", "TAP_OUT"] = Field(
        ...,
        description="Jenis transaksi tap pada halte/bus",
    )
    corridor_id: str = Field(..., min_length=3, description="Kode koridor BRT/kereta (mis. COR-01)")
    stop_id: str = Field(..., min_length=3, description="Kode halte atau gerbang parkir")
    latitude: float = Field(..., ge=-11.0, le=6.0, description="Garis lintang GPS halte/TOB")
    longitude: float = Field(..., ge=95.0, le=141.0, description="Garis bujur GPS halte/TOB")
    reader_device_id: str = Field(..., min_length=3, description="ID perangkat TOB atau gate NFC")
    fare_amount: float = Field(..., ge=0.0, le=100000.0, description="Nominal tarif dipotong (IDR)")
    reader_latency_ms: float = Field(
        ...,
        ge=1.0,
        le=30000.0,
        description="Latensi pemrosesan NFC/QRIS Tap dalam milidetik",
    )
    is_offline_sync: bool = Field(
        default=False,
        description="Apakah transaksi berasal dari sinkronisasi store-and-forward offline",
    )
    server_received_timestamp: Optional[str] = Field(
        default=None,
        description="Waktu transaksi diterima server pusat (ISO-8601)",
    )
    is_anomaly: Optional[int] = Field(default=0, description="Label biner anomali (0=Normal, 1=Anomali)")
    anomaly_category: Optional[
        Literal[
            "NORMAL",
            "DOUBLE_TAP_GLITCH",
            "IMPOSSIBLE_TRAVEL_CLONE",
            "FARE_EVASION_INCOMPLETE_OUT",
            "SETTLEMENT_BATCH_ANOMALY",
            "NOVEL_ZERO_DAY_PATTERN",
        ]
    ] = Field(default="NORMAL", description="Subtipe patologi anomali")

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, value: str) -> str:
        """Memastikan format waktu valid ISO-8601."""
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Format timestamp tidak valid ISO-8601: {value}") from exc
        return value


def validate_transaction_dataframe(df: pd.DataFrame) -> bool:
    """Memvalidasi integritas kolom, nilai kosong, dan batas geografis Indonesia pada DataFrame."""
    if df is None or df.empty:
        raise ValueError("DataFrame transaksi kosong.")

    missing = [col for col in REQUIRED_STREAM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada DataFrame: {missing}")

    if df[REQUIRED_STREAM_COLUMNS].isnull().any().any():
        null_cols = df[REQUIRED_STREAM_COLUMNS].columns[df[REQUIRED_STREAM_COLUMNS].isnull().any()].tolist()
        raise ValueError(f"Ditemukan nilai null pada kolom wajib: {null_cols}")

    if not ((df["latitude"] >= -11.0) & (df["latitude"] <= 6.0)).all():
        raise ValueError("Terdapat koordinat latitude di luar batas geografis Indonesia (-11.0 s.d. 6.0).")

    if not ((df["longitude"] >= 95.0) & (df["longitude"] <= 141.0)).all():
        raise ValueError("Terdapat koordinat longitude di luar batas geografis Indonesia (95.0 s.d. 141.0).")

    if not (df["fare_amount"] >= 0.0).all():
        raise ValueError("Terdapat fare_amount bernilai negatif.")

    if not (df["reader_latency_ms"] >= 1.0).all():
        raise ValueError("Terdapat reader_latency_ms < 1.0 ms.")

    return True
