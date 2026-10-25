"""Kontrak skema data Pydantic v2 untuk FastAPI (/v1/transactions/score & /v1/settlement/audit)."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TapTransactionEvent(BaseModel):
    """Skema input transaksi tap kartu/QRIS sesuai PRD Bagian 7."""

    tx_id: str = Field(..., pattern=r"^TX-\d{8,12}$", description="ID unik transaksi tap")
    timestamp: str = Field(..., description="Waktu kejadian ISO-8601 (YYYY-MM-DDTHH:MM:SS)")
    card_pan_hash: str = Field(
        ...,
        min_length=12,
        max_length=64,
        description="Hash SHA-256 nomor kartu/token QRIS",
    )
    user_segment: Literal["REGULER", "PELAJAR_LANSIA"] = Field(
        default="REGULER",
        description="Segmen tarif pengguna",
    )
    payment_channel: Literal["KMT", "E_MONEY_BANK", "QRIS_TAP"]
    tap_type: Literal["TAP_IN", "TAP_OUT"]
    corridor_id: str = Field(..., description="Kode koridor BRT/kereta (mis. COR-01)")
    stop_id: str = Field(..., description="Kode halte atau gerbang parkir")
    latitude: float = Field(..., ge=-11.0, le=6.0)
    longitude: float = Field(..., ge=95.0, le=141.0)
    reader_device_id: str = Field(..., description="ID perangkat TOB atau gate NFC")
    fare_amount: float = Field(..., ge=0.0, le=100000.0)
    reader_latency_ms: float = Field(..., ge=1.0, le=30000.0)
    is_offline_sync: bool = Field(default=False)


# Alias untuk kompatibilitas penamaan di struktur direktori PRD Bagian 5.2
TapScoreRequest = TapTransactionEvent


class FeatureContribution(BaseModel):
    """Kontribusi fitur lokal dari TreeSHAP atau Z-Score."""

    feature_name: str
    feature_value: float
    shap_or_z_impact: float


class AnomalyScoreResponse(BaseModel):
    """Respons penilaian transaksi real-time sesuai PRD Bagian 7."""

    tx_id: str
    is_anomaly: bool
    anomaly_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_threshold_used: float = Field(..., ge=0.0, le=1.0)
    anomaly_category: Literal[
        "NORMAL",
        "DOUBLE_TAP_GLITCH",
        "IMPOSSIBLE_TRAVEL_CLONE",
        "FARE_EVASION_INCOMPLETE_OUT",
        "SETTLEMENT_BATCH_ANOMALY",
        "NOVEL_ZERO_DAY_PATTERN",
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: Literal[
        "APPROVE_NORMAL",
        "AUTO_REFUND_DUPLICATE",
        "TEMP_FREEZE_TOKEN_AND_ALERT",
        "APPLY_MAX_CORRIDOR_PENALTY_FARE",
        "HOLD_BATCH_SETTLEMENT",
    ]
    vae_reconstruction_error: float
    isolation_forest_score: float
    computed_velocity_kmh: float
    seconds_since_last_tap: Optional[float] = None
    top_contributing_features: List[FeatureContribution]
    scoring_latency_ms: float


class SettlementBatchRecord(BaseModel):
    """Satu rekaman batch settlement harian per perangkat TOB."""

    batch_id: str = Field(..., min_length=3)
    settlement_date: str = Field(..., description="Tanggal settlement YYYY-MM-DD")
    reader_device_id: str = Field(..., min_length=3)
    corridor_id: str = Field(default="COR-01")
    acquirer_bank: Literal["MANDIRI", "BRI", "BCA", "BNI", "BSI_QRIS"] = Field(default="MANDIRI")
    device_tx_count: int = Field(..., ge=0)
    settled_tx_count: int = Field(..., ge=0)
    device_total_idr: float = Field(..., ge=0.0)
    acquirer_settled_idr: float = Field(..., ge=0.0)
    sync_delay_minutes: float = Field(default=5.0, ge=0.0)


class BatchAuditRequest(BaseModel):
    """Permintaan audit rekonsiliasi batch settlement harian."""

    batches: List[SettlementBatchRecord] = Field(..., min_length=1)


# Alias untuk kompatibilitas penamaan di PRD Bagian 5.2
BatchAudit = BatchAuditRequest


class BatchAuditItemResult(BaseModel):
    """Hasil audit per satu batch settlement."""

    batch_id: str
    reader_device_id: str
    acquirer_bank: str
    signed_discrepancy_idr: float
    abs_discrepancy_idr: float
    missing_packet_count: int
    missing_packet_ratio: float
    robust_mad_zscore: float
    is_flagged_discrepancy: bool
    audit_status: str
    recommended_action: str
    reconciliation_ticket_id: str


class BatchAuditResponse(BaseModel):
    """Respons endpoint POST /v1/settlement/audit."""

    total_batches: int
    flagged_batches: int
    total_discrepancy_idr: float
    results: List[BatchAuditItemResult]
