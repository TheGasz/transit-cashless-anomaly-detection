"""F-07: Auditor Rekonsiliasi Batch Settlement Harian menggunakan Statistik Robust MAD Z-Score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_robust_mad_zscore(values: np.ndarray, min_mad: float = 1.0) -> np.ndarray:
    """Menghitung Modified Z-Score berbasis Median & Median Absolute Deviation (MAD):
    Z_MAD = 0.6745 * |x - median(x)| / max(MAD, min_mad).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.empty(0, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    safe_mad = max(mad, min_mad)
    return 0.6745 * np.abs(arr - med) / safe_mad


@dataclass
class SettlementAuditSummary:
    """Ringkasan hasil audit rekonsiliasi settlement harian."""

    total_batches: int
    flagged_batches: int
    total_discrepancy_idr: float
    recall_on_injected_anomalies: float
    precision_mad_detector: float
    audit_dataframe: pd.DataFrame


class DailySettlementReconciliationAuditor:
    """Mengaudit selisih kas antara log perangkat TOB vs laporan settlement bank mitra (acquirer)."""

    def __init__(
        self,
        mad_zscore_threshold: float = 3.5,
        discrepancy_idr_threshold: float = 50000.0,
        missing_packet_ratio_threshold: float = 0.005,
        late_sync_minutes_threshold: float = 90.0,
    ) -> None:
        self.mad_zscore_threshold = mad_zscore_threshold
        self.discrepancy_idr_threshold = discrepancy_idr_threshold
        self.missing_packet_ratio_threshold = missing_packet_ratio_threshold
        self.late_sync_minutes_threshold = late_sync_minutes_threshold

    def audit_batches(
        self,
        batches_df: pd.DataFrame,
        save_csv_path: Optional[str | Path] = "reports/settlement_audit_table.csv",
    ) -> SettlementAuditSummary:
        """Menjalankan audit rekonsiliasi per batch harian dan menghasilkan tiket investigasi otomatis."""
        if batches_df is None or batches_df.empty:
            raise ValueError("DataFrame settlement batch kosong.")

        work = batches_df.copy()

        dev_idr = work["device_total_idr"].to_numpy(dtype=np.float64)
        acq_idr = work["acquirer_settled_idr"].to_numpy(dtype=np.float64)
        dev_cnt = work["device_tx_count"].to_numpy(dtype=np.float64)
        set_cnt = work["settled_tx_count"].to_numpy(dtype=np.float64)
        sync_min = (
            work["sync_delay_minutes"].to_numpy(dtype=np.float64)
            if "sync_delay_minutes" in work.columns
            else np.zeros(len(work), dtype=np.float64)
        )

        signed_diff_idr = acq_idr - dev_idr
        abs_diff_idr = np.abs(signed_diff_idr)
        missing_packets = np.abs(dev_cnt - set_cnt)
        missing_packet_ratio = missing_packets / np.maximum(dev_cnt, 1.0)

        # Hitung Robust MAD Z-Score per metrik
        z_idr = compute_robust_mad_zscore(abs_diff_idr, min_mad=100.0)
        z_pkt = compute_robust_mad_zscore(missing_packet_ratio, min_mad=0.0005)
        z_sync = compute_robust_mad_zscore(sync_min, min_mad=4.0)

        composite_mad_z = np.maximum.reduce([z_idr, z_pkt, z_sync])

        flagged_mask = (
            (composite_mad_z > self.mad_zscore_threshold)
            | (abs_diff_idr > self.discrepancy_idr_threshold)
            | (missing_packet_ratio > self.missing_packet_ratio_threshold)
        )

        work["signed_discrepancy_idr"] = np.round(signed_diff_idr, 2)
        work["abs_discrepancy_idr"] = np.round(abs_diff_idr, 2)
        work["missing_packet_count"] = missing_packets.astype(int)
        work["missing_packet_ratio"] = np.round(missing_packet_ratio, 5)
        work["mad_zscore_idr"] = np.round(z_idr, 3)
        work["mad_zscore_packet"] = np.round(z_pkt, 3)
        work["mad_zscore_sync"] = np.round(z_sync, 3)
        work["robust_mad_zscore"] = np.round(composite_mad_z, 3)
        work["is_flagged_discrepancy"] = flagged_mask.astype(int)

        statuses: List[str] = []
        actions: List[str] = []
        tickets: List[str] = []

        ticket_seq = 1
        for i, is_flag in enumerate(flagged_mask):
            if not is_flag:
                statuses.append("RECONCILED_OK")
                actions.append("AUTO_CLEAR_SETTLEMENT")
                tickets.append("NONE")
                continue

            t_id = f"RECON-TICKET-{ticket_seq:04d}"
            ticket_seq += 1
            tickets.append(t_id)

            if signed_diff_idr[i] < -self.discrepancy_idr_threshold and missing_packet_ratio[i] > self.missing_packet_ratio_threshold:
                statuses.append("CRITICAL_MISSING_OFFLINE_BATCH")
                actions.append("HOLD_BATCH_SETTLEMENT & TRIGGER_TOB_RESYNC")
            elif signed_diff_idr[i] > self.discrepancy_idr_threshold:
                statuses.append("CRITICAL_DUPLICATE_BATCH_UPLOAD")
                actions.append("HOLD_BATCH_SETTLEMENT & DEDUPLICATE_ACQUIRER_FILE")
            elif abs_diff_idr[i] > self.discrepancy_idr_threshold:
                statuses.append("HIGH_CASH_SETTLEMENT_MISMATCH")
                actions.append("HOLD_BATCH_SETTLEMENT & AUDIT_ACQUIRER_LOG")
            else:
                statuses.append("MEDIUM_LATE_STORE_AND_FORWARD_SYNC")
                actions.append("FLAG_TOB_TELEMETRY_MAINTENANCE")

        work["audit_status"] = statuses
        work["recommended_action"] = actions
        work["reconciliation_ticket_id"] = tickets

        if "is_settlement_anomaly" in work.columns:
            y_true = work["is_settlement_anomaly"].to_numpy(dtype=np.int64)
            y_pred = flagged_mask.astype(np.int64)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            total_anom = int(np.sum(y_true == 1))
            total_pred = int(np.sum(y_pred == 1))
            recall = float(tp / total_anom) if total_anom > 0 else 1.0
            precision = float(tp / total_pred) if total_pred > 0 else 1.0
        else:
            recall = 1.0
            precision = 1.0

        if save_csv_path is not None:
            out_path = Path(save_csv_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            work.to_csv(out_path, index=False)
            compat_path = out_path.parent / "settlement_audit.csv"
            work.to_csv(compat_path, index=False)

        return SettlementAuditSummary(
            total_batches=int(len(work)),
            flagged_batches=int(np.sum(flagged_mask)),
            total_discrepancy_idr=round(float(np.sum(abs_diff_idr[flagged_mask])), 2),
            recall_on_injected_anomalies=round(recall, 4),
            precision_mad_detector=round(precision, 4),
            audit_dataframe=work,
        )
