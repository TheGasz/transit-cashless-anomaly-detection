"""Tier 1: Detektor deterministik sub-milidetik untuk Double-Tap Glitch & Idempotency Check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features.velocity_features import FEATURE_COLUMNS


@dataclass
class GuardrailDecision:
    """Hasil evaluasi instan dari Tier 1 Deterministic Guardrail."""

    is_triggered: bool
    rule_name: str
    anomaly_category: str
    confidence_score: float
    recommended_action: str


class DeterministicRuleGuardrail:
    """Tier 1 Sub-Millisecond Guardrail untuk menangkap Idempotency Duplicate & Hard Double-Tap."""

    def __init__(
        self,
        double_tap_max_seconds: float = 8.0,
        hard_velocity_kmh: float = 180.0,
        min_distance_km: float = 3.0,
    ) -> None:
        self.double_tap_max_seconds = double_tap_max_seconds
        self.hard_velocity_kmh = hard_velocity_kmh
        self.min_distance_km = min_distance_km
        self.seen_tx_ids: Set[str] = set()
        self._feat_idx = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

    def reset_idempotency_cache(self) -> None:
        """Mengosongkan cache ID transaksi."""
        self.seen_tx_ids.clear()

    def evaluate_single(
        self,
        feature_vector: np.ndarray,
        tx_id: Optional[str] = None,
        record_tx_id: bool = True,
    ) -> GuardrailDecision:
        """Mengevaluasi satu vektor 28 fitur dalam waktu < 0.05 ms."""
        if tx_id is not None and tx_id in self.seen_tx_ids:
            return GuardrailDecision(
                is_triggered=True,
                rule_name="IDEMPOTENCY_DUPLICATE_TX_ID",
                anomaly_category="DOUBLE_TAP_GLITCH",
                confidence_score=0.995,
                recommended_action="AUTO_REFUND_DUPLICATE",
            )

        if tx_id is not None and record_tx_id:
            if len(self.seen_tx_ids) > 250_000:
                self.seen_tx_ids.clear()
            self.seen_tx_ids.add(tx_id)

        dt_sec = float(feature_vector[self._feat_idx["delta_t_card_sec"]])
        same_reader = float(feature_vector[self._feat_idx["is_same_reader"]])
        same_tap_type = float(feature_vector[self._feat_idx["is_same_tap_type"]])
        dist_km = float(feature_vector[self._feat_idx["delta_d_km"]])
        vel_kmh = float(feature_vector[self._feat_idx["velocity_kmh"]])

        # Hard Double-Tap Rule sesuai PRD Bagian 3.1 Tier 1:
        # delta_t_card <= 8 detik, is_same_reader = 1, dan tap_type identik
        if 0.0 <= dt_sec <= self.double_tap_max_seconds and same_reader > 0.5 and same_tap_type > 0.5:
            return GuardrailDecision(
                is_triggered=True,
                rule_name="HARD_DOUBLE_TAP_8S_SAME_READER",
                anomaly_category="DOUBLE_TAP_GLITCH",
                confidence_score=0.985,
                recommended_action="AUTO_REFUND_DUPLICATE",
            )

        # Hard Impossible Travel Rule
        if dist_km >= self.min_distance_km and vel_kmh >= self.hard_velocity_kmh:
            return GuardrailDecision(
                is_triggered=True,
                rule_name="HARD_IMPOSSIBLE_GEODESIC_VELOCITY",
                anomaly_category="IMPOSSIBLE_TRAVEL_CLONE",
                confidence_score=0.960,
                recommended_action="TEMP_FREEZE_TOKEN_AND_ALERT",
            )

        return GuardrailDecision(
            is_triggered=False,
            rule_name="NONE",
            anomaly_category="NORMAL",
            confidence_score=0.02,
            recommended_action="APPROVE_NORMAL",
        )

    def score_batch(self, x_features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Menghasilkan skor heuristik aturan statis [0, 1] untuk studi ablasi 5 model."""
        if isinstance(x_features, pd.DataFrame):
            mat = x_features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        else:
            mat = np.asarray(x_features, dtype=np.float64)

        dt = mat[:, self._feat_idx["delta_t_card_sec"]]
        dist = mat[:, self._feat_idx["delta_d_km"]]
        vel = mat[:, self._feat_idx["velocity_kmh"]]
        same_reader = mat[:, self._feat_idx["is_same_reader"]]
        same_tap = mat[:, self._feat_idx["is_same_tap_type"]]
        consec_in = mat[:, self._feat_idx["consecutive_tapin_count"]]
        sync_delay = mat[:, self._feat_idx["sync_delay_seconds"]]
        lat_ms = mat[:, self._feat_idx["reader_latency_ms"]]

        scores = np.full(len(mat), 0.03, dtype=np.float64)

        # Aturan 1: Double-tap
        cond_dt = (dt <= self.double_tap_max_seconds) & (same_reader > 0.5) & (same_tap > 0.5)
        scores = np.where(cond_dt, 0.92, scores)

        # Aturan 2: Kecepatan tinggi statis (hanya menangkap sebagian yang ekstrem)
        cond_vel = (dist > 6.0) & (vel > 220.0)
        scores = np.where(cond_vel, np.maximum(scores, 0.78), scores)

        # Aturan 3: Consecutive tap-in tinggi (>= 4)
        cond_evade = consec_in >= 4.0
        scores = np.where(cond_evade, np.maximum(scores, 0.68), scores)

        # Aturan 4: Sinkronisasi terlambat ekstrem (> 4 jam)
        cond_sync = sync_delay > 14400.0
        scores = np.where(cond_sync, np.maximum(scores, 0.72), scores)

        # Tambahkan sedikit gradasi dari latensi agar kurva PR dapat dihitung
        lat_bonus = np.clip((lat_ms - 250.0) / 10000.0, 0.0, 0.15)
        return np.clip(scores + lat_bonus, 0.0, 0.99)
