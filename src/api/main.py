"""Layanan REST API Produksi FastAPI AinoSentinel dengan Lifespan Model Loader (< 20 ms p50, < 40 ms p95)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, AsyncIterator, Dict
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas_api import (
    AnomalyScoreResponse,
    BatchAuditItemResult,
    BatchAuditRequest,
    BatchAuditResponse,
    FeatureContribution,
    TapTransactionEvent,
)
from src.features.velocity_features import FEATURE_COLUMNS, OnlineVelocityFeatureExtractor
from src.models.hybrid_ensemble import HybridStackingEnsemble
from src.reconciliation.settlement_auditor import DailySettlementReconciliationAuditor


class AinoSentinelRuntimeState:
    """Kontainer artefak model yang dimuat satu kali di dalam memori saat startup FastAPI."""

    def __init__(self) -> None:
        self.ensemble: HybridStackingEnsemble | None = None
        self.online_extractor: OnlineVelocityFeatureExtractor = OnlineVelocityFeatureExtractor()
        self.settlement_auditor: DailySettlementReconciliationAuditor = DailySettlementReconciliationAuditor()
        self.loaded_via_lifespan: bool = False

    def ensure_loaded(self) -> None:
        """Memuat model dari disk jika belum dimuat."""
        if self.ensemble is not None and self.ensemble.is_fitted:
            return

        model_path = Path("models/hybrid_lgbm.pkl")
        if not model_path.exists():
            from src.models.evaluate_models import main as eval_main
            eval_main()

        self.ensemble = HybridStackingEnsemble.load(model_path)


runtime_state = AinoSentinelRuntimeState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Memuat seluruh bobot PyTorch VAE, RobustScaler, IsolationForest, dan LightGBM sekali saat startup."""
    runtime_state.ensure_loaded()
    runtime_state.loaded_via_lifespan = True
    # Warm-up satu kali inferensi agar JIT/PyTorch dan LightGBM siap di bawah 5 ms
    dummy_vec = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float64)
    if runtime_state.ensemble is not None:
        _ = runtime_state.ensemble.predict_proba(dummy_vec)
    app.state.runtime = runtime_state
    yield
    runtime_state.online_extractor.reset()


app = FastAPI(
    title="AinoSentinel Real-Time Cashless Anomaly Detection & Settlement Audit API",
    description=(
        "Platform deteksi anomali waktu nyata (Hybrid 3-Tier: Deterministic Guardrail + PyTorch Residual VAE "
        "+ Isolation Forest & Calibrated LightGBM) dan audit rekonsiliasi settlement harian PT Aino Indonesia."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Memeriksa kesiapan model dan status lifespan."""
    runtime_state.ensure_loaded()
    assert runtime_state.ensemble is not None
    return {
        "status": "healthy",
        "model_loaded": runtime_state.ensemble.is_fitted,
        "loaded_via_lifespan": runtime_state.loaded_via_lifespan,
        "calibrated_threshold_tau": runtime_state.ensemble.optimal_threshold,
        "feature_count": len(FEATURE_COLUMNS),
    }


@app.post("/v1/transactions/score", response_model=AnomalyScoreResponse)
async def score_transaction(event: TapTransactionEvent) -> AnomalyScoreResponse:
    """Menilai satu transaksi tap kartu/QRIS secara real-time (< 20 ms p50, < 40 ms p95)."""
    t0 = time.perf_counter()
    runtime_state.ensure_loaded()
    ens = runtime_state.ensemble
    if ens is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    event_dict = event.model_dump()
    x_vec = runtime_state.online_extractor.extract_single(event_dict, update_state=True)
    x_2d = x_vec.reshape(1, -1)

    # Tier 1: Deterministic Guardrail Check
    guard_dec = ens.guardrail.evaluate_single(x_vec, tx_id=event.tx_id, record_tx_id=True)

    # Tier 2 & Tier 3: PyTorch VAE + Isolation Forest + Calibrated LightGBM
    x_stacked, vae_out, if_scores = ens.build_stacked_features(x_2d, return_details=True)
    prob_anom = float(ens.predict_proba_from_stacked(x_stacked, vae_out)[0])

    if guard_dec.is_triggered:
        prob_anom = max(prob_anom, guard_dec.confidence_score)

    vae_err = float(vae_out.total_anomaly_score[0])
    if_score = float(if_scores[0])
    tau = float(ens.optimal_threshold)

    category, severity, action = ens.classify_subtype_and_action(
        x_stacked_row=x_stacked[0],
        prob_anom=prob_anom,
        vae_error=vae_err,
        threshold=tau,
    )
    if guard_dec.is_triggered and prob_anom >= tau:
        category = guard_dec.anomaly_category
        action = guard_dec.recommended_action
        severity = "HIGH" if category == "DOUBLE_TAP_GLITCH" else "CRITICAL"

    is_anom = bool(prob_anom >= tau)

    # Penjelasan fitur lokal menggunakan TreeSHAP cepat (< 0.3 ms)
    top_feats_raw = ens.explain_single_shap(x_stacked[0], top_k=5)
    top_feats = [FeatureContribution(**item) for item in top_feats_raw]

    dt_sec = float(x_vec[FEATURE_COLUMNS.index("delta_t_card_sec")])
    vel_kmh = float(x_vec[FEATURE_COLUMNS.index("velocity_kmh")])
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return AnomalyScoreResponse(
        tx_id=event.tx_id,
        is_anomaly=is_anom,
        anomaly_probability=round(prob_anom, 4),
        calibrated_threshold_used=round(tau, 4),
        anomaly_category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        recommended_action=action,  # type: ignore[arg-type]
        vae_reconstruction_error=round(vae_err, 4),
        isolation_forest_score=round(if_score, 4),
        computed_velocity_kmh=round(vel_kmh, 2),
        seconds_since_last_tap=round(dt_sec, 2) if dt_sec < 86400.0 else None,
        top_contributing_features=top_feats,
        scoring_latency_ms=latency_ms,
    )


@app.post("/v1/settlement/audit", response_model=BatchAuditResponse)
async def audit_settlement_batches(req: BatchAuditRequest) -> BatchAuditResponse:
    """Mengaudit sekumpulan batch settlement harian per perangkat TOB & bank mitra dengan Robust MAD Z-Score."""
    rows = [b.model_dump() for b in req.batches]
    df_batches = pd.DataFrame(rows)

    summary = runtime_state.settlement_auditor.audit_batches(df_batches, save_csv_path=None)
    adf = summary.audit_dataframe

    results = []
    for _, row in adf.iterrows():
        results.append(
            BatchAuditItemResult(
                batch_id=str(row["batch_id"]),
                reader_device_id=str(row["reader_device_id"]),
                acquirer_bank=str(row["acquirer_bank"]),
                signed_discrepancy_idr=float(row["signed_discrepancy_idr"]),
                abs_discrepancy_idr=float(row["abs_discrepancy_idr"]),
                missing_packet_count=int(row["missing_packet_count"]),
                missing_packet_ratio=float(row["missing_packet_ratio"]),
                robust_mad_zscore=float(row["robust_mad_zscore"]),
                is_flagged_discrepancy=bool(row["is_flagged_discrepancy"]),
                audit_status=str(row["audit_status"]),
                recommended_action=str(row["recommended_action"]),
                reconciliation_ticket_id=str(row["reconciliation_ticket_id"]),
            )
        )

    return BatchAuditResponse(
        total_batches=summary.total_batches,
        flagged_batches=summary.flagged_batches,
        total_discrepancy_idr=summary.total_discrepancy_idr,
        results=results,
    )
