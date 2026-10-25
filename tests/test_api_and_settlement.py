"""Integration tests untuk audit rekonsiliasi settlement harian & SLA endpoint FastAPI (DoD-3)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.reconciliation.settlement_auditor import DailySettlementReconciliationAuditor


def test_daily_settlement_auditor_100pct_recall() -> None:
    """Memastikan 100% batch settlement dengan selisih > Rp50.000 atau missing packet > 0.5% terdeteksi oleh MAD Z-Score."""
    batches_path = Path("data/raw/daily_settlement_batches.csv")
    if not batches_path.exists():
        from data.generate_transit_stream import main as gen_main
        gen_main()

    df_batches = pd.read_csv(batches_path)
    auditor = DailySettlementReconciliationAuditor(
        mad_zscore_threshold=3.5,
        discrepancy_idr_threshold=50000.0,
        missing_packet_ratio_threshold=0.005,
    )
    summary = auditor.audit_batches(df_batches, save_csv_path="reports/settlement_audit_table.csv")

    assert Path("reports/settlement_audit_table.csv").exists()
    assert Path("reports/settlement_audit.csv").exists()
    assert summary.total_batches == len(df_batches)
    assert summary.flagged_batches >= 40
    assert summary.recall_on_injected_anomalies == pytest.approx(1.0, abs=1e-6)


def test_fastapi_lifespan_and_transaction_scoring_sla() -> None:
    """Memverifikasi pemuatan model sekali di lifespan FastAPI, kontrak Pydantic v2, dan latency p95 < 40 ms."""
    with TestClient(app) as client:
        # 1. Cek health dan lifespan loader
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["loaded_via_lifespan"] is True

        # 2. Transaksi 1: Tap-In Normal di Halte Blok M
        tx1 = {
            "tx_id": "TX-90000001",
            "timestamp": "2026-10-14T08:00:00",
            "card_pan_hash": "ABCDEF1234567890ABCDEF12",
            "user_segment": "REGULER",
            "payment_channel": "E_MONEY_BANK",
            "tap_type": "TAP_IN",
            "corridor_id": "COR-01",
            "stop_id": "STOP-001",
            "latitude": -6.2442,
            "longitude": 106.8006,
            "reader_device_id": "TOB-001",
            "fare_amount": 3500.0,
            "reader_latency_ms": 85.0,
            "is_offline_sync": False,
        }
        r1 = client.post("/v1/transactions/score", json=tx1)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["tx_id"] == "TX-90000001"
        assert len(d1["top_contributing_features"]) == 5

        # 3. Transaksi 2: Double-Tap Glitch (3 detik kemudian pada reader & halte yang sama)
        tx2 = {
            **tx1,
            "tx_id": "TX-90000002",
            "timestamp": "2026-10-14T08:00:03",
            "reader_latency_ms": 1150.0,
        }
        r2 = client.post("/v1/transactions/score", json=tx2)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["is_anomaly"] is True
        assert d2["anomaly_category"] == "DOUBLE_TAP_GLITCH"
        assert d2["recommended_action"] == "AUTO_REFUND_DUPLICATE"
        assert d2["seconds_since_last_tap"] == pytest.approx(3.0)

        # 4. Transaksi 3: Impossible Travel Clone (20 detik kemudian di Bogor, jarak > 38 km)
        tx3 = {
            **tx1,
            "tx_id": "TX-90000003",
            "timestamp": "2026-10-14T08:00:23",
            "corridor_id": "COR-02",
            "stop_id": "STOP-010",
            "latitude": -6.5950,
            "longitude": 106.7905,
            "reader_device_id": "TOB-012",
            "fare_amount": 6000.0,
            "reader_latency_ms": 210.0,
        }
        r3 = client.post("/v1/transactions/score", json=tx3)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["is_anomaly"] is True
        assert d3["anomaly_category"] == "IMPOSSIBLE_TRAVEL_CLONE"
        assert d3["recommended_action"] == "TEMP_FREEZE_TOKEN_AND_ALERT"
        assert d3["computed_velocity_kmh"] > 120.0

        # 5. Benchmark Latency SLA (50 transaksi berurutan -> p50 < 20 ms, p95 < 40 ms)
        latencies = []
        for i in range(50):
            tx_bench = {
                **tx1,
                "tx_id": f"TX-9100{i:04d}",
                "timestamp": f"2026-10-14T09:{(i // 60):02d}:{(i % 60):02d}",
                "card_pan_hash": f"BENCHCARD{i % 15:015d}",
            }
            resp = client.post("/v1/transactions/score", json=tx_bench)
            assert resp.status_code == 200
            latencies.append(float(resp.json()["scoring_latency_ms"]))

        p50_ms = float(np.percentile(latencies, 50))
        p95_ms = float(np.percentile(latencies, 95))
        assert p50_ms < 20.0, f"Scoring latency p50 melebihi SLA 20 ms: {p50_ms:.2f} ms"
        assert p95_ms < 40.0, f"Scoring latency p95 melebihi SLA 40 ms: {p95_ms:.2f} ms"


def test_fastapi_settlement_audit_endpoint() -> None:
    """Memverifikasi endpoint POST /v1/settlement/audit mendeteksi batch anomali secara akurat."""
    payload = {
        "batches": [
            {
                "batch_id": "BATCH-00001",
                "settlement_date": "2026-10-14",
                "reader_device_id": "TOB-001",
                "corridor_id": "COR-01",
                "acquirer_bank": "MANDIRI",
                "device_tx_count": 800,
                "settled_tx_count": 800,
                "device_total_idr": 2800000.0,
                "acquirer_settled_idr": 2800000.0,
                "sync_delay_minutes": 4.5,
            },
            {
                "batch_id": "BATCH-00002",
                "settlement_date": "2026-10-14",
                "reader_device_id": "TOB-002",
                "corridor_id": "COR-01",
                "acquirer_bank": "BRI",
                "device_tx_count": 820,
                "settled_tx_count": 820,
                "device_total_idr": 2870000.0,
                "acquirer_settled_idr": 2870000.0,
                "sync_delay_minutes": 5.2,
            },
            {
                "batch_id": "BATCH-00003",
                "settlement_date": "2026-10-14",
                "reader_device_id": "TOB-003",
                "corridor_id": "COR-02",
                "acquirer_bank": "BCA",
                "device_tx_count": 900,
                "settled_tx_count": 810,  # 10% paket hilang (> 0.5%)
                "device_total_idr": 5400000.0,
                "acquirer_settled_idr": 4860000.0,  # Selisih Rp540.000 (> Rp50.000)
                "sync_delay_minutes": 240.0,
            },
        ]
    }
    with TestClient(app) as client:
        resp = client.post("/v1/settlement/audit", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_batches"] == 3
        assert data["flagged_batches"] == 1
        flagged_item = [r for r in data["results"] if r["batch_id"] == "BATCH-00003"][0]
        assert flagged_item["is_flagged_discrepancy"] is True
        assert flagged_item["reconciliation_ticket_id"].startswith("RECON-TICKET-")


def test_dod4_streamlit_dashboard_readme_and_animations() -> None:
    """Memverifikasi DoD-4: ke-4 tab app_streamlit.py bebas warning/exception, README lengkap, & animasi tersedia."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app_streamlit.py").run(timeout=30)
    assert len(at.exception) == 0, f"Ditemukan exception pada app_streamlit.py: {at.exception}"
    assert len(at.warning) == 0, f"Ditemukan warning pada app_streamlit.py: {at.warning}"
    assert len(at.tabs) == 4

    # Uji klik tombol simulasi inferensi di Tab 1
    at.button[0].click().run(timeout=30)
    assert len(at.exception) == 0

    readme_path = Path("README.md")
    assert readme_path.exists()
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "Sprint 1" in readme_text and "Sprint 2" in readme_text and "Sprint 3" in readme_text and "Sprint 4" in readme_text
    assert "PR-AUC" in readme_text and "Hybrid Stacking" in readme_text

    assert Path("reports/ainosentinel_pipeline_animation.gif").exists()
    assert Path("reports/ainosentinel_cost_sweep_animation.gif").exists()
    assert Path("reports/ainosentinel_interactive_animation.html").exists()

