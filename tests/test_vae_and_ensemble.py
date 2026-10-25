"""Unit tests untuk memverifikasi PyTorch VAE, torch.inference_mode(), Hybrid Stacking, dan metrik DoD-2."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.hybrid_ensemble import HybridStackingEnsemble
from src.models.pytorch_vae import (
    PyTorchVAEAnomalyDetector,
    ResidualVariationalAutoencoder,
    vae_loss_function,
)
from src.reconciliation.cost_optimizer import BusinessCostThresholdOptimizer


def test_pytorch_vae_tensor_dimensions_and_loss() -> None:
    """Memastikan dimensi tensor PyTorch Residual VAE (28 -> 64 -> 12 -> 64 -> 28) & fungsi rugi beta-VAE."""
    model = ResidualVariationalAutoencoder(input_dim=28, hidden_dim=64, latent_dim=12, beta_kl=0.05)
    x = torch.randn(32, 28)

    model.train()
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == (32, 28)
    assert mu.shape == (32, 12)
    assert logvar.shape == (32, 12)

    total_loss, mse_val, kl_val = vae_loss_function(x, x_hat, mu, logvar, beta=0.05)
    assert torch.isfinite(total_loss)
    assert float(mse_val.item()) > 0.0
    assert float(kl_val.item()) >= 0.0


def test_pytorch_vae_inference_mode_and_anomaly_separation() -> None:
    """Memastikan evaluasi PyTorch VAE menggunakan torch.inference_mode() dan memisahkan anomali dari normal."""
    rng = np.random.default_rng(42)
    x_normal = rng.normal(0.0, 0.5, size=(400, 28)).astype(np.float32)
    x_anom = rng.normal(4.5, 1.5, size=(50, 28)).astype(np.float32)

    detector = PyTorchVAEAnomalyDetector(input_dim=28, hidden_dim=64, latent_dim=12, epochs=12, seed=42)
    detector.fit_normal(x_normal)

    out_norm = detector.score_details(x_normal[:50])
    out_anom = detector.score_details(x_anom)

    assert detector.last_used_inference_mode is True, "PyTorch VAE wajib dievaluasi di dalam torch.inference_mode()!"
    assert float(np.median(out_anom.total_anomaly_score)) > float(np.median(out_norm.total_anomaly_score)) * 3.0
    assert out_anom.per_feature_abs_error.shape == (50, 28)


def test_business_cost_optimizer_finds_optimal_tau() -> None:
    """Memastikan Cost Optimizer menemukan threshold tau* yang meminimalkan fungsi biaya Rupiah."""
    y_true = np.array([0] * 950 + [1] * 50, dtype=np.int64)
    y_prob = np.concatenate(
        [
            np.linspace(0.01, 0.35, 950),
            np.linspace(0.42, 0.99, 50),
        ]
    )
    opt = BusinessCostThresholdOptimizer(cost_fraud_fn_idr=45000.0, cost_false_alarm_fp_idr=6500.0)
    res = opt.optimize(y_true, y_prob)

    assert 0.30 <= res.optimal_threshold <= 0.48
    assert res.optimal_total_cost_idr < res.baseline_no_filter_cost_idr
    assert res.net_cost_savings_pct >= 85.0


def test_benchmark_metrics_and_dod2_thresholds() -> None:
    """Memverifikasi artefak model terlatih dan pencapaian target DoD-2 (PR-AUC >= 0.92, Recall@FPR=1% >= 0.89)."""
    metrics_path = Path("reports/benchmark_metrics.json")
    if not metrics_path.exists():
        from src.models.evaluate_models import main as eval_main
        eval_main()

    assert Path("models/vae_normal.pt").exists()
    assert Path("models/hybrid_lgbm.pkl").exists()
    assert Path("models/scaler.pkl").exists()
    assert Path("reports/optimal_threshold.json").exists()
    assert Path("reports/cost_curve.png").exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    headline = payload["headline_metrics"]

    assert headline["hybrid_pr_auc"] >= 0.92, f"PR-AUC kurang dari target 0.92: {headline['hybrid_pr_auc']}"
    assert headline["hybrid_f2_score"] >= 0.88, f"F2-Score kurang dari target 0.88: {headline['hybrid_f2_score']}"
    assert (
        headline["hybrid_recall_at_fpr_1pct"] >= 0.89
    ), f"Recall@FPR=1% kurang dari target 0.89: {headline['hybrid_recall_at_fpr_1pct']}"
    assert (
        headline["double_tap_precision"] >= 0.97
    ), f"Double-Tap Precision kurang dari target 0.97: {headline['double_tap_precision']}"
    assert headline["hybrid_brier_score"] <= 0.025, f"Brier Score melebihi 0.025: {headline['hybrid_brier_score']}"
    assert (
        headline["net_cost_savings_pct"] >= 78.0
    ), f"Net Cost Savings kurang dari 78%: {headline['net_cost_savings_pct']}%"

    # Uji pemuatan artefak dan penjelasan fitur SHAP lokal
    ens = HybridStackingEnsemble.load("models/hybrid_lgbm.pkl")
    dummy_row = np.zeros((1, 28), dtype=np.float64)
    dummy_row[0, 0] = 3.0  # delta_t_card_sec = 3s
    dummy_row[0, 10] = 1.0  # is_same_reader = 1
    dummy_row[0, 12] = 1.0  # is_same_tap_type = 1
    stacked = ens.build_stacked_features(dummy_row)
    shap_top = ens.explain_single_shap(stacked[0], top_k=5)
    assert len(shap_top) == 5
    assert "feature_name" in shap_top[0] and "shap_or_z_impact" in shap_top[0]
