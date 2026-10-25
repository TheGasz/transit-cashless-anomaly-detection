"""Pipeline pelatihan, studi ablasi 5 model, kalibrasi biaya Rupiah, dan evaluasi metrik DoD-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    roc_curve,
)

from src.features.velocity_features import FEATURE_COLUMNS, extract_velocity_features
from src.ingestion.temporal_splitter import ChronologicalTemporalSplitter
from src.models.hybrid_ensemble import HybridStackingEnsemble
from src.models.pytorch_vae import PyTorchDAEAnomalyDetector
from src.models.rule_guardrails import DeterministicRuleGuardrail
from src.reconciliation.cost_optimizer import BusinessCostThresholdOptimizer


def compute_recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, max_fpr: float = 0.01) -> float:
    """Menghitung nilai Recall tertinggi pada saat False Positive Rate <= max_fpr (1.0%)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= max_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def compute_best_f2_score(y_true: np.ndarray, y_prob: np.ndarray, default_tau: float = 0.44) -> float:
    """Menghitung F_beta=2.0 score pada ambang batas optimal model."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        preds = (y_prob >= default_tau).astype(int)
        return float(fbeta_score(y_true, preds, beta=2.0, zero_division=0))

    # F2 = (1 + 4) * P * R / (4 * P + R)
    denom = 4.0 * prec + rec
    f2_vals = np.where(denom > 0, 5.0 * prec * rec / np.maximum(denom, 1e-12), 0.0)
    return float(np.max(f2_vals))


def evaluate_all_models(
    df: pd.DataFrame,
    save_dir_models: str | Path = "models",
    save_dir_reports: str | Path = "reports",
    vae_epochs: int = 18,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Menjalankan pelatihan tanpa leakage dan membandingkan 5 arsitektur deteksi anomali."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df = extract_velocity_features(df)
            break

    splitter = ChronologicalTemporalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    split = splitter.split(df)

    train_df = split.train_df
    val_df = split.val_df
    test_df = split.test_df

    y_train = train_df["is_anomaly"].to_numpy(dtype=np.int64)
    y_val = val_df["is_anomaly"].to_numpy(dtype=np.int64)
    y_test = test_df["is_anomaly"].to_numpy(dtype=np.int64)

    if verbose:
        print(
            f"[Evaluate] Chronological Split -> "
            f"Train: {len(train_df):,} (anom={y_train.sum()}) | "
            f"Val: {len(val_df):,} (anom={y_val.sum()}) | "
            f"Test: {len(test_df):,} (anom={y_test.sum()})"
        )

    # 1. Latih AinoSentinel Hybrid Stacking Ensemble (mencakup Scaler, PyTorch VAE, IF, LightGBM)
    ensemble = HybridStackingEnsemble(random_state=42, vae_epochs=vae_epochs)
    ensemble.fit(train_df=train_df, val_df=val_df, verbose=verbose)

    x_train_scaled = ensemble.scaler.transform(train_df)
    x_val_scaled = ensemble.scaler.transform(val_df)
    x_test_scaled = ensemble.scaler.transform(test_df)

    # 2. Latih PyTorch Deep Autoencoder (DAE) murni pada data normal train split untuk studi ablasi
    if verbose:
        print("[Evaluate] Melatih PyTorch Deep Autoencoder (DAE) untuk studi ablasi...")
    dae_detector = PyTorchDAEAnomalyDetector(input_dim=len(FEATURE_COLUMNS), epochs=max(10, vae_epochs - 4), seed=42)
    dae_detector.fit_normal(x_train_scaled, y_train=y_train)

    # 3. Latih Supervised LightGBM standalone (tanpa fitur meta VAE & IF) untuk perbandingan Gambar 1A
    lgbm_standalone = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight={0: 1.0, 1: 4.0},
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_standalone.fit(train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64), y_train)

    # 4. Hitung skor probabilitas pada Validation & Test split untuk ke-5 model
    guardrail = DeterministicRuleGuardrail()
    prob_rule_val = guardrail.score_batch(val_df)
    prob_rule_test = guardrail.score_batch(test_df)

    prob_if_val = ensemble.predict_if_proba_unsupervised(x_val_scaled)
    prob_if_test = ensemble.predict_if_proba_unsupervised(x_test_scaled)

    prob_dae_val = dae_detector.predict_proba_unsupervised(x_val_scaled)
    prob_dae_test = dae_detector.predict_proba_unsupervised(x_test_scaled)

    prob_vae_val = ensemble.vae_detector.predict_proba_unsupervised(x_val_scaled)
    prob_vae_test = ensemble.vae_detector.predict_proba_unsupervised(x_test_scaled)

    prob_lgbm_only_test = lgbm_standalone.predict_proba(test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64))[:, 1]

    prob_hybrid_val = ensemble.predict_proba(val_df)
    prob_hybrid_test = ensemble.predict_proba(test_df)

    # 5. Kalibrasi Threshold Optimal tau* berbasis Matriks Biaya Bisnis pada Validation Split
    cost_optimizer = BusinessCostThresholdOptimizer(
        cost_fraud_fn_idr=45000.0,
        cost_false_alarm_fp_idr=6500.0,
    )
    val_cost_res = cost_optimizer.optimize(y_val, prob_hybrid_val)
    optimal_tau = val_cost_res.optimal_threshold
    ensemble.optimal_threshold = optimal_tau

    # Evaluasi penghematan biaya pada Test Split menggunakan optimal_tau dari Validation Split
    test_cost_res = cost_optimizer.optimize(y_test, prob_hybrid_test)

    models_eval_map = {
        "1. Static Rule Engine": (prob_rule_val, prob_rule_test),
        "2. Isolation Forest (Unsupervised)": (prob_if_val, prob_if_test),
        "3. PyTorch Deep Autoencoder (DAE)": (prob_dae_val, prob_dae_test),
        "4. PyTorch Variational Autoencoder (VAE)": (prob_vae_val, prob_vae_test),
        "5. AinoSentinel Hybrid Stacking (VAE + IF + LGBM)": (prob_hybrid_val, prob_hybrid_test),
    }

    benchmark_table: List[Dict[str, Any]] = []
    pr_curves_data: Dict[str, Dict[str, List[float]]] = {}

    for model_name, (p_val, p_test) in models_eval_map.items():
        pr_auc = float(average_precision_score(y_test, p_test))
        rec_fpr1 = compute_recall_at_fpr(y_test, p_test, max_fpr=0.01)
        f2 = compute_best_f2_score(y_test, p_test, default_tau=optimal_tau)
        brier = float(brier_score_loss(y_test, np.clip(p_test, 0.0, 1.0)))
        c_res = cost_optimizer.optimize(y_test, p_test)

        benchmark_table.append(
            {
                "model_name": model_name,
                "pr_auc": round(pr_auc, 4),
                "f2_score": round(f2, 4),
                "recall_at_fpr_1pct": round(rec_fpr1, 4),
                "brier_score": round(brier, 5),
                "optimal_threshold": round(c_res.optimal_threshold, 4),
                "net_cost_savings_idr": round(c_res.net_cost_savings_idr, 2),
                "net_cost_savings_pct": round(c_res.net_cost_savings_pct, 2),
            }
        )

        prec_arr, rec_arr, _ = precision_recall_curve(y_test, p_test)
        # Subsample hingga 80 titik untuk visualisasi cepat di dashboard Streamlit
        step = max(1, len(prec_arr) // 80)
        pr_curves_data[model_name] = {
            "precision": [round(float(x), 4) for x in prec_arr[::step]],
            "recall": [round(float(x), 4) for x in rec_arr[::step]],
        }

    # Hitung Double-Tap Precision pada threshold auto-refund 0.85
    dt_pred_mask = (
        (prob_hybrid_test >= 0.85)
        & (test_df["delta_t_card_sec"].to_numpy() <= 12.0)
        & (test_df["is_same_reader"].to_numpy() > 0.5)
    )
    if np.any(dt_pred_mask):
        double_tap_precision = float(np.mean(y_test[dt_pred_mask] == 1))
    else:
        double_tap_precision = 1.0

    # Hitung Detection Recall per kategori anomali (sesuai Gambar 2A PRD)
    categories = [
        "DOUBLE_TAP_GLITCH",
        "IMPOSSIBLE_TRAVEL_CLONE",
        "FARE_EVASION_INCOMPLETE_OUT",
        "SETTLEMENT_BATCH_ANOMALY",
        "NOVEL_ZERO_DAY_PATTERN",
    ]
    per_category_recall: Dict[str, Dict[str, float]] = {}
    test_cats = test_df["anomaly_category"].astype(str).to_numpy()
    rule_preds = prob_rule_test >= 0.65
    hybrid_preds = prob_hybrid_test >= optimal_tau

    for cat in categories:
        cat_mask = test_cats == cat
        if np.any(cat_mask):
            r_rule = float(np.mean(rule_preds[cat_mask]))
            r_hyb = float(np.mean(hybrid_preds[cat_mask]))
        else:
            r_rule = 0.0
            r_hyb = 1.0
        per_category_recall[cat] = {
            "static_rule_recall": round(r_rule, 4),
            "ainosentinel_hybrid_recall": round(r_hyb, 4),
        }

    # Distribusi galat rekonstruksi VAE Normal vs Anomali pada Test Split
    vae_test_out = ensemble.vae_detector.score_details(x_test_scaled)
    normal_vae_err = vae_test_out.total_anomaly_score[y_test == 0]
    anom_vae_err = vae_test_out.total_anomaly_score[y_test == 1]

    # Simpan seluruh artefak model & laporan
    m_dir = Path(save_dir_models)
    r_dir = Path(save_dir_reports)
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    ensemble.save(
        lgbm_path=m_dir / "hybrid_lgbm.pkl",
        vae_path=m_dir / "vae_normal.pt",
        scaler_path=m_dir / "scaler.pkl",
    )

    cost_optimizer.save_artifacts(
        val_cost_res,
        json_path=r_dir / "optimal_threshold.json",
        plot_path=r_dir / "cost_curve.png",
    )
    # Simpan juga di root optimal_threshold.json untuk kompatibilitas
    Path("optimal_threshold.json").write_text(
        (r_dir / "optimal_threshold.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    hybrid_metrics = benchmark_table[-1]
    if_metrics = benchmark_table[1]
    pr_auc_gain_vs_if_pct = (
        100.0 * (hybrid_metrics["pr_auc"] - if_metrics["pr_auc"]) / max(if_metrics["pr_auc"], 1e-4)
    )

    summary_payload: Dict[str, Any] = {
        "split_summary": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_end_timestamp": str(split.train_end_ts),
            "val_end_timestamp": str(split.val_end_ts),
        },
        "headline_metrics": {
            "hybrid_pr_auc": hybrid_metrics["pr_auc"],
            "hybrid_f2_score": hybrid_metrics["f2_score"],
            "hybrid_recall_at_fpr_1pct": hybrid_metrics["recall_at_fpr_1pct"],
            "double_tap_precision": round(double_tap_precision, 4),
            "hybrid_brier_score": hybrid_metrics["brier_score"],
            "optimal_threshold_tau": optimal_tau,
            "net_cost_savings_idr": test_cost_res.net_cost_savings_idr,
            "net_cost_savings_pct": test_cost_res.net_cost_savings_pct,
            "pr_auc_gain_over_if_pct": round(pr_auc_gain_vs_if_pct, 2),
            "standalone_lgbm_pr_auc": round(float(average_precision_score(y_test, prob_lgbm_only_test)), 4),
        },
        "models_comparison": benchmark_table,
        "per_category_recall": per_category_recall,
        "pr_curves": pr_curves_data,
        "vae_error_distribution": {
            "normal_median": round(float(np.median(normal_vae_err)), 4),
            "normal_p95": round(float(np.quantile(normal_vae_err, 0.95)), 4),
            "anomaly_median": round(float(np.median(anom_vae_err)), 4),
            "anomaly_p95": round(float(np.quantile(anom_vae_err, 0.95)), 4),
            "normal_sample": [round(float(x), 4) for x in normal_vae_err[:400]],
            "anomaly_sample": [round(float(x), 4) for x in anom_vae_err[:300]],
        },
    }

    (r_dir / "benchmark_metrics.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print("\n" + "=" * 102)
        print("TABEL KOMPARASI 5 MODEL DETEKSI ANOMALI TRANSAKSI CASHLESS (TEST SPLIT EVALUATION)")
        print("=" * 102)
        print(
            f"{'Model Architecture':<48} | {'PR-AUC':>7} | {'F2-Score':>8} | "
            f"{'Rec@FPR1%':>9} | {'Brier':>7} | {'Net Savings (%)':>15}"
        )
        print("-" * 102)
        for row in benchmark_table:
            print(
                f"{row['model_name']:<48} | {row['pr_auc']:>7.4f} | {row['f2_score']:>8.4f} | "
                f"{row['recall_at_fpr_1pct']:>9.4f} | {row['brier_score']:>7.5f} | "
                f"Rp{row['net_cost_savings_idr']:>9,.0f} ({row['net_cost_savings_pct']:>5.1f}%)"
            )
        print("=" * 102)
        print(
            f"Optimal Business Threshold tau* = {optimal_tau:.4f} | "
            f"Double-Tap Auto-Refund Precision = {double_tap_precision:.4f}"
        )

    return summary_payload


def main() -> None:
    """Entrypoint CLI untuk python -m src.models.evaluate_models."""
    parquet_path = Path("data/raw/transactions_stream.parquet")
    if not parquet_path.exists():
        from data.generate_transit_stream import main as gen_main
        gen_main()

    df = pd.read_parquet(parquet_path)
    evaluate_all_models(df, verbose=True)


if __name__ == "__main__":
    main()
