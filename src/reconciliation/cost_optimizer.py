"""F-06: Kalibrator ambang batas keputusan (tau*) berbasis matriks biaya bisnis dalam Rupiah."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CostOptimizationResult:
    """Hasil pencarian threshold optimal berbasis biaya bisnis (IDR)."""

    optimal_threshold: float
    cost_fraud_fn_idr: float
    cost_false_alarm_fp_idr: float
    baseline_no_filter_cost_idr: float
    default_050_cost_idr: float
    optimal_total_cost_idr: float
    net_cost_savings_idr: float
    net_cost_savings_pct: float
    fn_at_optimal: int
    fp_at_optimal: int
    tp_at_optimal: int
    tn_at_optimal: int
    thresholds: List[float]
    total_costs_idr: List[float]
    fn_costs_idr: List[float]
    fp_costs_idr: List[float]


class BusinessCostThresholdOptimizer:
    """Mencari threshold keputusan tau* yang meminimalkan C(tau) = FN(tau)*C_fraud + FP(tau)*C_audit."""

    def __init__(
        self,
        cost_fraud_fn_idr: float = 45000.0,
        cost_false_alarm_fp_idr: float = 6500.0,
        num_grid_points: int = 181,
    ) -> None:
        if cost_fraud_fn_idr <= 0 or cost_false_alarm_fp_idr <= 0:
            raise ValueError("Biaya FN dan FP harus bernilai positif (IDR).")
        self.cost_fraud_fn_idr = float(cost_fraud_fn_idr)
        self.cost_false_alarm_fp_idr = float(cost_false_alarm_fp_idr)
        self.num_grid_points = num_grid_points

    def optimize(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> CostOptimizationResult:
        """Mengevaluasi kurva biaya finansial pada rentang tau in [0.05, 0.95] dan memilih tau*."""
        y_t = np.asarray(y_true, dtype=np.int64)
        y_p = np.asarray(y_prob, dtype=np.float64)

        if len(y_t) == 0 or len(y_t) != len(y_p):
            raise ValueError("Panjang y_true dan y_prob harus sama dan tidak kosong.")

        thresholds = np.linspace(0.05, 0.95, self.num_grid_points)
        total_pos = int(np.sum(y_t == 1))

        # Baseline tanpa filter otomatis (semua transaksi dianggap normal -> semua anomali menjadi FN)
        baseline_cost = float(total_pos * self.cost_fraud_fn_idr)

        fn_costs: List[float] = []
        fp_costs: List[float] = []
        total_costs: List[float] = []
        fn_counts: List[int] = []
        fp_counts: List[int] = []
        tp_counts: List[int] = []
        tn_counts: List[int] = []

        pos_mask = y_t == 1
        neg_mask = y_t == 0

        for tau in thresholds:
            pred_pos = y_p >= tau
            tp = int(np.sum(pred_pos & pos_mask))
            fp = int(np.sum(pred_pos & neg_mask))
            fn = int(np.sum((~pred_pos) & pos_mask))
            tn = int(np.sum((~pred_pos) & neg_mask))

            c_fn = float(fn * self.cost_fraud_fn_idr)
            c_fp = float(fp * self.cost_false_alarm_fp_idr)
            c_tot = c_fn + c_fp

            fn_costs.append(c_fn)
            fp_costs.append(c_fp)
            total_costs.append(c_tot)
            fn_counts.append(fn)
            fp_counts.append(fp)
            tp_counts.append(tp)
            tn_counts.append(tn)

        best_idx = int(np.argmin(total_costs))
        opt_tau = round(float(thresholds[best_idx]), 4)
        opt_cost = float(total_costs[best_idx])

        # Hitung biaya pada threshold naif 0.50 sebagai pembanding
        idx_050 = int(np.argmin(np.abs(thresholds - 0.50)))
        cost_050 = float(total_costs[idx_050])

        savings_idr = max(0.0, baseline_cost - opt_cost)
        savings_pct = (100.0 * savings_idr / baseline_cost) if baseline_cost > 0 else 0.0

        return CostOptimizationResult(
            optimal_threshold=opt_tau,
            cost_fraud_fn_idr=self.cost_fraud_fn_idr,
            cost_false_alarm_fp_idr=self.cost_false_alarm_fp_idr,
            baseline_no_filter_cost_idr=round(baseline_cost, 2),
            default_050_cost_idr=round(cost_050, 2),
            optimal_total_cost_idr=round(opt_cost, 2),
            net_cost_savings_idr=round(savings_idr, 2),
            net_cost_savings_pct=round(savings_pct, 2),
            fn_at_optimal=fn_counts[best_idx],
            fp_at_optimal=fp_counts[best_idx],
            tp_at_optimal=tp_counts[best_idx],
            tn_at_optimal=tn_counts[best_idx],
            thresholds=[round(float(x), 4) for x in thresholds],
            total_costs_idr=[round(float(x), 2) for x in total_costs],
            fn_costs_idr=[round(float(x), 2) for x in fn_costs],
            fp_costs_idr=[round(float(x), 2) for x in fp_costs],
        )

    @staticmethod
    def save_artifacts(
        result: CostOptimizationResult,
        json_path: str | Path = "reports/optimal_threshold.json",
        plot_path: str | Path = "reports/cost_curve.png",
    ) -> None:
        """Menyimpan konfigurasi threshold optimal ke JSON dan mem-plot kurva biaya bisnis."""
        j_target = Path(json_path)
        j_target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(result)
        j_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        p_target = Path(plot_path)
        p_target.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=140)
        taus = np.array(result.thresholds)
        tot_m = np.array(result.total_costs_idr) / 1_000_000.0
        fn_m = np.array(result.fn_costs_idr) / 1_000_000.0
        fp_m = np.array(result.fp_costs_idr) / 1_000_000.0

        ax.plot(taus, tot_m, color="#0F766E", linewidth=2.6, label="Total Biaya Ekspektasi (FN + FP)")
        ax.plot(taus, fn_m, color="#DC2626", linestyle="--", linewidth=1.8, label="Kerugian Fraud / Kebocoran (FN)")
        ax.plot(taus, fp_m, color="#2563EB", linestyle=":", linewidth=1.8, label="Biaya Investigasi & Komplain (FP)")

        opt_m = result.optimal_total_cost_idr / 1_000_000.0
        ax.axvline(result.optimal_threshold, color="#D97706", linestyle="-.", linewidth=1.8)
        ax.scatter([result.optimal_threshold], [opt_m], color="#DC2626", zorder=5, s=65)
        ax.annotate(
            f"Optimal τ* = {result.optimal_threshold:.2f}\n(Biaya Min: Rp{opt_m:.2f} Jt | Hemat {result.net_cost_savings_pct:.1f}%)",
            xy=(result.optimal_threshold, opt_m),
            xytext=(min(0.62, result.optimal_threshold + 0.06), opt_m + max(0.4, tot_m.max() * 0.18)),
            arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.4},
            fontsize=9.5,
            fontweight="bold",
            color="#991B1B",
        )

        ax.set_title("Kalibrasi Threshold Berbasis Matriks Biaya Bisnis (AinoSentinel)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ambang Batas Probabilitas Anomali (Threshold τ)", fontsize=10.5)
        ax.set_ylabel("Estimasi Biaya Finansial (Juta IDR)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(p_target)
        plt.close(fig)
