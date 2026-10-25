"""Generator animasi visual end-to-end AinoSentinel (GIF Multi-Panel & Interactive HTML5 Player)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from src.features.velocity_features import FEATURE_COLUMNS
from src.models.hybrid_ensemble import HybridStackingEnsemble


def generate_pipeline_gif(
    tx_df: pd.DataFrame,
    ensemble: HybridStackingEnsemble,
    metrics: dict,
    output_path: str | Path = "reports/ainosentinel_pipeline_animation.gif",
    num_frames: int = 36,
) -> Path:
    """Membuat animasi GIF 4-Panel Operations Center yang menampilkan deteksi real-time AinoSentinel."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Pilih urutan menarik dari Test Split yang memuat transaksi normal + ke-5 tipe anomali
    tail_df = tx_df.tail(18000).copy().reset_index(drop=True)
    anom_rows = tail_df[tail_df["is_anomaly"] == 1].head(18)
    norm_rows = tail_df[tail_df["is_anomaly"] == 0].head(54)

    # Susun 72 langkah aliran event (selang-seling normal dan anomali supaya visualisasi hidup)
    stream_indices: List[int] = []
    n_idx_list = norm_rows.index.tolist()
    a_idx_list = anom_rows.index.tolist()
    for step in range(num_frames):
        if step < len(n_idx_list):
            stream_indices.append(n_idx_list[step])
        if step % 2 == 1 and (step // 2) < len(a_idx_list):
            stream_indices.append(a_idx_list[step // 2])

    stream_df = tail_df.loc[stream_indices[:num_frames]].reset_index(drop=True)
    x_raw = stream_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    x_stk, vae_out, if_scores = ensemble.build_stacked_features(x_raw, return_details=True)
    probs = ensemble.predict_proba_from_stacked(x_stk, vae_out)
    tau = float(ensemble.optimal_threshold)

    # Ambil 2 dimensi laten utama dari PyTorch VAE untuk radar ruang laten
    latent_2d = vae_out.latent_mu[:, :2]

    # Koordinat normalisasi koridor Jakarta untuk visualisasi peta jaringan
    jkt_stops = tx_df[tx_df["corridor_id"].isin(["COR-01", "COR-02", "COR-05"])][
        ["stop_id", "corridor_id", "latitude", "longitude"]
    ].drop_duplicates()

    frames: List[Image.Image] = []
    cum_saved_idr = 0.0
    cum_refund_cnt = 0
    cum_freeze_cnt = 0

    saved_history: List[float] = []

    for f_idx in range(len(stream_df)):
        row = stream_df.iloc[f_idx]
        p_val = float(probs[f_idx])
        v_err = float(vae_out.total_anomaly_score[f_idx])
        cat = str(row["anomaly_category"])
        is_anom = p_val >= tau

        if is_anom:
            if cat == "DOUBLE_TAP_GLITCH":
                cum_refund_cnt += 1
                cum_saved_idr += float(row["fare_amount"])
            else:
                cum_freeze_cnt += 1
                cum_saved_idr += 45000.0
        saved_history.append(cum_saved_idr / 1000.0)

        fig = plt.figure(figsize=(13.5, 7.6), dpi=105, facecolor="#0B1120")
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22, left=0.06, right=0.96, top=0.88, bottom=0.08)

        fig.suptitle(
            f"AINOSENTINEL REAL-TIME CASHLESS ANOMALY ENGINE — LIVE STREAM FRAME [{f_idx + 1:02d}/{len(stream_df):02d}]",
            color="#F8FAFC",
            fontsize=13.5,
            fontweight="bold",
            y=0.96,
        )

        # ---------------------------------------------------------------------
        # PANEL 1: LIVE TRANSIT CORRIDOR & GEODESIC VELOCITY RADAR
        # ---------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0], facecolor="#111827")
        for cid, col_c in [("COR-01", "#38BDF8"), ("COR-02", "#A78BFA"), ("COR-05", "#34D399")]:
            sub = jkt_stops[jkt_stops["corridor_id"] == cid].sort_values("latitude")
            ax1.plot(sub["longitude"], sub["latitude"], "-o", color=col_c, markersize=4, alpha=0.55, linewidth=1.5, label=cid)

        cur_lon = float(row["longitude"])
        cur_lat = float(row["latitude"])
        # Petakan ke bounding box tampilan jika berasal dari luar Jabodetabek
        disp_lon = np.clip(cur_lon, 106.77, 106.91)
        disp_lat = np.clip(cur_lat, -6.62, -6.12)

        point_color = "#EF4444" if is_anom else "#10B981"
        ax1.scatter([disp_lon], [disp_lat], s=210 if is_anom else 110, color=point_color, edgecolors="white", linewidths=2.0, zorder=6)

        if is_anom and f_idx > 0:
            prev_lon = np.clip(float(stream_df.iloc[f_idx - 1]["longitude"]), 106.77, 106.91)
            prev_lat = np.clip(float(stream_df.iloc[f_idx - 1]["latitude"]), -6.62, -6.12)
            ax1.plot([prev_lon, disp_lon], [prev_lat, disp_lat], "--", color="#F59E0B", linewidth=2.2, zorder=5)

        status_badge = f"ALERT: {cat} (v={row['velocity_kmh']:.0f} km/h, dt={row['delta_t_card_sec']:.0f}s)" if is_anom else f"NORMAL TAP ({row['payment_channel']} - {row['stop_id']})"
        ax1.set_title(f"1. Geodesic Corridor Telemetry | {status_badge}", color="#FDE047" if is_anom else "#6EE7B7", fontsize=10, fontweight="bold")
        ax1.set_xlim(106.76, 106.92)
        ax1.set_ylim(-6.64, -6.10)
        ax1.set_xlabel("Longitude (GPS)", color="#94A3B8", fontsize=8.5)
        ax1.set_ylabel("Latitude (GPS)", color="#94A3B8", fontsize=8.5)
        ax1.tick_params(colors="#94A3B8", labelsize=8)
        ax1.grid(True, linestyle=":", color="#334155", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=7.5, facecolor="#1E293B", labelcolor="white")

        # ---------------------------------------------------------------------
        # PANEL 2: REAL-TIME 3-TIER ANOMALY PROBABILITY STREAM vs THRESHOLD tau*
        # ---------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1], facecolor="#111827")
        x_steps = np.arange(1, f_idx + 2)
        y_probs = probs[: f_idx + 1]
        ax2.plot(x_steps, y_probs, "-o", color="#38BDF8", linewidth=2.0, markersize=4.5, label="Hybrid Score p_anom")
        ax2.axhline(tau, color="#F59E0B", linestyle="--", linewidth=1.8, label=f"Cost-Optimal Threshold τ* = {tau:.2f}")

        anom_steps = x_steps[y_probs >= tau]
        anom_vals = y_probs[y_probs >= tau]
        if len(anom_steps) > 0:
            ax2.scatter(anom_steps, anom_vals, color="#EF4444", s=75, zorder=5, label="Intercepted Anomaly")

        ax2.set_title(
            f"2. Real-Time Hybrid Score Stream (Current p_anom = {p_val:.3f})",
            color="#F8FAFC",
            fontsize=10,
            fontweight="bold",
        )
        ax2.set_xlim(1, len(stream_df))
        ax2.set_ylim(-0.05, 1.08)
        ax2.set_xlabel("Incoming Transaction Sequence", color="#94A3B8", fontsize=8.5)
        ax2.set_ylabel("Calibrated Anomaly Prob", color="#94A3B8", fontsize=8.5)
        ax2.tick_params(colors="#94A3B8", labelsize=8)
        ax2.grid(True, linestyle=":", color="#334155", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=7.5, facecolor="#1E293B", labelcolor="white")

        # ---------------------------------------------------------------------
        # PANEL 3: PYTORCH VAE LATENT MANIFOLD & RECONSTRUCTION ERROR
        # ---------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0], facecolor="#111827")
        theta = np.linspace(0, 2 * np.pi, 100)
        ax3.plot(2.2 * np.cos(theta), 2.2 * np.sin(theta), "--", color="#10B981", alpha=0.6, label="99% Normal Gaussian Prior N(0,I)")

        past_z = latent_2d[: f_idx + 1]
        past_p = probs[: f_idx + 1]
        norm_m = past_p < tau
        anom_m = past_p >= tau

        if np.any(norm_m):
            ax3.scatter(past_z[norm_m, 0], past_z[norm_m, 1], color="#10B981", s=32, alpha=0.75, label="Normal Latent z")
        if np.any(anom_m):
            ax3.scatter(past_z[anom_m, 0], past_z[anom_m, 1], color="#EF4444", s=75, marker="^", label="Out-of-Manifold Anomaly")

        ax3.scatter([latent_2d[f_idx, 0]], [latent_2d[f_idx, 1]], color="#FDE047", s=150, edgecolors="white", zorder=6)
        ax3.set_title(
            f"3. PyTorch Residual VAE Latent Space (e_VAE = {v_err:.3f})",
            color="#F8FAFC",
            fontsize=10,
            fontweight="bold",
        )
        ax3.set_xlim(-5.5, 5.5)
        ax3.set_ylim(-5.5, 5.5)
        ax3.set_xlabel("Latent Dimension z_1", color="#94A3B8", fontsize=8.5)
        ax3.set_ylabel("Latent Dimension z_2", color="#94A3B8", fontsize=8.5)
        ax3.tick_params(colors="#94A3B8", labelsize=8)
        ax3.grid(True, linestyle=":", color="#334155", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=7.5, facecolor="#1E293B", labelcolor="white")

        # ---------------------------------------------------------------------
        # PANEL 4: CUMULATIVE FINANCIAL SAVINGS & AUTO-REMEDIATION ROUTER
        # ---------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1], facecolor="#111827")
        ax4.fill_between(x_steps, 0, saved_history, color="#14B8A6", alpha=0.30)
        ax4.plot(x_steps, saved_history, "-o", color="#2DD4BF", linewidth=2.2, markersize=4)
        ax4.set_title(
            f"4. Cumulative Revenue Protected: Rp {cum_saved_idr:,.0f} "
            f"(Refunds: {cum_refund_cnt} | Blocked Fraud: {cum_freeze_cnt})",
            color="#5EEAD4",
            fontsize=10,
            fontweight="bold",
        )
        ax4.set_xlim(1, len(stream_df))
        ax4.set_ylim(0, max(50.0, max(saved_history) * 1.25))
        ax4.set_xlabel("Incoming Transaction Sequence", color="#94A3B8", fontsize=8.5)
        ax4.set_ylabel("Saved Loss (Thousand IDR)", color="#94A3B8", fontsize=8.5)
        ax4.tick_params(colors="#94A3B8", labelsize=8)
        ax4.grid(True, linestyle=":", color="#334155", alpha=0.6)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=160,
        loop=0,
        optimize=True,
    )
    return out_file


def generate_cost_sweep_gif(
    opt_json_path: str | Path = "reports/optimal_threshold.json",
    metrics_json_path: str | Path = "reports/benchmark_metrics.json",
    output_path: str | Path = "reports/ainosentinel_cost_sweep_animation.gif",
) -> Path:
    """Membuat animasi GIF pencarian threshold optimal tau* pada kurva biaya bisnis & komparasi 5 model."""
    out_file = Path(output_path)
    opt_data = json.loads(Path(opt_json_path).read_text(encoding="utf-8"))
    metrics_data = json.loads(Path(metrics_json_path).read_text(encoding="utf-8"))

    taus = np.array(opt_data["thresholds"])
    tot_m = np.array(opt_data["total_costs_idr"]) / 1e6
    fn_m = np.array(opt_data["fn_costs_idr"]) / 1e6
    fp_m = np.array(opt_data["fp_costs_idr"]) / 1e6
    opt_tau = float(opt_data["optimal_threshold"])
    opt_cost_m = float(opt_data["optimal_total_cost_idr"]) / 1e6

    models_comp = metrics_data["models_comparison"]
    short_names = ["Rule Engine", "Isolation Forest", "PyTorch DAE", "PyTorch VAE", "AinoSentinel\nHybrid"]
    pr_aucs = [m["pr_auc"] for m in models_comp]
    rec_fpr1 = [m["recall_at_fpr_1pct"] for m in models_comp]

    sweep_indices = np.linspace(0, len(taus) - 1, 24, dtype=int).tolist()
    best_idx = int(np.argmin(tot_m))
    # Tambahkan beberapa frame berhenti di titik optimal tau*
    sweep_indices.extend([best_idx] * 6)

    frames: List[Image.Image] = []
    for step_num, idx in enumerate(sweep_indices):
        cur_tau = float(taus[idx])
        cur_cost = float(tot_m[idx])

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=105, facecolor="#0F172A")

        # Panel Kiri: Animasi Pertumbuhan Benchmark 5 Model
        prog = min(1.0, (step_num + 1) / 14.0)
        x_pos = np.arange(len(short_names))
        w = 0.36
        ax_l.set_facecolor("#1E293B")
        ax_l.bar(x_pos - w / 2, np.array(pr_aucs) * prog, width=w, color="#14B8A6", label="PR-AUC (Test Split)")
        ax_l.bar(x_pos + w / 2, np.array(rec_fpr1) * prog, width=w, color="#3B82F6", label="Recall @ FPR <= 1%")
        ax_l.axhline(0.92, color="#F59E0B", linestyle="--", linewidth=1.4, label="Target DoD PR-AUC (0.92)")
        ax_l.set_xticks(x_pos)
        ax_l.set_xticklabels(short_names, color="#E2E8F0", fontsize=8.5)
        ax_l.set_ylim(0.0, 1.12)
        ax_l.set_title("A. Ablation Study: 5 Anomaly Detection Architectures", color="white", fontsize=10.5, fontweight="bold")
        ax_l.tick_params(colors="#CBD5E1")
        ax_l.grid(True, axis="y", linestyle=":", color="#475569", alpha=0.6)
        ax_l.legend(loc="lower right", fontsize=8, facecolor="#0F172A", labelcolor="white")

        # Panel Kanan: Animasi Pencarian Threshold Optimal tau*
        ax_r.set_facecolor("#1E293B")
        ax_r.plot(taus, tot_m, color="#2DD4BF", linewidth=2.5, label="Total Expected Cost (FN + FP)")
        ax_r.plot(taus, fn_m, color="#F87171", linestyle="--", linewidth=1.6, label="Fraud / Leakage Cost (FN)")
        ax_r.plot(taus, fp_m, color="#60A5FA", linestyle=":", linewidth=1.6, label="False Alarm Audit Cost (FP)")

        ax_r.axvline(cur_tau, color="#FDE047", linestyle="-.", linewidth=1.8)
        ax_r.scatter([cur_tau], [cur_cost], color="#FDE047", s=95, zorder=6)

        if idx == best_idx:
            ax_r.scatter([opt_tau], [opt_cost_m], color="#EF4444", s=140, edgecolors="white", zorder=7)
            ax_r.annotate(
                f"LOCKED OPTIMAL τ* = {opt_tau:.2f}\nMin Cost: Rp {opt_cost_m:.2f} Jt (Saved {opt_data['net_cost_savings_pct']:.1f}%)",
                xy=(opt_tau, opt_cost_m),
                xytext=(min(0.55, opt_tau + 0.05), opt_cost_m + max(1.5, tot_m.max() * 0.25)),
                arrowprops={"arrowstyle": "->", "color": "#FDE047", "lw": 1.8},
                color="#FDE047",
                fontweight="bold",
                fontsize=9,
            )

        ax_r.set_title(
            f"B. Cost-Matrix Threshold Calibration (Scanning τ = {cur_tau:.2f} -> Cost Rp {cur_cost:.2f} Jt)",
            color="white",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_r.set_xlabel("Anomaly Probability Decision Threshold (τ)", color="#CBD5E1", fontsize=9)
        ax_r.set_ylabel("Financial Cost on Validation Split (Million IDR)", color="#CBD5E1", fontsize=9)
        ax_r.tick_params(colors="#CBD5E1")
        ax_r.grid(True, linestyle=":", color="#475569", alpha=0.6)
        ax_r.legend(loc="upper right", fontsize=8, facecolor="#0F172A", labelcolor="white")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
        optimize=True,
    )
    return out_file


def generate_interactive_html_animation(
    tx_df: pd.DataFrame,
    output_path: str | Path = "reports/ainosentinel_interactive_animation.html",
) -> Path:
    """Membuat berkas animasi HTML5 interaktif dengan tombol Play/Pause menggunakan Plotly Frames."""
    out_file = Path(output_path)
    sample = tx_df.tail(2400).copy()
    sample["hour_bucket"] = sample["timestamp"].str.slice(0, 13)

    agg = (
        sample.groupby(["hour_bucket", "corridor_id", "latitude", "longitude"], as_index=False)
        .agg(
            tx_count=("tx_id", "count"),
            anom_count=("is_anomaly", "sum"),
            max_vel=("velocity_kmh", "max"),
        )
    )

    fig = px.scatter(
        agg,
        x="longitude",
        y="latitude",
        animation_frame="hour_bucket",
        size="tx_count",
        color="corridor_id",
        hover_name="corridor_id",
        hover_data=["tx_count", "anom_count", "max_vel"],
        title="AinoSentinel Interactive Multi-Corridor Telemetry Player",
        template="plotly_dark",
    )
    fig.write_html(str(out_file), include_plotlyjs="cdn")
    return out_file


def main() -> None:
    """Menjalankan pembangkitan seluruh animasi visual AinoSentinel."""
    tx_df = pd.read_parquet("data/raw/transactions_stream.parquet")
    ensemble = HybridStackingEnsemble.load("models/hybrid_lgbm.pkl")
    metrics = json.loads(Path("reports/benchmark_metrics.json").read_text(encoding="utf-8"))

    p1 = generate_pipeline_gif(tx_df, ensemble, metrics)
    p2 = generate_cost_sweep_gif()
    p3 = generate_interactive_html_animation(tx_df)

    print(f"[Animation] Berhasil membuat animasi pipeline: {p1} ({p1.stat().st_size // 1024} KB)")
    print(f"[Animation] Berhasil membuat animasi kurva biaya: {p2} ({p2.stat().st_size // 1024} KB)")
    print(f"[Animation] Berhasil membuat animasi interaktif HTML: {p3} ({p3.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    import plotly.express as px
    main()
