"""AinoSentinel 4-Tab Real-Time Operations Center Dashboard (Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.velocity_features import FEATURE_COLUMNS, OnlineVelocityFeatureExtractor
from src.models.hybrid_ensemble import STACKED_FEATURE_COLUMNS, HybridStackingEnsemble
from src.reconciliation.cost_optimizer import BusinessCostThresholdOptimizer
from src.reconciliation.settlement_auditor import DailySettlementReconciliationAuditor


st.set_page_config(
    page_title="AinoSentinel — Cashless Anomaly & Settlement Ops Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_dashboard_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Memuat sampel transaksi, tabel audit settlement, metadata halte, dan metrik benchmark."""
    tx_path = Path("data/raw/transactions_stream.parquet")
    batches_path = Path("data/raw/daily_settlement_batches.csv")
    stops_path = Path("data/raw/stops_metadata.csv")
    metrics_path = Path("reports/benchmark_metrics.json")

    if not tx_path.exists() or not batches_path.exists():
        from data.generate_transit_stream import main as gen_main
        gen_main()

    if not metrics_path.exists():
        from src.models.evaluate_models import main as eval_main
        eval_main()

    tx_df = pd.read_parquet(tx_path)
    batches_df = pd.read_csv(batches_path)
    stops_df = pd.read_csv(stops_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    audit_path = Path("reports/settlement_audit_table.csv")
    if audit_path.exists():
        audit_df = pd.read_csv(audit_path)
    else:
        auditor = DailySettlementReconciliationAuditor()
        audit_df = auditor.audit_batches(batches_df).audit_dataframe

    return tx_df, audit_df, stops_df, metrics


@st.cache_resource(show_spinner=False)
def load_trained_ensemble() -> HybridStackingEnsemble:
    """Memuat model HybridStackingEnsemble satu kali ke dalam cache Streamlit."""
    model_path = Path("models/hybrid_lgbm.pkl")
    if not model_path.exists():
        from src.models.evaluate_models import main as eval_main
        eval_main()
    return HybridStackingEnsemble.load(model_path)


def render_dashboard() -> None:
    """Merender ke-4 tab operasional sesuai spesifikasi PRD Bagian 7.1."""
    tx_df, audit_df, stops_df, metrics = load_dashboard_datasets()
    ensemble = load_trained_ensemble()
    headline = metrics["headline_metrics"]

    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0F172A 0%, #1E293B 60%, #0F766E 100%);
                    padding: 18px 24px; border-radius: 12px; color: white; margin-bottom: 16px;">
            <h2 style="margin:0; color:white;">🛡️ AinoSentinel: Real-Time Cashless Transit Anomaly & Settlement Ops Center</h2>
            <p style="margin:4px 0 0 0; opacity:0.9; font-size:0.95rem;">
                PT Aino Indonesia Production Blueprint — Hybrid 3-Tier Engine (Deterministic Guardrails + PyTorch Residual VAE + LightGBM Stacking)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Filter & Kontrol Operasional")
        selected_corridors = st.multiselect(
            "Filter Koridor Transportasi",
            options=sorted(tx_df["corridor_id"].unique().tolist()),
            default=sorted(tx_df["corridor_id"].unique().tolist()),
        )
        selected_channels = st.multiselect(
            "Filter Kanal Pembayaran",
            options=["E_MONEY_BANK", "KMT", "QRIS_TAP"],
            default=["E_MONEY_BANK", "KMT", "QRIS_TAP"],
        )
        st.divider()
        st.caption("**Ringkasan Model Produksi Aktif**")
        st.metric("Hybrid PR-AUC (Test)", f"{headline['hybrid_pr_auc']:.4f}")
        st.metric("F2-Score (Imbalanced)", f"{headline['hybrid_f2_score']:.4f}")
        st.metric("Recall @ FPR ≤ 1%", f"{headline['hybrid_recall_at_fpr_1pct']:.4f}")
        st.metric("Threshold Optimal (τ*)", f"{headline['optimal_threshold_tau']:.4f}")

    filtered_tx = tx_df[
        tx_df["corridor_id"].isin(selected_corridors)
        & tx_df["payment_channel"].isin(selected_channels)
    ]

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗺️ Tab 1: Command Center & Geo-Anomaly Map",
            "🔍 Tab 2: Double-Tap & Velocity Forensics",
            "📑 Tab 3: Daily Settlement & Batch Auditor",
            "📈 Tab 4: Model Benchmark & Cost-Curve Explorer",
        ]
    )

    # =========================================================================
    # TAB 1: REAL-TIME TRANSIT COMMAND CENTER & GEO-ANOMALY MAP
    # =========================================================================
    with tab1:
        total_tx = len(filtered_tx)
        anom_tx = int((filtered_tx["is_anomaly"] == 1).sum())
        anom_rate = 100.0 * anom_tx / max(total_tx, 1)
        dt_count = int((filtered_tx["anomaly_category"] == "DOUBLE_TAP_GLITCH").sum())
        saved_idr = dt_count * 4000.0 + (anom_tx - dt_count) * 45000.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transaksi Dipantau", f"{total_tx:,}", "14 Hari Multi-Koridor")
        c2.metric("Anomali Terdeteksi", f"{anom_tx:,}", f"{anom_rate:.2f}% dari volume")
        c3.metric("Double-Tap Auto-Refund", f"{dt_count:,} Tiket", f"Precision {headline['double_tap_precision']*100:.1f}%")
        c4.metric("Total Nilai Diselamatkan", f"Rp {saved_idr:,.0f}", f"Hemat {headline['net_cost_savings_pct']:.1f}% Biaya")

        col_map, col_sim = st.columns([1.65, 1.0])

        with col_map:
            st.subheader("📍 Peta Interaktif Halte Koridor & Titik Insiden Anomali")
            stop_stats = (
                filtered_tx.groupby(["stop_id", "corridor_id", "latitude", "longitude"], as_index=False)
                .agg(
                    total_tap=("tx_id", "count"),
                    anomaly_count=("is_anomaly", "sum"),
                    mean_latency_ms=("reader_latency_ms", "mean"),
                    max_velocity_kmh=("velocity_kmh", "max"),
                )
            )
            stop_stats["anomaly_ratio_pct"] = (
                100.0 * stop_stats["anomaly_count"] / np.maximum(stop_stats["total_tap"], 1)
            ).round(2)

            fig_map = px.scatter_map(
                stop_stats,
                lat="latitude",
                lon="longitude",
                color="anomaly_ratio_pct",
                size="anomaly_count",
                hover_name="stop_id",
                hover_data={
                    "corridor_id": True,
                    "total_tap": True,
                    "anomaly_count": True,
                    "mean_latency_ms": ":.1f",
                    "max_velocity_kmh": ":.1f",
                },
                color_continuous_scale="Turbo",
                size_max=22,
                zoom=6.2,
                center={"lat": -6.85, "lon": 108.2},
                map_style="carto-positron",
                height=440,
            )
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_map, width="stretch")

        with col_sim:
            st.subheader("⚡ Simulasi Uji Inferensi Langsung")
            sim_scenario = st.selectbox(
                "Pilih Skenario Transaksi Uji:",
                options=[
                    "1. Double-Tap Reader Glitch (Δt = 3 detik, Reader Sama)",
                    "2. Impossible Travel Card Clone (Δd = 39 km dalam 22 detik)",
                    "3. Fare Evasion Repeated Tap-In (Pass-Back Tanpa Tap-Out)",
                    "4. Normal Commuter Tap-In (Halte Blok M)",
                ],
            )
            if st.button("🚀 Jalankan Inferensi 3-Tier Engine", type="primary", width="stretch"):
                extractor = OnlineVelocityFeatureExtractor()
                base_ev = {
                    "tx_id": "TX-99000001",
                    "timestamp": "2026-10-14T08:15:00",
                    "card_pan_hash": "SIMCARD9988776655443322",
                    "user_segment": "REGULER",
                    "payment_channel": "QRIS_TAP",
                    "tap_type": "TAP_IN",
                    "corridor_id": "COR-01",
                    "stop_id": "STOP-001",
                    "latitude": -6.2442,
                    "longitude": 106.8006,
                    "reader_device_id": "TOB-001",
                    "fare_amount": 3500.0,
                    "reader_latency_ms": 95.0,
                    "is_offline_sync": False,
                }
                _ = extractor.extract_single(base_ev, update_state=True)

                if sim_scenario.startswith("1."):
                    test_ev = {**base_ev, "tx_id": "TX-99000002", "timestamp": "2026-10-14T08:15:03", "reader_latency_ms": 1350.0}
                elif sim_scenario.startswith("2."):
                    test_ev = {
                        **base_ev,
                        "tx_id": "TX-99000002",
                        "timestamp": "2026-10-14T08:15:22",
                        "corridor_id": "COR-02",
                        "stop_id": "STOP-010",
                        "latitude": -6.5950,
                        "longitude": 106.7905,
                        "reader_device_id": "TOB-015",
                        "fare_amount": 6000.0,
                        "reader_latency_ms": 260.0,
                    }
                elif sim_scenario.startswith("3."):
                    for s_i in range(2):
                        _ = extractor.extract_single(
                            {**base_ev, "tx_id": f"TX-9900001{s_i}", "timestamp": f"2026-10-14T08:15:{15 + s_i*20:02d}", "fare_amount": 800.0},
                            update_state=True,
                        )
                    test_ev = {**base_ev, "tx_id": "TX-99000002", "timestamp": "2026-10-14T08:15:55", "fare_amount": 800.0, "reader_latency_ms": 290.0}
                else:
                    test_ev = {
                        **base_ev,
                        "tx_id": "TX-99000002",
                        "timestamp": "2026-10-14T08:35:00",
                        "tap_type": "TAP_OUT",
                        "stop_id": "STOP-005",
                        "latitude": -6.1900,
                        "longitude": 106.8070,
                        "reader_device_id": "TOB-005",
                    }

                vec = extractor.extract_single(test_ev, update_state=True).reshape(1, -1)
                x_stk, vae_o, if_s = ensemble.build_stacked_features(vec, return_details=True)
                p_anom = float(ensemble.predict_proba_from_stacked(x_stk, vae_o)[0])
                cat, sev, act = ensemble.classify_subtype_and_action(x_stk[0], p_anom, float(vae_o.total_anomaly_score[0]))

                if p_anom >= ensemble.optimal_threshold:
                    st.error(f"**ANOMALI TERDETEKSI ({cat})** — Probabilitas: `{p_anom:.4f}` | Severity: `{sev}`")
                else:
                    st.success(f"**TRANSAKSI NORMAL** — Probabilitas Anomali: `{p_anom:.4f}` (< τ* `{ensemble.optimal_threshold}`)")

                st.json(
                    {
                        "tx_id": test_ev["tx_id"],
                        "anomaly_probability": round(p_anom, 4),
                        "anomaly_category": cat,
                        "recommended_action": act,
                        "vae_reconstruction_error": round(float(vae_o.total_anomaly_score[0]), 4),
                        "computed_velocity_kmh": round(float(vec[0, FEATURE_COLUMNS.index("velocity_kmh")]), 2),
                    }
                )

        gif_path = Path("reports/ainosentinel_pipeline_animation.gif")
        if gif_path.exists():
            with st.expander("🎬 Lihat Animasi Visual End-to-End Pipeline AinoSentinel", expanded=False):
                st.image(str(gif_path), caption="Animasi Deteksi Anomali Real-Time, VAE Reconstruction & Optimasi Biaya AinoSentinel", width="stretch")

    # =========================================================================
    # TAB 2: DOUBLE-TAP & CARD VELOCITY FORENSICS INSPECTOR
    # =========================================================================
    with tab2:
        st.subheader("🔬 Forensik Perjalanan Kartu, Kontribusi Fitur SHAP & Galat Rekonstruksi PyTorch VAE")
        anom_samples = filtered_tx[filtered_tx["is_anomaly"] == 1].head(150)
        if anom_samples.empty:
            anom_samples = tx_df[tx_df["is_anomaly"] == 1].head(150)

        options_label = [
            f"{row.tx_id} | {row.anomaly_category} | Card: {row.card_pan_hash[:10]}... | v={row.velocity_kmh:.1f} km/h | Δt={row.delta_t_card_sec:.0f}s"
            for _, row in anom_samples.iterrows()
        ]
        selected_label = st.selectbox("Pilih Transaksi Mencurigakan untuk Audit Forensik:", options=options_label)
        selected_tx_id = selected_label.split(" | ")[0]

        target_row = tx_df[tx_df["tx_id"] == selected_tx_id].iloc[0]
        target_card = str(target_row["card_pan_hash"])
        card_timeline = tx_df[tx_df["card_pan_hash"] == target_card].sort_values("timestamp").reset_index(drop=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kategori Patologi", str(target_row["anomaly_category"]))
        m2.metric("Selisih Waktu Tap (Δt)", f"{float(target_row['delta_t_card_sec']):.1f} detik")
        m3.metric("Jarak Haversine (Δd)", f"{float(target_row['delta_d_km']):.2f} km")
        m4.metric("Kecepatan Implisit", f"{float(target_row['velocity_kmh']):.1f} km/jam")

        x_row_2d = target_row[FEATURE_COLUMNS].to_numpy(dtype=np.float64).reshape(1, -1)
        x_stacked, vae_detail, _ = ensemble.build_stacked_features(x_row_2d, return_details=True)
        shap_contribs = ensemble.explain_single_shap(x_stacked[0], top_k=8)

        col_shap, col_vae = st.columns(2)
        with col_shap:
            shap_df = pd.DataFrame(shap_contribs).iloc[::-1]
            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["shap_or_z_impact"],
                    y=shap_df["feature_name"],
                    orientation="h",
                    marker_color=[
                        "#DC2626" if val > 0 else "#2563EB" for val in shap_df["shap_or_z_impact"]
                    ],
                    text=[f"{v:+.3f} (val={fv})" for v, fv in zip(shap_df["shap_or_z_impact"], shap_df["feature_value"])],
                    textposition="auto",
                )
            )
            fig_shap.update_layout(
                title=f"Kontribusi Fitur Lokal TreeSHAP ({selected_tx_id})",
                xaxis_title="Dampak Log-Odds SHAP terhadap Anomali",
                yaxis_title="Fitur",
                height=380,
                margin={"l": 10, "r": 10, "t": 40, "b": 20},
            )
            st.plotly_chart(fig_shap, width="stretch")

        with col_vae:
            feat_errs = vae_detail.per_feature_abs_error[0]
            top_vae_idx = np.argsort(feat_errs)[::-1][:8][::-1]
            fig_vae = go.Figure(
                go.Bar(
                    x=feat_errs[top_vae_idx],
                    y=[FEATURE_COLUMNS[i] for i in top_vae_idx],
                    orientation="h",
                    marker_color="#7C3AED",
                    text=[f"{feat_errs[i]:.3f}" for i in top_vae_idx],
                    textposition="auto",
                )
            )
            fig_vae.update_layout(
                title=f"Galat Rekonstruksi per Fitur PyTorch VAE (Total={vae_detail.total_anomaly_score[0]:.3f})",
                xaxis_title="|x_j - x̂_j| (Standardized Reconstruction Error)",
                yaxis_title="Fitur Spasio-Temporal",
                height=380,
                margin={"l": 10, "r": 10, "t": 40, "b": 20},
            )
            st.plotly_chart(fig_vae, width="stretch")

        st.markdown(f"**Riwayat Perjalanan Lengkap Kartu `{target_card}` ({len(card_timeline)} Transaksi):**")
        display_cols = [
            "tx_id",
            "timestamp",
            "corridor_id",
            "stop_id",
            "reader_device_id",
            "tap_type",
            "fare_amount",
            "delta_t_card_sec",
            "delta_d_km",
            "velocity_kmh",
            "reader_latency_ms",
            "anomaly_category",
        ]
        st.dataframe(card_timeline[display_cols], width="stretch", hide_index=True)

    # =========================================================================
    # TAB 3: DAILY SETTLEMENT RECONCILIATION & OFFLINE BATCH AUDITOR
    # =========================================================================
    with tab3:
        st.subheader("🏦 Audit Rekonsiliasi Batch Settlement Harian (Robust MAD Z-Score |Z| > 3.5)")
        flagged_df = audit_df[audit_df["is_flagged_discrepancy"] == 1].copy()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Batch Harian Diaudit", f"{len(audit_df):,}", "120 TOB × 14 Hari")
        s2.metric("Batch Bermasalah Ditandai", f"{len(flagged_df):,}", "100% Recall Selisih > Rp50rb")
        s3.metric("Total Selisih Kas Tertangkap", f"Rp {flagged_df['abs_discrepancy_idr'].sum():,.0f}", "Ditahan Otomatis")
        s4.metric("Rata-rata Max MAD Z-Score", f"{flagged_df['robust_mad_zscore'].median():.1f}", "Ambang |Z| > 3.5")

        col_b1, col_b2 = st.columns([1.2, 1.0])
        with col_b1:
            fig_mad = px.scatter(
                audit_df,
                x="sync_delay_minutes",
                y="abs_discrepancy_idr",
                color="audit_status",
                size="robust_mad_zscore",
                hover_data=["batch_id", "reader_device_id", "acquirer_bank", "missing_packet_count"],
                title="Peta Sebaran Selisih Settlement (IDR) vs Keterlambatan Sinkronisasi Offline",
                height=380,
            )
            st.plotly_chart(fig_mad, width="stretch")

        with col_b2:
            bank_summary = (
                flagged_df.groupby("acquirer_bank", as_index=False)
                .agg(
                    flagged_batches=("batch_id", "count"),
                    total_discrepancy_idr=("abs_discrepancy_idr", "sum"),
                )
            )
            fig_bank = px.bar(
                bank_summary,
                x="acquirer_bank",
                y="total_discrepancy_idr",
                color="acquirer_bank",
                text_auto=".3s",
                title="Total Selisih Kas per Bank Mitra (Acquirer)",
                height=380,
            )
            st.plotly_chart(fig_bank, width="stretch")

        st.markdown("**Daftar Tiket Rekonsiliasi Batch Settlement yang Ditahan (`is_flagged_discrepancy == 1`):**")
        audit_cols = [
            "reconciliation_ticket_id",
            "batch_id",
            "settlement_date",
            "reader_device_id",
            "acquirer_bank",
            "device_tx_count",
            "settled_tx_count",
            "signed_discrepancy_idr",
            "missing_packet_ratio",
            "robust_mad_zscore",
            "audit_status",
            "recommended_action",
        ]
        st.dataframe(
            flagged_df.sort_values("abs_discrepancy_idr", ascending=False)[audit_cols],
            width="stretch",
            hide_index=True,
        )

    # =========================================================================
    # TAB 4: MODEL BENCHMARK, ABLATION & BUSINESS COST-CURVE EXPLORER
    # =========================================================================
    with tab4:
        st.subheader("📊 Studi Ablasi 5 Model, Kurva PR-AUC & Simulator Matriks Biaya Bisnis")

        col_pr, col_cat = st.columns(2)
        with col_pr:
            pr_fig = go.Figure()
            for model_name, curve in metrics["pr_curves"].items():
                pr_fig.add_trace(
                    go.Scatter(
                        x=curve["recall"],
                        y=curve["precision"],
                        mode="lines",
                        name=model_name,
                        line={"width": 3 if "AinoSentinel" in model_name else 2},
                    )
                )
            pr_fig.update_layout(
                title="Kurva Precision-Recall (PR-AUC) ke-5 Arsitektur pada Test Split",
                xaxis_title="Recall (Cakupan Deteksi Anomali)",
                yaxis_title="Precision (Ketepatan Prediksi)",
                height=400,
                legend={"orientation": "h", "y": -0.25},
            )
            st.plotly_chart(pr_fig, width="stretch")

        with col_cat:
            cat_rows = []
            for cat_name, vals in metrics["per_category_recall"].items():
                cat_rows.append({"Kategori Anomali": cat_name, "Arsitektur": "Static Rule Engine", "Recall": vals["static_rule_recall"]})
                cat_rows.append({"Kategori Anomali": cat_name, "Arsitektur": "AinoSentinel Hybrid (VAE+IF+LGBM)", "Recall": vals["ainosentinel_hybrid_recall"]})
            cat_df = pd.DataFrame(cat_rows)
            fig_cat = px.bar(
                cat_df,
                x="Recall",
                y="Kategori Anomali",
                color="Arsitektur",
                barmode="group",
                orientation="h",
                text_auto=".2f",
                title="Perbandingan Detection Recall per Skenario (Termasuk Zero-Day Novel Pattern)",
                height=400,
            )
            st.plotly_chart(fig_cat, width="stretch")

        st.markdown("---")
        st.markdown("### 💰 Simulator Interaktif Kalibrasi Threshold Biaya Bisnis (Cost-Matrix Explorer)")
        sl1, sl2 = st.columns(2)
        user_cost_fn = sl1.slider(
            "Asumsi Biaya Kerugian Fraud / Kebocoran per FN (Rp):",
            min_value=10000,
            max_value=150000,
            value=45000,
            step=5000,
        )
        user_cost_fp = sl2.slider(
            "Asumsi Biaya Investigasi / Komplain Pelanggan per FP (Rp):",
            min_value=1000,
            max_value=30000,
            value=6500,
            step=500,
        )

        # Hitung kurva biaya interaktif pada sampel evaluasi
        opt_path = Path("reports/optimal_threshold.json")
        opt_data = json.loads(opt_path.read_text(encoding="utf-8"))
        base_fn_rate = np.array(opt_data["fn_costs_idr"]) / max(opt_data["cost_fraud_fn_idr"], 1.0)
        base_fp_rate = np.array(opt_data["fp_costs_idr"]) / max(opt_data["cost_false_alarm_fp_idr"], 1.0)
        taus = np.array(opt_data["thresholds"])

        sim_fn_cost = base_fn_rate * user_cost_fn
        sim_fp_cost = base_fp_rate * user_cost_fp
        sim_tot_cost = sim_fn_cost + sim_fp_cost
        best_i = int(np.argmin(sim_tot_cost))
        best_tau = float(taus[best_i])
        min_cost_idr = float(sim_tot_cost[best_i])
        no_filter_idr = float(base_fn_rate[-1] * user_cost_fn) if base_fn_rate[-1] > 0 else float(base_fn_rate.max() * user_cost_fn)
        savings_pct = 100.0 * max(0.0, no_filter_idr - min_cost_idr) / max(no_filter_idr, 1.0)

        col_cost_chart, col_vae_dist = st.columns(2)
        with col_cost_chart:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(x=taus, y=sim_tot_cost / 1e6, name="Total Biaya (FN + FP)", line={"color": "#0F766E", "width": 3}))
            fig_cost.add_trace(go.Scatter(x=taus, y=sim_fn_cost / 1e6, name="Biaya Kerugian Fraud (FN)", line={"color": "#DC2626", "dash": "dash"}))
            fig_cost.add_trace(go.Scatter(x=taus, y=sim_fp_cost / 1e6, name="Biaya Alarm Palsu (FP)", line={"color": "#2563EB", "dash": "dot"}))
            fig_cost.add_vline(x=best_tau, line_dash="dashdot", line_color="#D97706", annotation_text=f"Optimal τ* = {best_tau:.2f} (Hemat {savings_pct:.1f}%)")
            fig_cost.update_layout(
                title=f"Kurva Optimasi Biaya Bisnis (τ* = {best_tau:.2f} | Min Biaya: Rp {min_cost_idr:,.0f})",
                xaxis_title="Ambang Batas Probabilitas Anomali (Threshold τ)",
                yaxis_title="Estimasi Biaya (Juta IDR)",
                height=380,
            )
            st.plotly_chart(fig_cost, width="stretch")

        with col_vae_dist:
            vae_dist = metrics["vae_error_distribution"]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=vae_dist["normal_sample"], name="Transaksi Normal", marker_color="#10B981", opacity=0.75))
            fig_hist.add_trace(go.Histogram(x=vae_dist["anomaly_sample"], name="Transaksi Anomali", marker_color="#EF4444", opacity=0.75))
            fig_hist.update_layout(
                barmode="overlay",
                title="Distribusi Galat Rekonstruksi PyTorch VAE (Normal vs Anomali)",
                xaxis_title="Skor Rekonstruksi e_VAE(x) (Log-Scale Visual)",
                yaxis_title="Frekuensi Sampel",
                height=380,
            )
            st.plotly_chart(fig_hist, width="stretch")

        st.markdown("**Tabel Benchmark Komparasi 5 Model pada Test Split:**")
        st.dataframe(pd.DataFrame(metrics["models_comparison"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    render_dashboard()
