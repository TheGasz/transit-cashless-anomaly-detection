"""Tier 3: Hybrid Stacking Ensemble (Isolation Forest + PyTorch VAE Meta-Features + Calibrated LightGBM + SHAP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
import joblib
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from src.features.velocity_features import FEATURE_COLUMNS
from src.ingestion.temporal_splitter import ZeroLeakageScalerPipeline
from src.models.pytorch_vae import PyTorchVAEAnomalyDetector, VAEAnomalyOutput
from src.models.rule_guardrails import DeterministicRuleGuardrail


META_FEATURE_NAMES: List[str] = [
    "vae_reconstruction_error",
    "vae_max_feature_error",
    "isolation_forest_score",
]

STACKED_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + META_FEATURE_NAMES

SUBTYPE_CLASSES: List[str] = [
    "NORMAL",
    "DOUBLE_TAP_GLITCH",
    "IMPOSSIBLE_TRAVEL_CLONE",
    "FARE_EVASION_INCOMPLETE_OUT",
    "SETTLEMENT_BATCH_ANOMALY",
]


class HybridStackingEnsemble:
    """Arsitektur AinoSentinel Hybrid 3-Tier Anomaly Engine (Guardrail + VAE + IF + Calibrated LightGBM)."""

    def __init__(
        self,
        random_state: int = 42,
        if_n_estimators: int = 150,
        if_contamination: float = 0.015,
        lgbm_n_estimators: int = 240,
        lgbm_learning_rate: float = 0.05,
        vae_epochs: int = 18,
    ) -> None:
        self.random_state = random_state
        self.if_n_estimators = if_n_estimators
        self.if_contamination = if_contamination
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_learning_rate = lgbm_learning_rate
        self.vae_epochs = vae_epochs

        self.scaler = ZeroLeakageScalerPipeline(FEATURE_COLUMNS, fit_only_normal=True)
        self.guardrail = DeterministicRuleGuardrail()
        self.vae_detector = PyTorchVAEAnomalyDetector(
            input_dim=len(FEATURE_COLUMNS),
            hidden_dim=64,
            latent_dim=12,
            beta_kl=0.05,
            epochs=vae_epochs,
            seed=random_state,
        )
        self.isolation_forest = IsolationForest(
            n_estimators=if_n_estimators,
            max_samples=2048,
            contamination=if_contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.binary_lgbm = LGBMClassifier(
            n_estimators=lgbm_n_estimators,
            learning_rate=lgbm_learning_rate,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight={0: 1.0, 1: 4.5},
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.subtype_lgbm = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=23,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.isotonic_calibrator = IsotonicRegression(y_min=0.0001, y_max=0.9999, out_of_bounds="clip")
        self.if_min_score: float = 0.35
        self.if_max_score: float = 0.75
        self.optimal_threshold: float = 0.44
        self.is_fitted: bool = False

    def _compute_if_anomaly_score(self, x_scaled: np.ndarray) -> np.ndarray:
        """Menghitung skor anomali Isolation Forest (semakin tinggi semakin anomali)."""
        if len(x_scaled) == 1:
            return np.array([self._fast_score_single_if(x_scaled[0])], dtype=np.float64)

        prev_jobs = self.isolation_forest.n_jobs
        if len(x_scaled) <= 32:
            self.isolation_forest.n_jobs = 1
        try:
            raw_if = -self.isolation_forest.score_samples(x_scaled)
        finally:
            self.isolation_forest.n_jobs = prev_jobs
        return raw_if.astype(np.float64)

    def _fast_score_single_if(self, x_row: np.ndarray) -> float:
        """Evaluasi pohon Isolation Forest secara langsung tanpa overhead validasi sklearn (< 3 ms)."""
        if not hasattr(self, "_if_cached_trees"):
            from sklearn.ensemble._iforest import _average_path_length

            estimators = self.isolation_forest.estimators_
            feat_list = self.isolation_forest.estimators_features_
            max_samples = self.isolation_forest._max_samples
            avg_max = float(_average_path_length(np.array([max_samples]))[0])

            t_left = [t.tree_.children_left for t in estimators]
            t_right = [t.tree_.children_right for t in estimators]
            t_feat = [feat_list[i][t.tree_.feature] for i, t in enumerate(estimators)]
            t_thresh = [t.tree_.threshold for t in estimators]
            t_samples = [t.tree_.n_node_samples for t in estimators]

            apl_lookup = np.zeros(2049, dtype=np.float64)
            for n in range(2, 2049):
                apl_lookup[n] = 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

            self._if_cached_trees = (t_left, t_right, t_feat, t_thresh, t_samples, apl_lookup, avg_max)

        t_left, t_right, t_feat, t_thresh, t_samples, apl_lookup, avg_max = self._if_cached_trees
        paths = 0.0
        n_trees = len(t_left)

        for i in range(n_trees):
            l = t_left[i]
            r = t_right[i]
            f = t_feat[i]
            th = t_thresh[i]
            s = t_samples[i]

            node = 0
            depth = 0
            while l[node] != -1:
                if x_row[f[node]] <= th[node]:
                    node = l[node]
                else:
                    node = r[node]
                depth += 1
            n_samp = s[node]
            if n_samp > 1:
                paths += depth + apl_lookup[min(n_samp, 2048)]
            else:
                paths += depth

        return float(2.0 ** (-(paths / n_trees) / avg_max))

    def predict_if_proba_unsupervised(self, x_scaled: np.ndarray) -> np.ndarray:
        """Mengonversi skor Isolation Forest menjadi pseudo-probabilitas [0, 1] untuk benchmark."""
        raw = self._compute_if_anomaly_score(x_scaled)
        span = max(self.if_max_score - self.if_min_score, 1e-4)
        norm = (raw - self.if_min_score) / span
        return np.clip(norm, 0.0, 1.0)

    def build_stacked_features(
        self,
        df_or_x: pd.DataFrame | np.ndarray,
        return_details: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, VAEAnomalyOutput, np.ndarray]:
        """Membangun matriks 31 fitur (28 fitur spasio-temporal + 3 sinyal meta VAE & IF)."""
        if isinstance(df_or_x, pd.DataFrame):
            x_raw = df_or_x[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        else:
            x_raw = np.asarray(df_or_x, dtype=np.float64)
            if x_raw.ndim == 1:
                x_raw = x_raw.reshape(1, -1)

        x_scaled = self.scaler.transform(x_raw)
        vae_out = self.vae_detector.score_details(x_scaled)
        if_scores = self._compute_if_anomaly_score(x_scaled)

        meta_cols = np.column_stack(
            [
                vae_out.total_anomaly_score,
                vae_out.max_feature_error,
                if_scores,
            ]
        )
        x_stacked = np.hstack([x_raw, meta_cols]).astype(np.float64)
        if return_details:
            return x_stacked, vae_out, if_scores
        return x_stacked

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
    ) -> "HybridStackingEnsemble":
        """Melatih seluruh komponen (Scaler, PyTorch VAE pada data normal, IF, LightGBM, dan Isotonic Calibrator)."""
        y_train = train_df["is_anomaly"].to_numpy(dtype=np.int64)

        # 1. Fit RobustScaler murni pada transaksi normal di train split
        self.scaler.fit(train_df)
        x_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        x_train_scaled = self.scaler.transform(x_train_raw)

        # 2. Latih PyTorch Residual VAE murni pada transaksi normal di train split
        if verbose:
            print("[HybridEnsemble] Melatih PyTorch Residual VAE pada data normal train split...")
        self.vae_detector.fit_normal(x_train_scaled, y_train=y_train, verbose=verbose)

        # 3. Latih Isolation Forest hanya pada train split
        if verbose:
            print("[HybridEnsemble] Melatih Isolation Forest pada train split...")
        self.isolation_forest.fit(x_train_scaled)
        train_if_scores = self._compute_if_anomaly_score(x_train_scaled)
        self.if_min_score = float(np.quantile(train_if_scores, 0.02))
        self.if_max_score = float(np.quantile(train_if_scores, 0.995))

        # 4. Bangun matriks fitur meta (31 dimensi) pada train split
        x_train_stacked = self.build_stacked_features(x_train_raw)

        # 5. Latih LightGBM biner & kalibrasi probabilitas Isotonic
        if verbose:
            print("[HybridEnsemble] Melatih Stacked LightGBM & Isotonic Probability Calibrator...")
        self.binary_lgbm.fit(x_train_stacked, y_train)

        if val_df is not None and len(val_df) > 50:
            x_val_raw = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            y_val = val_df["is_anomaly"].to_numpy(dtype=np.int64)
            x_val_stacked, vae_val, _ = self.build_stacked_features(x_val_raw, return_details=True)
            raw_val_prob = self._raw_combined_score(x_val_stacked, vae_val)
            self.isotonic_calibrator.fit(raw_val_prob, y_val)
        else:
            # Gunakan out-of-fold prediksi pada train split untuk kalibrasi
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            oof_probs = np.zeros(len(y_train), dtype=np.float64)
            for tr_idx, hold_idx in skf.split(x_train_stacked, y_train):
                fold_clf = LGBMClassifier(
                    n_estimators=120,
                    learning_rate=self.lgbm_learning_rate,
                    num_leaves=31,
                    max_depth=6,
                    class_weight={0: 1.0, 1: 4.5},
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
                fold_clf.fit(x_train_stacked[tr_idx], y_train[tr_idx])
                oof_probs[hold_idx] = fold_clf.predict_proba(x_train_stacked[hold_idx])[:, 1]
            self.isotonic_calibrator.fit(oof_probs, y_train)

        # 6. Latih model klasifikasi subtipe anomali
        if "anomaly_category" in train_df.columns:
            subtypes = train_df["anomaly_category"].astype(str).to_numpy()
            known_mask = np.isin(subtypes, SUBTYPE_CLASSES)
            self.subtype_lgbm.fit(x_train_stacked[known_mask], subtypes[known_mask])

        self.is_fitted = True
        return self

    def _raw_combined_score(self, x_stacked: np.ndarray, vae_out: VAEAnomalyOutput) -> np.ndarray:
        """Menggabungkan probabilitas LightGBM dengan sinyal zero-day VAE & Guardrail."""
        lgbm_prob = np.asarray(self.binary_lgbm.booster_.predict(x_stacked), dtype=np.float64)

        # Sinyal tak-terawasi dari VAE untuk menangkap pola Zero-Day yang belum pernah dilatih
        vae_unsupervised_prob = np.clip(
            (vae_out.total_anomaly_score - self.vae_detector.normal_score_q99)
            / max(2.2 * self.vae_detector.normal_score_q99, 0.25),
            0.0,
            0.96,
        )

        # Sinyal deterministik Guardrail untuk hard double-tap & impossible travel
        dt_col = x_stacked[:, FEATURE_COLUMNS.index("delta_t_card_sec")]
        same_rd = x_stacked[:, FEATURE_COLUMNS.index("is_same_reader")]
        same_tp = x_stacked[:, FEATURE_COLUMNS.index("is_same_tap_type")]
        hard_dt = (dt_col <= 8.0) & (same_rd > 0.5) & (same_tp > 0.5)

        combined = 0.86 * lgbm_prob + 0.14 * vae_unsupervised_prob
        combined = np.where(vae_unsupervised_prob > 0.82, np.maximum(combined, vae_unsupervised_prob), combined)
        combined = np.where(hard_dt, np.maximum(combined, 0.985), combined)
        return np.clip(combined, 0.0, 1.0)

    def predict_proba_from_stacked(self, x_stacked: np.ndarray, vae_out: VAEAnomalyOutput) -> np.ndarray:
        """Menghitung probabilitas anomali terkalibrasi dari x_stacked & vae_out tanpa ekstraksi ulang."""
        raw_prob = self._raw_combined_score(x_stacked, vae_out)
        cal_prob = self.isotonic_calibrator.predict(raw_prob)

        dt_col = x_stacked[:, FEATURE_COLUMNS.index("delta_t_card_sec")]
        same_rd = x_stacked[:, FEATURE_COLUMNS.index("is_same_reader")]
        same_tp = x_stacked[:, FEATURE_COLUMNS.index("is_same_tap_type")]
        hard_dt = (dt_col <= 8.0) & (same_rd > 0.5) & (same_tp > 0.5)

        extreme_vae = vae_out.total_anomaly_score > (3.2 * max(self.vae_detector.normal_score_q99, 0.2))
        cal_prob = np.where(hard_dt, np.maximum(cal_prob, 0.98), cal_prob)
        cal_prob = np.where(extreme_vae, np.maximum(cal_prob, 0.88), cal_prob)
        return np.clip(cal_prob, 0.0, 1.0).astype(np.float64)

    def predict_proba(self, df_or_x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Menghasilkan probabilitas anomali terkalibrasi p_anom in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("HybridStackingEnsemble belum dilatih!")

        x_stacked, vae_out, _ = self.build_stacked_features(df_or_x, return_details=True)
        return self.predict_proba_from_stacked(x_stacked, vae_out)

    def classify_subtype_and_action(
        self,
        x_stacked_row: np.ndarray,
        prob_anom: float,
        vae_error: float,
        threshold: Optional[float] = None,
    ) -> Tuple[str, str, str]:
        """Menentukan subtipe anomali, tingkat keparahan (severity), dan tindakan auto-remediasi."""
        tau = threshold if threshold is not None else self.optimal_threshold

        if prob_anom < tau:
            return "NORMAL", "LOW", "APPROVE_NORMAL"

        dt_sec = float(x_stacked_row[FEATURE_COLUMNS.index("delta_t_card_sec")])
        same_reader = float(x_stacked_row[FEATURE_COLUMNS.index("is_same_reader")])
        same_tap = float(x_stacked_row[FEATURE_COLUMNS.index("is_same_tap_type")])
        dist_km = float(x_stacked_row[FEATURE_COLUMNS.index("delta_d_km")])
        vel_kmh = float(x_stacked_row[FEATURE_COLUMNS.index("velocity_kmh")])
        consec_in = float(x_stacked_row[FEATURE_COLUMNS.index("consecutive_tapin_count")])
        inc_ratio = float(x_stacked_row[FEATURE_COLUMNS.index("incomplete_tapout_ratio_7d")])
        is_offline = float(x_stacked_row[FEATURE_COLUMNS.index("is_offline_sync_int")])
        sync_delay = float(x_stacked_row[FEATURE_COLUMNS.index("sync_delay_seconds")])

        # 1. Double-Tap Glitch
        if dt_sec <= 12.0 and same_reader > 0.5 and same_tap > 0.5:
            category = "DOUBLE_TAP_GLITCH"
        # 2. Impossible Travel / Card Cloning
        elif dist_km >= 2.5 and vel_kmh > 120.0:
            category = "IMPOSSIBLE_TRAVEL_CLONE"
        # 3. Fare Evasion / Incomplete Tap-Out
        elif consec_in >= 2.0 or (inc_ratio >= 0.45 and same_stop_or_passback(x_stacked_row)):
            category = "FARE_EVASION_INCOMPLETE_OUT"
        # 4. Settlement Batch / Offline Anomaly vs Zero-Day Novel Pattern
        elif is_offline > 0.5 and sync_delay > 7200.0:
            category = "SETTLEMENT_BATCH_ANOMALY"
        elif vae_error > 2.2 * max(self.vae_detector.normal_score_q99, 0.2):
            category = "NOVEL_ZERO_DAY_PATTERN"
        else:
            pred_sub = str(self.subtype_lgbm.predict(x_stacked_row.reshape(1, -1))[0])
            category = pred_sub if pred_sub != "NORMAL" else "NOVEL_ZERO_DAY_PATTERN"

        # Cost-Sensitive Decision Router sesuai PRD Bagian 3.1 Tahap 4 & Bagian 7
        if category == "DOUBLE_TAP_GLITCH":
            action = "AUTO_REFUND_DUPLICATE" if prob_anom >= 0.85 else "APPROVE_NORMAL"
            severity = "HIGH" if prob_anom >= 0.85 else "MEDIUM"
        elif category == "IMPOSSIBLE_TRAVEL_CLONE":
            action = "TEMP_FREEZE_TOKEN_AND_ALERT"
            severity = "CRITICAL"
        elif category == "FARE_EVASION_INCOMPLETE_OUT":
            action = "APPLY_MAX_CORRIDOR_PENALTY_FARE"
            severity = "HIGH"
        elif category == "SETTLEMENT_BATCH_ANOMALY":
            action = "HOLD_BATCH_SETTLEMENT"
            severity = "HIGH"
        else:
            action = "TEMP_FREEZE_TOKEN_AND_ALERT" if prob_anom >= 0.80 else "HOLD_BATCH_SETTLEMENT"
            severity = "CRITICAL" if prob_anom >= 0.85 else "HIGH"

        return category, severity, action

    def explain_single_shap(
        self,
        x_stacked_row: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, float | str]]:
        """Menghitung kontribusi lokal fitur menggunakan TreeSHAP bawaan LightGBM (< 0.3 ms)."""
        row_2d = np.asarray(x_stacked_row, dtype=np.float64).reshape(1, -1)
        # pred_contrib=True mengembalikan nilai SHAP eksak untuk setiap fitur + kolom bias di akhir
        shap_contrib = self.binary_lgbm.booster_.predict(row_2d, pred_contrib=True)[0][:-1]

        top_indices = np.argsort(np.abs(shap_contrib))[::-1][:top_k]
        contributions: List[Dict[str, float | str]] = []
        for idx in top_indices:
            i = int(idx)
            contributions.append(
                {
                    "feature_name": STACKED_FEATURE_COLUMNS[i],
                    "feature_value": round(float(row_2d[0, i]), 4),
                    "shap_or_z_impact": round(float(shap_contrib[i]), 4),
                }
            )
        return contributions

    def save(self, lgbm_path: str | Path, vae_path: Optional[str | Path] = None, scaler_path: Optional[str | Path] = None) -> None:
        """Menyimpan seluruh artefak model terlatih ke direktori models/."""
        target = Path(lgbm_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if vae_path is not None:
            self.vae_detector.save(vae_path)
        if scaler_path is not None:
            self.scaler.save(scaler_path)

        joblib.dump(
            {
                "scaler": self.scaler,
                "isolation_forest": self.isolation_forest,
                "binary_lgbm": self.binary_lgbm,
                "subtype_lgbm": self.subtype_lgbm,
                "isotonic_calibrator": self.isotonic_calibrator,
                "if_min_score": self.if_min_score,
                "if_max_score": self.if_max_score,
                "optimal_threshold": self.optimal_threshold,
                "vae_state_dict": self.vae_detector.model.state_dict(),
                "vae_config": {
                    "input_dim": self.vae_detector.input_dim,
                    "hidden_dim": self.vae_detector.hidden_dim,
                    "latent_dim": self.vae_detector.latent_dim,
                    "dropout": self.vae_detector.dropout,
                    "beta_kl": self.vae_detector.beta_kl,
                },
                "vae_q50": self.vae_detector.normal_score_q50,
                "vae_q99": self.vae_detector.normal_score_q99,
            },
            target,
        )

    @classmethod
    def load(cls, lgbm_path: str | Path) -> "HybridStackingEnsemble":
        """Memuat seluruh pipeline HybridStackingEnsemble dari berkas .pkl."""
        payload = joblib.load(lgbm_path)
        ens = cls()
        ens.scaler = payload["scaler"]
        ens.isolation_forest = payload["isolation_forest"]
        ens.binary_lgbm = payload["binary_lgbm"]
        ens.subtype_lgbm = payload["subtype_lgbm"]
        ens.isotonic_calibrator = payload["isotonic_calibrator"]
        ens.if_min_score = float(payload["if_min_score"])
        ens.if_max_score = float(payload["if_max_score"])
        ens.optimal_threshold = float(payload.get("optimal_threshold", 0.44))

        v_cfg = payload["vae_config"]
        ens.vae_detector = PyTorchVAEAnomalyDetector(
            input_dim=v_cfg["input_dim"],
            hidden_dim=v_cfg["hidden_dim"],
            latent_dim=v_cfg["latent_dim"],
            dropout=v_cfg.get("dropout", 0.10),
            beta_kl=v_cfg.get("beta_kl", 0.05),
        )
        ens.vae_detector.model.load_state_dict(payload["vae_state_dict"])
        ens.vae_detector.model.eval()
        ens.vae_detector.normal_score_q50 = float(payload.get("vae_q50", 0.1))
        ens.vae_detector.normal_score_q99 = float(payload.get("vae_q99", 1.0))
        ens.is_fitted = True
        return ens


def same_stop_or_passback(x_row: np.ndarray) -> bool:
    """Helper untuk mengecek indikasi pass-back di halte yang sama."""
    passback_cnt = float(x_row[FEATURE_COLUMNS.index("passback_same_stop_count_5m")])
    fare_ratio = float(x_row[FEATURE_COLUMNS.index("fare_to_corridor_std_ratio")])
    return passback_cnt >= 2.0 or fare_ratio < 0.5
