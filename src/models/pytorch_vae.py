"""Tier 2: Arsitektur PyTorch Residual Deep Autoencoder (DAE) & Variational Autoencoder (VAE)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock1D(nn.Module):
    """Blok fully-connected dengan koneksi residual, BatchNorm1d, GELU, dan Dropout."""

    def __init__(self, dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)


class ResidualDeepAutoencoder(nn.Module):
    """PyTorch Residual Deep Autoencoder (DAE: 28 -> 64 -> 16 -> 64 -> 28) untuk studi ablasi."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.encoder_res = ResidualBlock1D(hidden_dim, dropout=dropout)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.decoder_in = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.decoder_res = ResidualBlock1D(hidden_dim, dropout=dropout)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_res(self.encoder_in(x))
        z = self.bottleneck(h)
        d = self.decoder_res(self.decoder_in(z))
        x_hat = self.decoder_out(d)
        return x_hat, z


class ResidualVariationalAutoencoder(nn.Module):
    """PyTorch Residual Variational Autoencoder (VAE) dengan loss gabungan MSE + beta * KL."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 64,
        latent_dim: int = 12,
        dropout: float = 0.10,
        beta_kl: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_kl = beta_kl

        self.encoder_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.encoder_res = ResidualBlock1D(hidden_dim, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_in = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.decoder_res = ResidualBlock1D(hidden_dim, dropout=dropout)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memetakan masukan x ke parameter distribusi Gaussian laten (mu, logvar)."""
        h = self.encoder_res(self.encoder_in(x))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Trik reparameterisasi z = mu + sigma * eps saat training, dan deterministik mu saat eval."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Merekonstruksi vektor fitur x_hat dari vektor laten z."""
        d = self.decoder_res(self.decoder_in(z))
        return self.decoder_out(d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass mengembalikan (x_hat, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss_function(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Menghitung fungsi rugi beta-VAE sesuai spesifikasi matematis PRD Bagian 5.1:
    L_VAE(x) = (1/D) sum_j (x_j - x_hat_j)^2 - (beta/2) sum_k (1 + log(sigma_k^2) - mu_k^2 - sigma_k^2).
    """
    recon_mse_per_sample = torch.mean((x - x_hat) ** 2, dim=1)
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    total_loss = torch.mean(recon_mse_per_sample + beta * kl_per_sample)
    return total_loss, torch.mean(recon_mse_per_sample), torch.mean(kl_per_sample)


@dataclass
class VAEAnomalyOutput:
    """Keluaran sinyal anomali tak-terawasi dari PyTorch VAE."""

    total_anomaly_score: np.ndarray
    mse_error: np.ndarray
    max_feature_error: np.ndarray
    kl_divergence: np.ndarray
    per_feature_abs_error: np.ndarray
    latent_mu: np.ndarray


class PyTorchVAEAnomalyDetector:
    """Pembungkus pelatihan dan inferensi cepat (torch.inference_mode) untuk ResidualVAE."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 64,
        latent_dim: int = 12,
        dropout: float = 0.10,
        beta_kl: float = 0.05,
        learning_rate: float = 1.5e-3,
        batch_size: int = 512,
        epochs: int = 18,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_kl = beta_kl
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(self.seed)
        self.model = ResidualVariationalAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            beta_kl=beta_kl,
        ).to(self.device)
        self.train_history: List[Dict[str, float]] = []
        self.normal_score_q50: float = 0.1
        self.normal_score_q99: float = 1.0
        self.last_used_inference_mode: bool = False

    def fit_normal(
        self,
        x_train_scaled: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "PyTorchVAEAnomalyDetector":
        """Melatih model VAE murni pada transaksi NORMAL (y_train == 0) di train split."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if y_train is not None:
            normal_mask = np.asarray(y_train) == 0
            x_normal = np.asarray(x_train_scaled[normal_mask], dtype=np.float32)
        else:
            x_normal = np.asarray(x_train_scaled, dtype=np.float32)

        if len(x_normal) < 8:
            raise ValueError("Jumlah sampel normal terlalu sedikit untuk melatih PyTorch VAE.")

        dataset = TensorDataset(torch.from_numpy(x_normal))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(x_normal)),
            shuffle=True,
            drop_last=len(x_normal) > self.batch_size,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))

        self.model.train()
        self.train_history.clear()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                x_hat, mu, logvar = self.model(batch_x)
                loss, mse_val, kl_val = vae_loss_function(
                    batch_x, x_hat, mu, logvar, beta=self.beta_kl
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_mse += float(mse_val.item())
                epoch_kl += float(kl_val.item())
                n_batches += 1

            scheduler.step()
            metrics = {
                "epoch": float(epoch),
                "total_loss": epoch_loss / max(1, n_batches),
                "mse_loss": epoch_mse / max(1, n_batches),
                "kl_loss": epoch_kl / max(1, n_batches),
            }
            self.train_history.append(metrics)
            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.epochs):
                print(
                    f"  [PyTorchVAE] Epoch {epoch:02d}/{self.epochs:02d} | "
                    f"Loss: {metrics['total_loss']:.4f} (MSE: {metrics['mse_loss']:.4f}, KL: {metrics['kl_loss']:.4f})"
                )

        # Kalibrasi kuantil skor rekonstruksi pada data normal latih
        normal_out = self.score_details(x_normal)
        self.normal_score_q50 = float(np.quantile(normal_out.total_anomaly_score, 0.50))
        self.normal_score_q99 = float(np.quantile(normal_out.total_anomaly_score, 0.99))
        return self

    def score_details(self, x_scaled: np.ndarray) -> VAEAnomalyOutput:
        """Menghitung galat rekonstruksi, galat fitur maksimum, dan KL-divergence di dalam torch.inference_mode()."""
        self.model.eval()
        x_arr = np.asarray(x_scaled, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        with torch.inference_mode():
            self.last_used_inference_mode = torch.is_inference_mode_enabled()
            x_tensor = torch.from_numpy(x_arr).to(self.device)
            x_hat, mu, logvar = self.model(x_tensor)

            diff_abs = torch.abs(x_tensor - x_hat)
            mse_err = torch.mean(diff_abs ** 2, dim=1)
            max_feat_err, _ = torch.max(diff_abs, dim=1)
            kl_div = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_score = mse_err + self.beta_kl * kl_div

            return VAEAnomalyOutput(
                total_anomaly_score=total_score.cpu().numpy().astype(np.float64),
                mse_error=mse_err.cpu().numpy().astype(np.float64),
                max_feature_error=max_feat_err.cpu().numpy().astype(np.float64),
                kl_divergence=kl_div.cpu().numpy().astype(np.float64),
                per_feature_abs_error=diff_abs.cpu().numpy().astype(np.float64),
                latent_mu=mu.cpu().numpy().astype(np.float64),
            )

    def predict_proba_unsupervised(self, x_scaled: np.ndarray) -> np.ndarray:
        """Mengonversi skor rekonstruksi VAE menjadi pseudo-probabilitas [0, 1] untuk evaluasi standalone."""
        details = self.score_details(x_scaled)
        raw = details.total_anomaly_score + 0.25 * details.max_feature_error
        scale = max(self.normal_score_q99 - self.normal_score_q50, 1e-4)
        z = (raw - self.normal_score_q99) / (0.65 * scale)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str | Path) -> None:
        """Menyimpan state_dict model dan metadata ke berkas .pt."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim,
                    "dropout": self.dropout,
                    "beta_kl": self.beta_kl,
                },
                "normal_score_q50": self.normal_score_q50,
                "normal_score_q99": self.normal_score_q99,
                "train_history": self.train_history,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PyTorchVAEAnomalyDetector":
        """Memuat bobot terlatih PyTorch VAE dari berkas .pt."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        detector = cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg.get("dropout", 0.10),
            beta_kl=cfg.get("beta_kl", 0.05),
            device=device,
        )
        detector.model.load_state_dict(checkpoint["state_dict"])
        detector.model.eval()
        detector.normal_score_q50 = float(checkpoint.get("normal_score_q50", 0.1))
        detector.normal_score_q99 = float(checkpoint.get("normal_score_q99", 1.0))
        detector.train_history = checkpoint.get("train_history", [])
        return detector


class PyTorchDAEAnomalyDetector:
    """Pembungkus pelatihan dan inferensi untuk ResidualDeepAutoencoder (tanpa regularisasi KL)."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        dropout: float = 0.10,
        learning_rate: float = 1.5e-3,
        batch_size: int = 512,
        epochs: int = 14,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.device = torch.device(device)

        torch.manual_seed(self.seed)
        self.model = ResidualDeepAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        ).to(self.device)
        self.normal_q50: float = 0.1
        self.normal_q99: float = 1.0

    def fit_normal(
        self,
        x_train_scaled: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> "PyTorchDAEAnomalyDetector":
        torch.manual_seed(self.seed)
        if y_train is not None:
            x_normal = np.asarray(x_train_scaled[np.asarray(y_train) == 0], dtype=np.float32)
        else:
            x_normal = np.asarray(x_train_scaled, dtype=np.float32)

        dataset = TensorDataset(torch.from_numpy(x_normal))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(x_normal)),
            shuffle=True,
            drop_last=len(x_normal) > self.batch_size,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                x_hat, _ = self.model(batch_x)
                loss = criterion(x_hat, batch_x)
                loss.backward()
                optimizer.step()

        scores = self.score_raw(x_normal)
        self.normal_q50 = float(np.quantile(scores, 0.50))
        self.normal_q99 = float(np.quantile(scores, 0.99))
        return self

    def score_raw(self, x_scaled: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_arr = np.asarray(x_scaled, dtype=np.float32)
        with torch.inference_mode():
            x_t = torch.from_numpy(x_arr).to(self.device)
            x_hat, _ = self.model(x_t)
            mse = torch.mean((x_t - x_hat) ** 2, dim=1)
            return mse.cpu().numpy().astype(np.float64)

    def predict_proba_unsupervised(self, x_scaled: np.ndarray) -> np.ndarray:
        raw = self.score_raw(x_scaled)
        scale = max(self.normal_q99 - self.normal_q50, 1e-4)
        z = np.clip((raw - self.normal_q99) / (0.75 * scale), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
