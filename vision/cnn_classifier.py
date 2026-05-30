from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from vision.ts2image import series_to_rgb

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CNNTimeSeriesClassifier(nn.Module):
    """CNN classifier that operates on time-series-to-image transformations.

    Maps 1D time series to RGB images via GAF/MTF, then feeds a pretrained
    ResNet/EfficientNet backbone for classification or anomaly scoring.
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50", "efficientnet_b0"] = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.backbone.fc = nn.Identity()
            feat_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.backbone.fc = nn.Identity()
            feat_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights="IMAGENET1K_V1" if pretrained else None
            )
            self.backbone.classifier = nn.Identity()
            feat_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """x_rgb: (B, 3, H, W) -> (B, num_classes)"""
        features = self.backbone(x_rgb)
        return self.head(features)


class ImageAnomalyScorer:
    """Use a pretrained CNN to score anomaly degree of industrial images.

    Computes the distance between query image features and a reference set
    of normal images, using Mahalanobis distance or cosine similarity.
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50", "wide_resnet50_2"] = "wide_resnet50_2",
    ):
        if backbone == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(weights="IMAGENET1K_V2")
            self.model.fc = nn.Identity()
            self.feat_dim = 2048
        elif backbone == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            self.model.fc = nn.Identity()
            self.feat_dim = 2048
        else:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            self.model.fc = nn.Identity()
            self.feat_dim = 512

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._reference_mean: np.ndarray | None = None
        self._reference_cov_inv: np.ndarray | None = None

    def extract_features(self, images: list[np.ndarray]) -> np.ndarray:
        """Extract CNN features from a list of (H, W, 3) images."""
        features = []
        with torch.no_grad():
            for img in images:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.model(tensor).cpu().numpy()
                features.append(feat)
        return np.concatenate(features, axis=0)

    def fit_reference(self, normal_images: list[np.ndarray]) -> None:
        """Build reference distribution from normal images."""
        features = self.extract_features(normal_images)
        self._reference_mean = features.mean(axis=0)
        cov = np.cov(features, rowvar=False)
        reg = 1e-6 * np.eye(cov.shape[0])
        self._reference_cov_inv = np.linalg.inv(cov + reg)
        logger.info("Fitted reference on %d normal images", len(normal_images))

    def mahalanobis_score(self, image: np.ndarray) -> float:
        """Higher score -> more anomalous."""
        if self._reference_mean is None or self._reference_cov_inv is None:
            raise RuntimeError("fit_reference() must be called first")
        feat = self.extract_features([image])[0]
        diff = feat - self._reference_mean
        return float(np.sqrt(diff @ self._reference_cov_inv @ diff))

    def score_series_image(self, x: np.ndarray, image_size: int = 224) -> float:
        """Score a 1D time series by converting to image then evaluating anomaly."""
        rgb = series_to_rgb(x, image_size=image_size)  # (3, H, W)
        img = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
        return self.mahalanobis_score(img)
