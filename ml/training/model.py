"""
ResNet-50 dual-head model for vehicle damage classification.

Architecture (blueprint Section 2.2):
- Backbone: ResNet-50 pretrained on ImageNet
- Head A: 8-class multi-label (sigmoid) — damage type presence
- Head B: 3-class single-label (softmax) — severity

Training strategy:
  Stage 1: freeze backbone, train heads only (5 epochs, lr=1e-3)
  Stage 2: unfreeze layer3+layer4, fine-tune (10-15 epochs, lr=1e-5)
"""

import torch
import torch.nn as nn
from torchvision import models

from ml.training.config import NUM_DAMAGE_CLASSES, NUM_SEVERITY_CLASSES


class DualHeadResNet50(nn.Module):
    """ResNet-50 backbone with two independent classification heads."""

    def __init__(
        self,
        num_damage_classes: int = NUM_DAMAGE_CLASSES,
        num_severity_classes: int = NUM_SEVERITY_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC head — keep everything up to AdaptiveAvgPool
        self.features = nn.Sequential(*list(backbone.children())[:-1])   # output: (B, 2048, 1, 1)

        in_features = backbone.fc.in_features   # 2048

        # Head A — multi-label damage type
        self.head_damage = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_damage_classes),
            # No sigmoid here — applied in forward() for training convenience
            # (BCEWithLogitsLoss expects raw logits)
        )

        # Head B — severity (single-label)
        self.head_severity = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_severity_classes),
            # No softmax here — CrossEntropyLoss expects raw logits
        )

    # ── Stage control ────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Stage 1: freeze all backbone parameters, keep heads trainable."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_stage2(self) -> None:
        """Stage 2: unfreeze layer3 and layer4 in the ResNet backbone."""
        # features is Sequential; ResNet children in order:
        # 0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4, 8:avgpool
        # We unfreeze indices 6 (layer3) and 7 (layer4).
        for i, child in enumerate(self.features.children()):
            if i >= 6:
                for param in child.parameters():
                    param.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            damage_logits  (B, NUM_DAMAGE_CLASSES)  — raw (pre-sigmoid)
            severity_logits (B, NUM_SEVERITY_CLASSES) — raw (pre-softmax)
        """
        features = self.features(x)            # (B, 2048, 1, 1)
        features = features.flatten(1)         # (B, 2048)

        damage_logits   = self.head_damage(features)
        severity_logits = self.head_severity(features)

        return damage_logits, severity_logits

    # ── Inference helper ──────────────────────────────────────────────────────

    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sigmoid probabilities for damage types and softmax probs for severity.
        Detaches from computation graph for inference use.
        """
        self.eval()
        with torch.no_grad():
            d_logits, s_logits = self.forward(x)
            damage_probs   = torch.sigmoid(d_logits)
            severity_probs = torch.softmax(s_logits, dim=-1)
        return damage_probs, severity_probs


def build_model(pretrained: bool = True) -> DualHeadResNet50:
    """Construct the model and apply Stage 1 freeze."""
    model = DualHeadResNet50(pretrained=pretrained)
    model.freeze_backbone()
    return model


def load_checkpoint(path: str | None = None) -> DualHeadResNet50:
    """Load a saved model checkpoint. Returns an eval-mode model."""
    from ml.training.config import CHECKPOINT_PATH
    checkpoint_path = path or CHECKPOINT_PATH

    model = DualHeadResNet50(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
