"""Small digit-image encoder with a size-based depth prior and learned correction."""

import torch
from torch import nn


class StickerDepthNet(nn.Module):
    def __init__(self, feature_mean, feature_std, log_height_prior, classes):
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.register_buffer("log_height_prior", torch.tensor(float(log_height_prior)))
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.shared = nn.Sequential(nn.Linear(53, 48), nn.ReLU(), nn.Dropout(.10))
        self.depth_correction = nn.Linear(48, 1)
        self.classifier = nn.Linear(48, classes)
        nn.init.zeros_(self.depth_correction.weight)
        nn.init.zeros_(self.depth_correction.bias)

    def forward(self, image, geometry):
        visual = self.encoder(image)
        features = (geometry-self.feature_mean)/self.feature_std
        hidden = self.shared(torch.cat([visual, features], dim=1))
        correction = self.depth_correction(hidden).squeeze(1).clamp(-1.5, 1.5)
        # geometry[:,1] is log(h/fy), measured from the student digit box.
        depth = torch.exp(self.log_height_prior-geometry[:, 1]+correction)
        return depth, self.classifier(hidden)
