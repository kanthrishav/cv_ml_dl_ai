# household_classifier.py
# Wrapper around your MobileNetV2 household-items classifier (PyTorch).

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


DEFAULT_LABELS = [
    # Replace with YOUR household label set in index order used for training
    "mug", "cup", "bottle", "bowl", "plate", "spoon", "fork",
    "phone", "laptop", "book", "remote", "scissors", "toothbrush",
    "knife", "keyboard", "mouse", "headphones", "wallet", "keys"
]


class HouseholdClassifier:
    def __init__(self, ckpt_path: Path, labels_path: Path = None, input_size=224, device="cpu"):
        self.device = device
        self.input_size = input_size
        self.model = self._load_model(ckpt_path).to(self.device).eval()
        self.labels = self._load_labels(labels_path)

        self.tf = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225])
        ])

    def _load_labels(self, labels_path):
        if labels_path is not None and Path(labels_path).exists():
            with open(labels_path, "r") as f:
                labs = [ln.strip() for ln in f if ln.strip()]
            if len(labs) > 0:
                return labs
        return DEFAULT_LABELS

    def _load_model(self, ckpt_path: Path):
        # Expect a scripted or state_dict MobileNetV2 consistent with your training
        ckpt_path = Path(ckpt_path)
        try:
            # Try TorchScript first
            model = torch.jit.load(str(ckpt_path), map_location="cpu")
            return model
        except Exception:
            # Fall back to state_dict load; build vanilla MobileNetV2 head
            import torchvision.models as models
            base = models.mobilenet_v2(weights=None)  # your head was fine-tuned
            num_feats = base.classifier[1].in_features
            base.classifier[1] = nn.Linear(num_feats, len(DEFAULT_LABELS))
            sd = torch.load(str(ckpt_path), map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
                # Optionally strip prefixes if trained under DataParallel
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            base.load_state_dict(sd, strict=False)
            return base

    def classify(self, roi_bgr: np.ndarray, topk=1):
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0
        # Convert to PIL RGB
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.tf(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)
            label = self.labels[top_idx.item()] if top_idx.item() < len(self.labels) else str(top_idx.item())
            return label, float(top_prob.item() * 100.0)
