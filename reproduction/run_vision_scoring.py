"""Run vision anomaly scoring on synthetic industrial images."""
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
from PIL import Image
from vision.cnn_classifier import ImageAnomalyScorer

scorer = ImageAnomalyScorer(backbone="resnet18")

# Load normal reference images (as PIL Images, not numpy arrays)
normal_dir = PROJECT_ROOT / "reproduction" / "data" / "vision" / "normal"
normal_images = [Image.open(f).convert("RGB") for f in sorted(normal_dir.glob("*.png"))[:10]]
scorer.fit_reference(normal_images)

# Score defect images
defect_dir = PROJECT_ROOT / "reproduction" / "data" / "vision" / "defect"
results = {}
for d_type in sorted(defect_dir.iterdir()):
    if d_type.is_dir():
        imgs = sorted(d_type.glob("*.png"))
        scores = []
        for f in imgs[:3]:
            img = Image.open(f).convert("RGB")
            score = round(scorer.mahalanobis_score(img), 2)
            scores.append(score)
        results[d_type.name] = {"scores": scores, "mean": round(np.mean(scores), 2)}

# Score some normal images
normal_scores = [round(scorer.mahalanobis_score(img), 2) for img in normal_images[:3]]
results["normal_reference"] = {"scores": normal_scores, "mean": round(np.mean(normal_scores), 2)}

out = PROJECT_ROOT / "reproduction" / "results" / "etth1_benchmark" / "traces" / "vision_scores.json"
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("Vision anomaly scores saved to:", out)
print(json.dumps(results, indent=2, ensure_ascii=False))
