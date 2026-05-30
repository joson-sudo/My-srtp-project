import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def vision_inspect(
    image_path: str,
    task: str = "anomaly_score",
    reference_dir: Optional[str] = None,
) -> str:
    """Analyze industrial image for defects or anomalies.

    Args:
        image_path: Path to image file (jpg, png, etc.).
        task: "anomaly_score" or "classify".
        reference_dir: Optional directory of normal reference images.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        result = {
            "status": "success",
            "image_path": image_path,
            "dimensions": list(img_array.shape),
            "task": task,
        }

        if task == "anomaly_score":
            # Use pretrained CNN feature extraction for anomaly scoring
            try:
                from vision.cnn_classifier import ImageAnomalyScorer
                scorer = ImageAnomalyScorer(backbone="resnet18")

                if reference_dir:
                    import os
                    from pathlib import Path
                    ref_images = []
                    ref_path = Path(reference_dir)
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                        for f in ref_path.glob(ext):
                            ref_img = np.array(Image.open(f).convert("RGB"))
                            ref_images.append(ref_img)
                    if ref_images:
                        scorer.fit_reference(ref_images)
                        result["score"] = scorer.mahalanobis_score(img_array)
                        result["reference_count"] = len(ref_images)
                    else:
                        result["score"] = 0.0
                        result["warning"] = "No reference images found"
                else:
                    # Without reference, return basic stats
                    result["score"] = 0.0
                    result["warning"] = "No reference set provided"

            except ImportError:
                result["score"] = 0.0
                result["warning"] = "PyTorch/torchvision not available"

        elif task == "classify":
            # Statistical feature analysis as fallback
            stats = {
                "mean_r": float(img_array[:, :, 0].mean()),
                "mean_g": float(img_array[:, :, 1].mean()),
                "mean_b": float(img_array[:, :, 2].mean()),
                "std_r": float(img_array[:, :, 0].std()),
                "std_g": float(img_array[:, :, 1].std()),
                "std_b": float(img_array[:, :, 2].std()),
            }
            result["color_stats"] = stats
            # Simple brightness-based defect heuristic
            brightness = img_array.mean()
            result["brightness_mean"] = float(brightness)
            result["potential_defect"] = brightness < 30 or brightness > 240

        return json.dumps(result)

    except Exception as e:
        logger.exception("vision_inspect failed")
        return json.dumps({"status": "error", "message": str(e)})


def multimodal_diagnosis(
    data_path: str,
    column: str,
    image_path: Optional[str] = None,
    forecast_steps: int = 10,
) -> str:
    """Run cross-modal industrial diagnosis combining sensor data and vision.

    Args:
        data_path: Path to CSV sensor data.
        column: Target sensor column.
        image_path: Optional path to equipment image.
        forecast_steps: Number of forecast steps.
    """
    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(data_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"Column {column} not found."})

        series = df[column].dropna().values

        # Run multimodal agent
        from fusion.multimodal_agent import MultimodalAgent
        agent = MultimodalAgent()

        ts_results = agent.run_time_series_pipeline(
            series.reshape(-1, 1), steps=forecast_steps
        )
        vision_results = agent.run_vision_pipeline(series_data=series)
        fusion_results = agent.run_fusion_pipeline(ts_results, vision_results)

        report = agent.generate_report(ts_results, vision_results, fusion_results)

        return report

    except Exception as e:
        logger.exception("multimodal_diagnosis failed")
        return json.dumps({"status": "error", "message": str(e)})
