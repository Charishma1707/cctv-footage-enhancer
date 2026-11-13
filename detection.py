"""
detection.py
------------
Downstream Computer Vision Benchmark Module:
Evaluates object and person detection on RAW dark images vs. ENHANCED images
using a lightweight, high-performance NanoDet ONNX neural network.

Proves the practical utility of DIP enhancement in forensic & surveillance pipelines:
shows how enhancement recovers targets invisible to automated detectors.
"""

import os
import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nanodet.onnx")

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is a 3-channel BGR numpy array."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        return img
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class NanoDetector:
    def __init__(self, model_path: str = MODEL_PATH, prob_threshold: float = 0.30, iou_threshold: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detection model not found at {model_path}")

        self.image_shape = (416, 416)
        self.strides = (8, 16, 32)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        self.net = cv2.dnn.readNet(model_path)

        # Generate anchor grids
        self.anchors_mlvl = []
        for stride in self.strides:
            feat_h = int(self.image_shape[0] / stride)
            feat_w = int(self.image_shape[1] / stride)
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, self.image_shape)
        img_norm = (resized.astype(np.float32) - self.mean) / self.std
        return cv2.dnn.blobFromImage(img_norm)

    def detect(self, img: np.ndarray, prob_thresh: float | None = None) -> list[dict]:
        """
        Run inference on image (BGR format).
        Returns list of dicts: {'class_id': int, 'class_name': str, 'score': float, 'box': [x1, y1, x2, y2]}
        Coordinates are scaled to original input image dimensions.
        """
        img = to_bgr(img)
        thresh = prob_thresh if prob_thresh is not None else self.prob_threshold
        orig_h, orig_w = img.shape[:2]

        blob = self.pre_process(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Layer outputs: first 3 are cls scores, next 3 are bbox regressions
        cls_scores = outs[:3]
        bbox_preds = outs[3:]

        det_bboxes = []
        det_conf = []
        det_class = []

        scale_x = orig_w / float(self.image_shape[1])
        scale_y = orig_h / float(self.image_shape[0])

        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
                bbox_pred = bbox_pred.squeeze(axis=0)

            # Softmax on regression predictions
            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            bbox_pred = x_exp / np.sum(x_exp, axis=1, keepdims=True)
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4) * stride

            # Filter candidates above threshold
            max_scores = cls_score.max(axis=1)
            classes = cls_score.argmax(axis=1)
            valid_mask = max_scores >= thresh

            if not np.any(valid_mask):
                continue

            valid_scores = max_scores[valid_mask]
            valid_classes = classes[valid_mask]
            valid_anchors = anchors[valid_mask]
            valid_dist = bbox_pred[valid_mask]

            x1 = np.clip(valid_anchors[:, 0] - valid_dist[:, 0], 0, self.image_shape[1]) * scale_x
            y1 = np.clip(valid_anchors[:, 1] - valid_dist[:, 1], 0, self.image_shape[0]) * scale_y
            x2 = np.clip(valid_anchors[:, 0] + valid_dist[:, 2], 0, self.image_shape[1]) * scale_x
            y2 = np.clip(valid_anchors[:, 1] + valid_dist[:, 3], 0, self.image_shape[0]) * scale_y

            for b_x1, b_y1, b_x2, b_y2, s, c in zip(x1, y1, x2, y2, valid_scores, valid_classes):
                det_bboxes.append([int(b_x1), int(b_y1), int(b_x2 - b_x1), int(b_y2 - b_y1)])
                det_conf.append(float(s))
                det_class.append(int(c))

        if not det_bboxes:
            return []

        # Run Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(det_bboxes, det_conf, thresh, self.iou_threshold)

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = det_bboxes[idx]
                cls_id = det_class[idx]
                c_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                results.append({
                    "class_id": cls_id,
                    "class_name": c_name,
                    "score": round(det_conf[idx], 3),
                    "box": [box[0], box[1], box[0] + box[2], box[1] + box[3]]  # [x1, y1, x2, y2]
                })

        return results


def draw_detections(img: np.ndarray, detections: list[dict], color=(0, 255, 0), banner_title=""):
    """Render bounding boxes and confidence badges on an image."""
    out = to_bgr(img).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name'].upper()} {int(det['score'] * 100)}%"

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Draw filled label badge
        (tw, th), baseline = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 3)), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top banner indicator
    h, w = out.shape[:2]
    badge_text = f"{banner_title}: {len(detections)} TARGETS IDENTIFIED"
    cv2.putText(out, badge_text, (10, 22), font, 0.55, color, 2, cv2.LINE_AA)

    return out


def evaluate_vision_benchmark(raw_img: np.ndarray,
                              enhanced_img: np.ndarray,
                              prob_thresh: float = 0.28) -> dict:
    """
    Run complete downstream object detection comparison.
    Returns annotated images and statistical audit report.
    """
    detector = NanoDetector(prob_threshold=prob_thresh)

    raw_bgr = to_bgr(raw_img)
    enh_bgr = to_bgr(enhanced_img)

    raw_dets = detector.detect(raw_bgr, prob_thresh)
    enh_dets = detector.detect(enh_bgr, prob_thresh)

    annotated_raw = draw_detections(raw_bgr, raw_dets, color=(0, 0, 255), banner_title="RAW DARK FOOTAGE")
    annotated_enh = draw_detections(enh_bgr, enh_dets, color=(0, 255, 64), banner_title="ENHANCED FOOTAGE")

    raw_confs = [d["score"] for d in raw_dets]
    enh_confs = [d["score"] for d in enh_dets]

    avg_raw_conf = round(float(np.mean(raw_confs)), 3) if raw_confs else 0.0
    avg_enh_conf = round(float(np.mean(enh_confs)), 3) if enh_confs else 0.0

    targets_recovered = max(0, len(enh_dets) - len(raw_dets))

    return {
        "annotated_raw": annotated_raw,
        "annotated_enhanced": annotated_enh,
        "raw_count": len(raw_dets),
        "enhanced_count": len(enh_dets),
        "raw_detections": raw_dets,
        "enhanced_detections": enh_dets,
        "raw_avg_confidence": avg_raw_conf,
        "enhanced_avg_confidence": avg_enh_conf,
        "targets_recovered": targets_recovered,
        "operational_status": "SUCCESS" if len(enh_dets) > len(raw_dets) else ("OPTIMAL" if len(enh_dets) > 0 else "NO_TARGETS"),
    }

