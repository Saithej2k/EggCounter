"""
Clownfish Egg Counter — Web App
================================
Flask backend that wraps the CV egg counter pipeline.

Run:  python app.py
Open: http://localhost:5000
"""

import cv2
import numpy as np
from scipy import ndimage
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import uuid
import os
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def count_eggs(image_path, nms_size=11, percentile=55, sigma_large=8.0, sigma_small=1.5):
    """Count clownfish eggs and return count + visualization paths."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image")
    H, W = img.shape[:2]

    # 1. Orange mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 70, 50], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)
    orange_mask = cv2.inRange(hsv, lower, upper)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # 2. Crop to bounding box of ALL orange regions
    num, labels, stats, _ = cv2.connectedComponentsWithStats(orange_mask, connectivity=8)
    if num <= 1:
        return 0, None, None, None
    all_x = stats[1:, cv2.CC_STAT_LEFT]
    all_y = stats[1:, cv2.CC_STAT_TOP]
    all_r = all_x + stats[1:, cv2.CC_STAT_WIDTH]
    all_b = all_y + stats[1:, cv2.CC_STAT_HEIGHT]
    margin = 30
    x0, y0 = max(0, all_x.min() - margin), max(0, all_y.min() - margin)
    x1, y1 = min(W, all_r.max() + margin), min(H, all_b.max() + margin)
    crop = img[y0:y1, x0:x1].copy()

    hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_c = cv2.inRange(hsv_c, lower, upper)
    mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, k3, iterations=1)
    mask_raw = cv2.inRange(hsv_c, lower, upper)
    ch, cw = crop.shape[:2]

    # 3. LAB A-channel -> Bandpass filter
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float64)
    lp_small = cv2.GaussianBlur(a_ch, (0, 0), sigma_small)
    lp_large = cv2.GaussianBlur(a_ch, (0, 0), sigma_large)
    bandpass = lp_small - lp_large
    bandpass[mask_raw == 0] = 0

    # 4. Peak detection
    signal = cv2.GaussianBlur(bandpass, (3, 3), 0.8)
    signal[mask_raw == 0] = 0
    vals = signal[mask_raw > 0]
    thresh = np.percentile(vals, percentile)
    local_max = ndimage.maximum_filter(signal, size=nms_size)
    peaks = (signal == local_max) & (signal > thresh) & (mask_raw > 0)
    labeled, num_features = ndimage.label(peaks)
    raw_centers = ndimage.center_of_mass(peaks, labeled, range(1, num_features + 1))

    # 5. Post-filter
    sat_ch = hsv_c[:, :, 1]
    a_raw = lab[:, :, 1]
    check_r = 4
    centers = []
    for (cy, cx) in raw_centers:
        iy, ix = int(round(cy)), int(round(cx))
        y1_ = max(0, iy - check_r)
        y2_ = min(ch, iy + check_r + 1)
        x1_ = max(0, ix - check_r)
        x2_ = min(cw, ix + check_r + 1)
        local_sat = sat_ch[y1_:y2_, x1_:x2_]
        mean_sat = np.mean(local_sat)
        local_a = a_raw[y1_:y2_, x1_:x2_]
        mean_a = np.mean(local_a)
        h_val = hsv_c[iy, ix, 0]
        s_val = hsv_c[iy, ix, 1]
        v_val = hsv_c[iy, ix, 2]
        is_orange = (5 <= h_val <= 30) and (s_val >= 50) and (v_val >= 50)
        if mean_sat >= 70 and mean_a >= 140 and is_orange:
            centers.append((cy, cx))

    count = len(centers)

    # 6. Generate output images
    run_id = str(uuid.uuid4())[:8]

    # Overlay
    overlay = crop.copy()
    for (cy, cx) in centers:
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), max(2, int(min(ch, cw) / 400)), (0, 255, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Count: {count}"
    scale = max(0.6, min(ch, cw) / 1000)
    thick = max(1, int(scale * 2))
    (tw, th2), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(overlay, (5, 5), (15 + tw, 15 + th2 + 5), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 10 + th2), font, scale, (0, 255, 0), thick)
    overlay_path = RESULTS_DIR / f"{run_id}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Bandpass
    bp_min = bandpass[mask_raw > 0].min()
    bp_max = bandpass[mask_raw > 0].max()
    bp_vis = ((bandpass - bp_min) / (bp_max - bp_min + 1e-8) * 255).astype(np.uint8)
    bp_vis[mask_raw == 0] = 0
    bandpass_path = RESULTS_DIR / f"{run_id}_bandpass.png"
    cv2.imwrite(str(bandpass_path), bp_vis)

    # Orange mask visualization
    mask_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
    mask_vis[mask_raw > 0] = [0, 140, 255]  # orange in BGR
    mask_path = RESULTS_DIR / f"{run_id}_mask.png"
    cv2.imwrite(str(mask_path), mask_vis)

    return count, overlay_path.name, bandpass_path.name, mask_path.name


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    ext = Path(file.filename).suffix or ".jpg"
    upload_path = RESULTS_DIR / f"upload_{uuid.uuid4().hex[:8]}{ext}"
    file.save(str(upload_path))

    try:
        count, overlay_name, bandpass_name, mask_name = count_eggs(upload_path)

        result = {"count": count}
        if overlay_name:
            result["overlay"] = img_to_base64(RESULTS_DIR / overlay_name)
            result["bandpass"] = img_to_base64(RESULTS_DIR / bandpass_name)
            result["mask"] = img_to_base64(RESULTS_DIR / mask_name)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup upload
        upload_path.unlink(missing_ok=True)


@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)