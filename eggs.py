"""
Clownfish Egg Counter — Final v2
==================================
v6 tuned + saturation filter to remove false positives on pale gaps.
"""

import cv2
import numpy as np
from scipy import ndimage
from pathlib import Path
import argparse

def count_eggs(image_path, nms_size=11, percentile=55, sigma_large=8.0, sigma_small=1.5, output_dir=None, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    H, W = img.shape[:2]
    if debug: print(f"Image: {W}x{H}")

    # 1. Orange mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 70, 50], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)
    orange_mask = cv2.inRange(hsv, lower, upper)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, k3, iterations=1)

    # 2. Crop to bounding box of ALL orange regions (not just largest)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(orange_mask, connectivity=8)
    if num <= 1: return 0, []
    # Get bounding box that covers all orange components (skip background label 0)
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
    # Raw mask (no morph) to preserve tiny isolated eggs for signal
    mask_raw = cv2.inRange(hsv_c, lower, upper)
    ch, cw = crop.shape[:2]
    if debug: print(f"Crop: {cw}x{ch}, orange px: {cv2.countNonZero(mask_c)}")

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

    # 5. Post-filter — reject peaks on pale/white gaps
    #    Real egg centers have: high saturation + high A-channel (orange)
    #    False positives sit on washed-out pale areas between eggs
    sat_ch = hsv_c[:, :, 1]  # saturation channel
    a_raw = lab[:, :, 1]     # raw A-channel (not bandpass)
    check_r = 4  # check neighborhood roughly half an egg
    centers = []
    rejected = 0
    for (cy, cx) in raw_centers:
        iy, ix = int(round(cy)), int(round(cx))
        y1_ = max(0, iy - check_r)
        y2_ = min(ch, iy + check_r + 1)
        x1_ = max(0, ix - check_r)
        x2_ = min(cw, ix + check_r + 1)
        # Check saturation: real eggs are solidly colored, not pale
        local_sat = sat_ch[y1_:y2_, x1_:x2_]
        mean_sat = np.mean(local_sat)
        # Check A-channel: real egg centers are orange (high A), gaps are neutral
        local_a = a_raw[y1_:y2_, x1_:x2_]
        mean_a = np.mean(local_a)
        # Check orange mask coverage: most of the neighborhood should be orange
        # Check raw HSV at center (not the morphologically-opened mask which erodes small eggs)
        h_val = hsv_c[iy, ix, 0]
        s_val = hsv_c[iy, ix, 1]
        v_val = hsv_c[iy, ix, 2]
        is_orange = (5 <= h_val <= 30) and (s_val >= 50) and (v_val >= 50)
        if mean_sat >= 70 and mean_a >= 140 and is_orange:
            centers.append((cy, cx))
        else:
            rejected += 1

    count = len(centers)
    if debug: print(f"Raw peaks: {len(raw_centers)}, rejected (low sat): {rejected}, final: {count}")

    # 6. Visualizations
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        overlay = crop.copy()
        for (cy, cx) in centers:
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, (0, 255, 0), -1)
        text = f"Count: {count}"
        (tw, th2), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(overlay, (5, 5), (15 + tw, 15 + th2 + 5), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 10 + th2), font, 0.8, (0, 255, 0), 2)
        cv2.imwrite(f"{output_dir}/egg_overlay.png", overlay)

        bp_min = bandpass[mask_raw > 0].min()
        bp_max = bandpass[mask_raw > 0].max()
        bp_vis = ((bandpass - bp_min) / (bp_max - bp_min + 1e-8) * 255).astype(np.uint8)
        bp_vis[mask_raw == 0] = 0
        cv2.imwrite(f"{output_dir}/bandpass.png", bp_vis)

        for name, ry, rx in [("center", ch//3, cw//4), ("edge", ch//6, 0), ("right", ch//3, cw//2), ("bottom", ch//2, cw//4)]:
            sz = min(300, ch - ry, cw - rx)
            if sz >= 100:
                z = cv2.resize(overlay[ry:ry+sz, rx:rx+sz], (900, 900), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"{output_dir}/zoom_{name}.png", z)

    return count, [(cy + y0, cx + x0) for (cy, cx) in centers]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count clownfish eggs")
    parser.add_argument("image", nargs="?", default="eggsimg.jpeg")
    parser.add_argument("--nms", type=int, default=11)
    parser.add_argument("--pctl", type=int, default=55)
    parser.add_argument("--sig-large", type=float, default=8.0)
    parser.add_argument("--sig-small", type=float, default=1.5)
    parser.add_argument("--output", "-o", default="egg_output/v5")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'sig_L':>5} | {'NMS':>3} | {'pctl':>4} | {'count':>5}")
        print("-" * 32)
        for sl in [8.0, 10.0, 12.0]:
            for nms in [9, 11]:
                for pctl in [50, 55, 60, 65]:
                    c, _ = count_eggs(args.image, nms_size=nms, percentile=pctl, sigma_large=sl)
                    print(f"{sl:5.1f} | {nms:3d} | {pctl:4d} | {c:5d}")
        print()

    count, centers = count_eggs(args.image, args.nms, args.pctl, args.sig_large, args.sig_small, args.output, debug=True)
    print(f"\n  Egg count: {count}")