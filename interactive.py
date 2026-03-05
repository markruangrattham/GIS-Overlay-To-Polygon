#!/usr/bin/env python3
# Interactive color-picker mode for GIS-Overlay-To-Polygon.
# Usage: python interactive.py <overlay.kml>
#
# Click on any colored region in the displayed image to sample its color.
# The matching mask is shown as a green overlay in real time.
# Press +/- to widen or narrow the hue tolerance.
# Press Enter to confirm and write the KML polygon.
# Press ESC to cancel.

import cv2
import numpy as np
import sys
import os

# Reuse the shared utilities from contour.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contour import extractDataFromKML, transform_coordinates, writeKML, findMaskBounds

# ── tunable defaults ─────────────────────────────────────────────────────────
THETA_DEFAULT = 15   # hue tolerance, adjustable at runtime with +/-
THETA_STEP    = 5    # how much each +/- keypress changes THETA
THETA_MIN     = 5
THETA_MAX     = 60
DELTA         = 0.02  # polygon simplification (fraction of perimeter)
AREAS         = 1     # how many disjoint regions to extract
OPACITY       = 200   # KML polygon opacity (0-255)
NEIGHBORHOOD  = 10    # px radius around click used for median color sampling
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = "Click a region | +/- = tolerance | Enter = save KML | ESC = quit"


def build_mask(blurred_hsv, bgr_pixel, theta):
    """Return a cleaned binary mask for bgr_pixel +/- theta hue."""
    sample = np.uint8([[[int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])]]])
    lo, hi = findMaskBounds(sample, theta)
    mask = cv2.inRange(blurred_hsv, lo, hi)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask


def overlay_mask(image, mask, color=(0, 220, 0), alpha=0.4):
    """Return image with mask region tinted in color."""
    preview = image.copy()
    tint = np.zeros_like(preview)
    tint[mask > 0] = color
    cv2.addWeighted(tint, alpha, preview, 1.0 - alpha, 0, preview)
    return preview


def sample_color(image, x, y, radius=NEIGHBORHOOD):
    """Return median BGR color in a small neighborhood around (x, y)."""
    h, w = image.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = image[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0).astype(np.uint8)


def contours_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, DELTA * cv2.arcLength(c, True), True) for c in contours]


def build_final_coords(contours, areas, hsv, center, bounds, rotation):
    contour_areas = sorted([(cv2.contourArea(c), i) for i, c in enumerate(contours)],
                           key=lambda x: x[0], reverse=True)
    indices = [i for _, i in contour_areas[:areas]]

    close_pts = []
    final = None
    for i in indices:
        pts = np.array(contours[i]).reshape(-1, 2)
        close_pts.append(pts[0])
        pts = np.append(pts, [pts[0]], axis=0)
        final = pts if final is None else np.append(final, pts, axis=0)

    for p in reversed(close_pts):
        final = np.append(final, [p], axis=0)
    final = np.append(final, [final[0]], axis=0)

    return transform_coordinates(final, hsv, center, bounds, rotation)


def status_bar(image, bgr, theta, picked):
    """Draw a status line at the bottom of the image."""
    out = image.copy()
    h = out.shape[0]
    if picked:
        msg = ("THETA={:d}  |  sampled BGR=({:d},{:d},{:d})  |  +/- adjust  Enter=save  ESC=quit"
               .format(theta, int(bgr[0]), int(bgr[1]), int(bgr[2])))
        swatch = tuple(int(v) for v in bgr)
    else:
        msg = "Click on a colored region to start  |  ESC=quit"
        swatch = None
    cv2.rectangle(out, (0, h - 26), (out.shape[1], h), (30, 30, 30), -1)
    cv2.putText(out, msg, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                cv2.LINE_AA)
    if swatch:
        cv2.rectangle(out, (out.shape[1] - 36, h - 22), (out.shape[1] - 6, h - 4), swatch, -1)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: interactive.py <overlay.kml>", file=sys.stderr)
        sys.exit(1)

    input_kml = sys.argv[1]
    base_path = os.path.dirname(input_kml) or '.'
    name, image_name, n, s, e, w, r = extractDataFromKML(input_kml)
    center = ((w + e) / 2, (n + s) / 2)
    bounds = (e - w, n - s)
    image_path = "{}/{}".format(base_path, image_name)
    output_kml = "poly-{}.kml".format(name)

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image: {}".format(image_path), file=sys.stderr)
        sys.exit(1)

    # Pre-blur once; reused for every mask rebuild
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred_hsv = cv2.GaussianBlur(hsv, (31, 31), 0)

    state = dict(bgr=np.array([0, 0, 0]), theta=THETA_DEFAULT, mask=None, picked=False)

    def redraw():
        if state['picked']:
            preview = overlay_mask(image, state['mask'])
        else:
            preview = image.copy()
        frame = status_bar(preview, state['bgr'], state['theta'], state['picked'])
        cv2.imshow(WINDOW, frame)

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state['bgr'] = sample_color(image, x, y)
        state['mask'] = build_mask(blurred_hsv, state['bgr'], state['theta'])
        state['picked'] = True
        b, g, r_ch = state['bgr']
        print("Sampled BGR=({},{},{})  THETA={}  -- press Enter to save, +/- to adjust"
              .format(int(b), int(g), int(r_ch), state['theta']))
        redraw()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()
    print("Window open. Click a colored region. +/- to adjust tolerance. Enter=save. ESC=quit.")

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("Cancelled.")
            break

        elif key in (13, 10):  # Enter
            if not state['picked']:
                print("Click a region first.")
                continue
            contours = contours_from_mask(state['mask'])
            if not contours:
                print("No contours found -- try clicking a different spot or pressing + to widen tolerance.")
                continue
            coords = build_final_coords(contours, AREAS, hsv, center, bounds, r)
            bgr = state['bgr']
            kml_color = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
            writeKML(output_kml, coords, kml_color)
            print("Saved polygon to {}".format(output_kml))
            break

        elif key in (ord('+'), ord('=')):  # + or = (same key without shift)
            state['theta'] = min(THETA_MAX, state['theta'] + THETA_STEP)
            print("THETA -> {}".format(state['theta']))
            if state['picked']:
                state['mask'] = build_mask(blurred_hsv, state['bgr'], state['theta'])
            redraw()

        elif key == ord('-'):
            state['theta'] = max(THETA_MIN, state['theta'] - THETA_STEP)
            print("THETA -> {}".format(state['theta']))
            if state['picked']:
                state['mask'] = build_mask(blurred_hsv, state['bgr'], state['theta'])
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
