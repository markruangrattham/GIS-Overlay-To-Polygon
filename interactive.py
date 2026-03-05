#!/usr/bin/env python3
# Interactive color-picker mode for GIS-Overlay-To-Polygon.
# Usage: python interactive.py <overlay.kml>
#
# Left-click any colored region to add it to your selection.
# Each region gets its own highlight color so you can tell them apart.
# Press +/- to adjust the hue tolerance of the LAST selected region.
# Press z to undo the last selection, c to clear all.
# Press Enter to save all selections as separate placemarks in one KML.
# Press ESC to cancel without saving.

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contour import extractDataFromKML, transform_coordinates, findMaskBounds
from bs4 import BeautifulSoup

# ── tunable defaults ─────────────────────────────────────────────────────────
THETA_DEFAULT = 15
THETA_STEP    = 5
THETA_MIN     = 5
THETA_MAX     = 60
DELTA         = 0.02   # polygon simplification (fraction of perimeter)
AREAS         = 1      # largest N contours kept per selection
OPACITY       = 200    # KML polygon opacity (0-255)
NEIGHBORHOOD  = 10     # px radius for median color sampling
# ─────────────────────────────────────────────────────────────────────────────

WINDOW = "Left-click=add  +/-=tolerance  z=undo  c=clear  Enter=save  ESC=quit"

# Distinct BGR tints cycled across selections so they're visually separable
TINT_COLORS = [
    (0,  220,  0),    # green
    (0,  140, 255),   # orange
    (220,  0, 220),   # magenta
    (0,  220, 220),   # yellow
    (255,  80,  0),   # blue
    (0,   80, 200),   # red-orange
    (200, 200,  0),   # teal
]

GOOGLE_KML_MULTI_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<kml xmlns="http://www.opengis.net/kml/2.2">'
    '<Document>'
    '<name></name>'
    '</Document>'
    '</kml>'
)


# ── image processing helpers ─────────────────────────────────────────────────

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
    ranked = sorted([(cv2.contourArea(c), i) for i, c in enumerate(contours)],
                    key=lambda x: x[0], reverse=True)
    indices = [i for _, i in ranked[:areas]]

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


# ── KML writer for multiple polygons ─────────────────────────────────────────

def write_multi_kml(filename, doc_name, polygons):
    """
    Write a KML file with one <Placemark> per polygon.

    polygons: list of dicts with keys:
        'name'   – placemark label
        'coords' – array of (lon, lat) pairs
        'bgr'    – np.uint8 [[[B, G, R]]] color array
    """
    soup = BeautifulSoup(GOOGLE_KML_MULTI_TEMPLATE, 'xml')
    soup.find('name').string = doc_name

    doc = soup.find('Document')
    altitude = 0

    for poly in polygons:
        bgr = poly['bgr']
        color_hex = "{:02x}{:02x}{:02x}{:02x}".format(
            OPACITY, int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2]))

        style_id = "style_{}".format(poly['name'].replace(' ', '_'))

        style_tag = BeautifulSoup(
            '<Style id="{sid}">'
            '  <LineStyle><color>{c}</color></LineStyle>'
            '  <PolyStyle><color>{c}</color></PolyStyle>'
            '</Style>'.format(sid=style_id, c=color_hex), 'xml').find('Style')
        doc.append(style_tag)

        coord_str = '\n\t'.join(
            ','.join(map(str, list(pair) + [altitude])) for pair in poly['coords'])

        placemark_tag = BeautifulSoup(
            '<Placemark>'
            '  <name>{n}</name>'
            '  <styleUrl>#{sid}</styleUrl>'
            '  <Polygon>'
            '    <tessellate>1</tessellate>'
            '    <outerBoundaryIs>'
            '      <LinearRing>'
            '        <coordinates>{coords}</coordinates>'
            '      </LinearRing>'
            '    </outerBoundaryIs>'
            '  </Polygon>'
            '</Placemark>'.format(n=poly['name'], sid=style_id, coords=coord_str),
            'xml').find('Placemark')
        doc.append(placemark_tag)

    print("Writing {} polygon(s) to {}".format(len(polygons), filename))
    with open(filename, 'w') as f:
        f.write(soup.prettify().replace('kml:', ''))


# ── display helpers ───────────────────────────────────────────────────────────

def render_frame(image, selections, current_theta):
    """Composite all selection masks onto the image and draw status bar."""
    preview = image.copy()

    for idx, sel in enumerate(selections):
        tint_color = TINT_COLORS[idx % len(TINT_COLORS)]
        tint = np.zeros_like(preview)
        tint[sel['mask'] > 0] = tint_color
        cv2.addWeighted(tint, 0.4, preview, 0.6, 0, preview)

    h, w = preview.shape[:2]
    n = len(selections)
    if n == 0:
        msg = "Left-click a colored region to add it  |  ESC=quit"
        swatch = None
    else:
        last = selections[-1]
        b, g, rc = int(last['bgr'][0]), int(last['bgr'][1]), int(last['bgr'][2])
        msg = ("{} region(s)  |  last BGR=({},{},{})  THETA={}  |  "
               "+/-=adjust last  z=undo  c=clear  Enter=save  ESC=quit"
               .format(n, b, g, rc, current_theta))
        swatch = (b, g, rc)

    cv2.rectangle(preview, (0, h - 26), (w, h), (30, 30, 30), -1)
    cv2.putText(preview, msg, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)
    if swatch:
        cv2.rectangle(preview, (w - 36, h - 22), (w - 6, h - 4), swatch, -1)

    return preview


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: interactive.py <overlay.kml>", file=sys.stderr)
        sys.exit(1)

    input_kml = sys.argv[1]
    base_path = os.path.dirname(input_kml) or '.'
    name, image_name, n_lat, s_lat, e_lon, w_lon, rotation = extractDataFromKML(input_kml)
    center = ((w_lon + e_lon) / 2, (n_lat + s_lat) / 2)
    bounds = (e_lon - w_lon, n_lat - s_lat)
    image_path = "{}/{}".format(base_path, image_name)
    output_kml = "poly-{}.kml".format(name)

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image: {}".format(image_path), file=sys.stderr)
        sys.exit(1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred_hsv = cv2.GaussianBlur(hsv, (31, 31), 0)

    # selections: list of {bgr, theta, mask}
    selections = []
    current_theta = THETA_DEFAULT

    def redraw():
        cv2.imshow(WINDOW, render_frame(image, selections, current_theta))

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bgr = sample_color(image, x, y)
        mask = build_mask(blurred_hsv, bgr, current_theta)
        selections.append({'bgr': bgr, 'theta': current_theta, 'mask': mask})
        b, g, rc = int(bgr[0]), int(bgr[1]), int(bgr[2])
        print("[{}] Added region  BGR=({},{},{})  THETA={}  |  z=undo  Enter=save"
              .format(len(selections), b, g, rc, current_theta))
        redraw()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()
    print("Left-click regions to select them. +/- adjusts last. z=undo. c=clear. Enter=save. ESC=quit.")

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("Cancelled.")
            break

        elif key in (13, 10):  # Enter — save all selections
            if not selections:
                print("No regions selected yet.")
                continue

            polygons = []
            for idx, sel in enumerate(selections):
                contours = contours_from_mask(sel['mask'])
                if not contours:
                    print("Region {} has no contours, skipping.".format(idx + 1))
                    continue
                coords = build_final_coords(contours, AREAS, hsv, center, bounds, rotation)
                bgr = sel['bgr']
                polygons.append({
                    'name': 'region-{}'.format(idx + 1),
                    'coords': coords,
                    'bgr': np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]]),
                })

            if not polygons:
                print("No valid polygons to save.")
                continue

            write_multi_kml(output_kml, name, polygons)

            # Save a PNG with each region filled in its own distinct color
            # (the same tint palette used in the live preview).
            output_png = output_kml.rsplit('.', 1)[0] + ".png"
            mask_image = np.zeros_like(image)
            for idx, sel in enumerate(selections):
                color = TINT_COLORS[idx % len(TINT_COLORS)]
                mask_image[sel['mask'] > 0] = color
            cv2.imwrite(output_png, mask_image)
            print("Saved mask image to {}".format(output_png))
            break

        elif key in (ord('+'), ord('=')):
            current_theta = min(THETA_MAX, current_theta + THETA_STEP)
            print("THETA -> {}".format(current_theta))
            if selections:
                sel = selections[-1]
                sel['theta'] = current_theta
                sel['mask'] = build_mask(blurred_hsv, sel['bgr'], current_theta)
            redraw()

        elif key == ord('-'):
            current_theta = max(THETA_MIN, current_theta - THETA_STEP)
            print("THETA -> {}".format(current_theta))
            if selections:
                sel = selections[-1]
                sel['theta'] = current_theta
                sel['mask'] = build_mask(blurred_hsv, sel['bgr'], current_theta)
            redraw()

        elif key == ord('z'):  # undo last selection
            if selections:
                removed = selections.pop()
                b, g, rc = int(removed['bgr'][0]), int(removed['bgr'][1]), int(removed['bgr'][2])
                print("Removed last region (BGR={},{},{}) — {} remaining"
                      .format(b, g, rc, len(selections)))
                redraw()
            else:
                print("Nothing to undo.")

        elif key == ord('c'):  # clear all
            selections.clear()
            print("Cleared all selections.")
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
