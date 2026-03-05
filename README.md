# GIS Overlay To Polygon

Takes a georeferenced image overlay (`.kml` + image file) and extracts colored regions from it as KML polygons. Useful for converting hand-drawn or scanned map overlays into usable GIS polygon data.

Two modes are available:

| Script | Mode |
|--------|------|
| `contour.py` | Headless — specify a color in code, run, get KML |
| `interactive.py` | GUI — click regions in the image to select them, then save |

---

## Requirements

```
pip install opencv-python numpy beautifulsoup4 lxml matplotlib
```

---

## `contour.py` — Headless Mode

### Usage

```bash
python contour.py <overlay.kml>
```

The script reads the KML to find the image path and geographic bounds, detects pixels matching `RGBCOLOR`, and writes `poly-<name>.kml`.

### Configuration (top of file)

| Variable | Default | Description |
|----------|---------|-------------|
| `RGBCOLOR` | `[0, 0, 200]` | Target color in **BGR** order (OpenCV convention). Edit this to match the color of the region you want to extract. |
| `THETA` | `15` | Hue tolerance in HSV degrees. Increase if the mask misses pixels; decrease if it bleeds into neighboring colors. |
| `AREAS` | `1` | Number of disjoint regions to extract (largest N by area). |
| `DELTA` | `0.02` | Polygon simplification factor (fraction of contour perimeter). Lower = more detail, higher = smoother. |
| `OPACITY` | `200` | KML polygon fill opacity, 0–255 (~78% at 200). |
| `SAVE_IMAGE` | `False` | Save the binary mask as a `.png` alongside the KML. |
| `PREVIEW_MASK` | `False` | Open a matplotlib window showing the detected region before writing KML. |
| `PREVIEW_MASK_ON_IMAGE` | `True` | When previewing, show the mask overlaid on the original image instead of alone. |
| `QUIT_UPON_PREVIEW` | `True` | Exit after the preview window closes instead of continuing to write KML. |

### How semi-transparent overlays are handled

Map overlays are often semi-transparent, so the underlying roads, labels, and terrain bleed through and break up the color mask. Three steps counteract this:

1. **Gaussian blur (31×31)** — smooths out fine map details (roads, text) before thresholding, so they don't punch holes in the mask.
2. **Two-pass morphological closing** — a 25×25 pass fills small gaps (thin roads, text characters); a 75×75 pass merges larger disconnected blobs left by highways or big labels.
3. **Lowered saturation/value floors (20)** — blended pixels have lower saturation than pure overlay pixels. The floor of 20 (down from the original 50) captures these blended areas.

---

## `interactive.py` — GUI Mode

No need to know the color in advance. Open the image, click on a region, see the mask live, adjust until it looks right, and save.

### Usage

```bash
python interactive.py <overlay.kml>
```

### Controls

| Key / Action | Effect |
|---|---|
| **Left-click** | Add the clicked region to your selection. Each region gets its own highlight color. |
| `+` or `=` | Widen hue tolerance (THETA +5) for the **last** added region — mask expands. |
| `-` | Narrow hue tolerance (THETA -5) for the **last** added region — mask shrinks. |
| `z` | Undo — remove the last selected region. |
| `c` | Clear all selections. |
| **Enter** | Save all selected regions and exit. |
| **ESC** | Cancel without saving. |

### Output

Pressing Enter writes two files:

- **`poly-<name>.kml`** — one `<Placemark>` per selected region, each styled with its sampled color.
- **`poly-<name>.png`** — mask image with each region filled in its own distinct color (green, orange, magenta, cyan, …) on a black background, matching what you saw in the live preview.

### How color sampling works

Clicking samples the **median BGR color** over a 10×10 pixel neighborhood around the cursor. The median avoids being thrown off by a single road pixel or label character that happens to be under the cursor. The same blur + morphological closing pipeline from `contour.py` is then applied to produce a clean mask.

---

## Notes

- **Rotation**: The rotation field in the KML is supported but is slightly off due to a known approximation in the coordinate transform. Overlays without rotation produce the most accurate results.
- **Multiple disjoint areas**: In `contour.py`, set `AREAS > 1` to capture more than one disconnected blob of the same color. In `interactive.py`, just click each region separately.
- **Output format**: Both scripts write Google Earth–compatible KML (tested with Google Earth Pro).
